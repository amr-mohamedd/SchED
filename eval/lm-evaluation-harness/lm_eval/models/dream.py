# lm_eval/models/dream.py
from __future__ import annotations
import json
import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from datasets import Dataset
from packaging import version
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate

# NEW: import our early-exit samplers
from dream_sampling.prophet import diffusion_generate_prophet
from dream_sampling.sched import diffusion_generate_schedule


eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


@register_model("dream")
class Dream(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 2048,
        add_bos_token: Optional[bool] = False,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        mc_num: Optional[int] = 128,
        classifier_free_guidance: Optional[float] = 1.0,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 128,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,  # ← int, not float
        alg: Optional[str] = "entropy",  # when set to 'prophet' or 'schedule' we use our generators
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        # --- NEW: early-exit (Prophet + Schedule) knobs ---
        # Prophet
        early_threshold: float = 7.5,
        mid_threshold: float = 5.0,
        late_threshold: float = 2.5,
        # Schedule
        tau_mode: str = "cosine",
        tau_high: float = 7.5,
        tau_low: float = 2.5,
        tau_k: float = 4.0,
        # Aggregate & gating
        stat: str = "mean",
        q: float = 0.20,
        answer_region: str = "all",
        answer_start: Optional[int] = None,
        answer_end: Optional[int] = None,
        average_over_masked: bool = True,
        min_progress: float = 0.0,
        patience_steps: int = 0,
        max_change_ratio: float = 0.0,
        # --- NEW: metadata logging ---
        meta_dir: Optional[str] = None,
        meta_filename: Optional[str] = "metadata.json",
        **kwargs,
    ) -> None:
        super().__init__()

        # --- device / accelerate setup (unchanged) ---
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        if not (parallelize or accelerator.num_processes > 1):
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(torch.__version__) < version.parse("2.1"):
                    raise RuntimeError(f"mps requires torch >= 2.1. You have {torch.__version__}")
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            if device != "cuda":
                eval_logger.info(
                    "Using accelerate launch or parallelize=True, device '{device}' will be overridden when placing model."
                )
            self._device = self.accelerator.device if hasattr(self, "accelerator") else torch.device(device)

        self.batch_size_per_gpu = int(batch_size) if isinstance(batch_size, str) else batch_size

        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. Possibly quantized or using device_map."
                        )
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "Using both a HF Accelerate device_map and accelerate launch; will attempt model+data parallelism."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: System GPUs != spawned processes. Launch with accelerate launch to use data parallelism."
                        )
                    if self.accelerator.is_local_main_process:
                        eval_logger.info(f"Using {gpus} devices with data parallelism")
                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator
                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    self._rank = 0
                    self._world_size = 1
            else:
                eval_logger.warning(
                    "Passed a pre-initialized model; assuming single-process evaluate() or custom distributed setup"
                )
                self._rank = 0
                self._world_size = 1

        # generation params
        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until

        # loglikelihood params
        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = mc_num
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps

        # NEW: store early-exit knobs
        self.early_threshold = early_threshold
        self.mid_threshold = mid_threshold
        self.late_threshold = late_threshold
        self.tau_mode = tau_mode
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.tau_k = tau_k
        self.stat = stat
        self.q = q
        self.answer_region = answer_region
        self.answer_start = answer_start
        self.answer_end = answer_end
        self.average_over_masked = average_over_masked
        self.min_progress = min_progress
        self.patience_steps = patience_steps
        self.max_change_ratio = max_change_ratio

        # --- Metadata / logging (like LLaDA) ---
        # Ensure rank/world exist in all branches
        if not hasattr(self, "_rank"):
            self._rank = 0
        if not hasattr(self, "_world_size"):
            self._world_size = 1

        self.meta_dir = Path(meta_dir or "eval_meta")
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.meta_filename = meta_filename or "metadata.json"
        self._meta_buffer: List[dict] = []
        self._sample_idx = 0

        # Run header captures fixed knobs for the whole job
        self._run_header = {
            "version": 1,
            "pretrained": getattr(self.model, "name_or_path", None) or str(pretrained),
            "add_bos_token": bool(self.add_bos_token),
            "diffusion_steps": int(self.diffusion_steps),
            "max_new_tokens": int(self.max_new_tokens),
            "mc_num": int(self.mc_num),
            "parallelize": bool(parallelize),
            "alg": str(self.alg),
            "early_thresholds": {
                "early": float(self.early_threshold),
                "mid": float(self.mid_threshold),
                "late": float(self.late_threshold),
            },
            "schedule": {
                "tau_mode": str(self.tau_mode),
                "tau_high": float(self.tau_high),
                "tau_low": float(self.tau_low),
                "tau_k": float(self.tau_k),
                "stat": str(self.stat),
                "q": float(self.q),
                "answer_region": str(self.answer_region),
                "answer_start": self.answer_start,
                "answer_end": self.answer_end,
                "average_over_masked": bool(self.average_over_masked),
                "min_progress": float(self.min_progress),
                "patience_steps": int(self.patience_steps),
                "max_change_ratio": float(self.max_change_ratio),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "world_size": int(self._world_size),
            "rank": int(self._rank),
        }

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        self.model = (
            transformers.AutoModel.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            ).eval()
        ).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        ).input_ids

    @classmethod
    def create_from_arg_string(
        cls: Type[T],
        arg_string: str,
        additional_config: Optional[dict] = None
    ) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    # ---- Metadata helpers ----
    def _normalize_stats(self, alg: str, sample_stats, default_steps: int):
        """Map various sampler stat formats to a common schema."""
        meta = {
            "committed_early": False,
            "steps_taken": int(default_steps),
            "Tmax": int(default_steps),
            "decision_step": None,
            "progress": None,
            "tau": None,
            "gbar": None,
            "changed_ratio": None,
            "block_index": None,
            "step_in_block": None,
            "inference_time": None,
            "raw": sample_stats,
        }
        if not sample_stats:
            return meta

        # Prophet-style (like LLaDA's generate_earlyexit)
        if isinstance(sample_stats, dict) and "exit_info" in sample_stats:
            ei = sample_stats["exit_info"]
            meta.update(
                committed_early=bool(ei.get("early_exit_triggered", False)),
                steps_taken=int(ei.get("actual_steps", default_steps)),
                Tmax=int(ei.get("total_steps", default_steps)),
                decision_step=ei.get("exit_decision_step"),
                inference_time=ei.get("inference_time"),
            )
            return meta

        # Schedule-style (like generate_earlyexit_schedule)
        if isinstance(sample_stats, dict) and ("steps_taken" in sample_stats or "Tmax" in sample_stats):
            meta.update(
                committed_early=bool(sample_stats.get("committed_early", False)),
                steps_taken=int(sample_stats.get("steps_taken", default_steps)),
                Tmax=int(sample_stats.get("Tmax", default_steps)),
                progress=sample_stats.get("progress"),
                tau=sample_stats.get("tau"),
                gbar=sample_stats.get("gbar"),
                changed_ratio=sample_stats.get("changed_ratio"),
                block_index=sample_stats.get("block_index"),
                step_in_block=sample_stats.get("step_in_block"),
            )
            return meta

        return meta

    def _flush_metadata(self):
        """Write the buffered samples + run header to JSON."""
        fname = f"metadata_rank_{self.rank}.json" if self.world_size > 1 else self.meta_filename
        out_path = self.meta_dir / fname
        payload = {"run": self._run_header, "samples": self._meta_buffer}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ----------------- Core generation -----------------
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        if self.add_bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]

        # left-pad to support batched decoding split
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        prompt_ids = enc.input_ids
        attn_mask = enc.attention_mask

        # sequence length check/slice BOTH ids and mask
        seq_len = prompt_ids.shape[1]
        keep = self.max_length - self.max_new_tokens
        if seq_len > keep:
            eval_logger.warning(
                f"Prompt length {seq_len} is larger than {keep}, truncating on left"
            )
            prompt_ids = prompt_ids[:, -keep:]
            attn_mask = attn_mask[:, -keep:]

        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        # Route to our early-exit samplers and request stats
        if self.alg == "prophet":
            mask_id = getattr(self.model.config, "mask_token_id", None)
            if mask_id is None:
                raise ValueError(
                    "model.config.mask_token_id is None. "
                    "Set the correct diffusion mask id in the model config."
                )

            # Call the Prophet sampler implemented in dream_sampling/prophet.py.
            # It should return either:
            #   (a) an object with `.sequences` and optional `.stats`, or
            #   (b) a tuple: (sequences, stats_list)
            generation_out = diffusion_generate_prophet(
                self.model,
                prompt_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                # phase thresholds
                early_threshold=self.early_threshold,
                mid_threshold=self.mid_threshold,
                late_threshold=self.late_threshold,
                # gating & aggregation
                min_progress=self.min_progress,
                patience_steps=self.patience_steps,
                stat=self.stat,
                q=self.q,
                answer_region=self.answer_region,
                answer_start=self.answer_start,
                answer_end=self.answer_end,
                average_over_masked=self.average_over_masked,
                max_change_ratio=self.max_change_ratio,
                # tokens
                mask_token_id=mask_id,
                # ask the sampler to return per-sample stats so our logger can record them
                return_stats=True,
            )
        elif self.alg == "schedule":
            mask_id = getattr(self.model.config, "mask_token_id", None)
            if mask_id is None:
                raise ValueError(
                    "model.config.mask_token_id is None. "
                    "Set the correct diffusion mask id in the model config."
                )
            # Use the schedule sampler (self-contained diffusion loop) and pass full knobs.
            # IMPORTANT: the sampler’s `alg` means the *selection rule* ('origin' | 'maskgit_plus' | 'topk_margin' | 'entropy').
            # Since `self.alg == "schedule"` selects this branch already, do NOT pass 'schedule' down as alg.
            generation_out = diffusion_generate_schedule(
                self.model,
                prompt_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                # selection filters / inner sampling rules
                top_p=self.top_p,
                top_k=self.top_k,
                alg="entropy",                 # keep stable; change to 'maskgit_plus'|'topk_margin'|'origin' if desired
                alg_temp=self.alg_temp,
                eps=self.sampling_eps,
                # early-exit schedule & gating
                tau_mode=self.tau_mode,
                tau_high=self.tau_high,
                tau_low=self.tau_low,
                tau_k=self.tau_k,
                min_progress=self.min_progress,
                patience_steps=self.patience_steps,
                stat=self.stat,
                q=self.q,
                answer_region=self.answer_region,
                answer_start=self.answer_start,
                answer_end=self.answer_end,
                average_over_masked=self.average_over_masked,
                max_change_ratio=self.max_change_ratio,
                # tokens / plumbing
                mask_token_id=mask_id,
                return_stats=True,             # request per-sample stats for your logger
            )
        else:
            # fallback to model-native diffusion_generate
            generation_out = self.model.diffusion_generate(
                prompt_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                alg=self.alg,
                alg_temp=self.alg_temp,
            )

        # ---- Decode & collect sequences/stats robustly ----
        stats_list = None
        if hasattr(generation_out, "sequences"):
            sequences = generation_out.sequences
            stats_list = getattr(generation_out, "stats", None)
        elif isinstance(generation_out, tuple) and len(generation_out) == 2:
            sequences, stats_list = generation_out
        else:
            # final fallback
            sequences = generation_out.sequences

        Lg = int(self.max_new_tokens)
        responses = []
        for row_idx, (p, g) in enumerate(zip(prompt_ids, sequences)):
            Lp = int(p.shape[0])
            gen_tokens = g[Lp : Lp + Lg]  # clamp strictly to generation window
            text = self.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos:
                text = text.split(eos)[0]
            responses.append(text)

            # ---- Append per-sample metadata ----
            self._sample_idx += 1
            # choose the i-th stats if we have a list, else reuse dict for all
            sample_stats = None
            if isinstance(stats_list, list):
                sample_stats = stats_list[row_idx] if row_idx < len(stats_list) else None
            else:
                sample_stats = stats_list

            norm = self._normalize_stats(self.alg, sample_stats, int(self.diffusion_steps))
            self._meta_buffer.append(
                {
                    "index": int(self._sample_idx),
                    "prompt": prompts[row_idx] if row_idx < len(prompts) else None,
                    "answer": text,
                    "alg": str(self.alg),
                    "config": {
                        "diffusion_steps": int(self.diffusion_steps),
                        "max_new_tokens": int(self.max_new_tokens),
                        "temperature": float(self.temperature or 0.0),
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                        "stat": str(self.stat),
                        "q": float(self.q),
                        "answer_region": str(self.answer_region),
                        "answer_start": self.answer_start,
                        "answer_end": self.answer_end,
                        "average_over_masked": bool(self.average_over_masked),
                        "min_progress": float(self.min_progress),
                        "patience_steps": int(self.patience_steps),
                        "max_change_ratio": float(self.max_change_ratio),
                        "thresholds": {
                            "early": float(self.early_threshold),
                            "mid": float(self.mid_threshold),
                            "late": float(self.late_threshold),
                        },
                        "schedule": {
                            "tau_mode": str(self.tau_mode),
                            "tau_high": float(self.tau_high),
                            "tau_low": float(self.tau_low),
                            "tau_k": float(self.tau_k),
                        },
                    },
                    "stats": norm,
                }
            )

        # Flush to disk for durability; cheap even if called often
        self._flush_metadata()
        return responses

    # --- loglikelihood utilities remain unchanged ---
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(contexts)
            if not self.escape_until:
                for i, r in enumerate(responses):
                    for s in gen_args[0]["until"]:
                        r = r.split(s)[0]
                    responses[i] = r
            res.extend(responses)
            pbar.update(len(contexts))
        # Ensure metadata hits disk at end of call
        self._flush_metadata()
        return res

    def _encode_pair(self, context: str, continuation: str):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
        context_enc = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        Lc = len(context_enc)
        continuation_enc = whole_enc[Lc:]
        return context_enc, continuation_enc

    def _eval_target_nll_mc(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        # Placeholder for your model's MC NLL implementation; retain your original logic here.
        raise NotImplementedError("MC NLL path not wired in this drop-in. Keep your existing implementation.")

    def _eval_target_nll_ar(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        # Placeholder for your model's AR NLL implementation; retain your original logic here.
        raise NotImplementedError("AR NLL path not wired in this drop-in. Keep your existing implementation.")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        from datasets import Dataset  # local import to avoid issues if datasets isn't needed elsewhere

        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out: List[Tuple[float, bool]] = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                if self.nll_type == "mc":
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == "union":
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type in ("ar_ftb", "ar_btf"):
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)
                is_target_greedy_dec = False  # TODO: add greedy check if needed
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError("Dream.loglikelihood_rolling is not implemented for this model.")


if __name__ == "__main__":
    cli_evaluate()
