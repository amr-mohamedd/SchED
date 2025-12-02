# lm_eval/models/diffllm.py
from __future__ import annotations
import json
import logging
import gc
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, get_dtype

# NEW: import our early-exit samplers
from dream_sampling.prophet import diffusion_generate_prophet
from dream_sampling.sched import diffusion_generate_schedule

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def empty_cache_by_memory(threshold_gb: float = 70):
    """
    Empty CUDA cache if allocated memory exceeds threshold.
    Args:
        threshold_gb: Memory threshold in GB.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        if allocated > threshold_gb:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Cache cleared. Memory freed: {allocated:.2f} GB")


@register_model("diffllm")
class DiffLLM(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_prompt_len: Optional[int] = 1024,
        max_new_tokens: Optional[int] = 128,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        classifier_free_guidance: Optional[float] = 1.0,
        pad_to_max_len: Optional[bool] = False,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 32,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        # -------- Generation knobs (kept from original) --------
        **kwargs,
    ) -> None:
        super().__init__()

        # prepare for parallelism
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

        # using one process with no model parallelism
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
            self._rank = 0
            self._world_size = 1
        else:  # Parallelism managed by accelerate OR device_map usage
            if device != "cuda":
                eval_logger.info(
                    "Using `accelerate launch` or `parallelize=True`, device '%s' may be overridden when placing model.",
                    device,
                )
            # Device: prefer accelerator.device if we actually have one
            if hasattr(self, "accelerator"):
                self._device = self.accelerator.device
                self._rank = self.accelerator.local_process_index
                self._world_size = self.accelerator.num_processes
            else:
                # parallelize=True but single-process (e.g., HF device_map on one GPU/CPU)
                self._device = torch.device(device)
                self._rank = 0
                self._world_size = 1


        # Batch size
        self.batch_size_per_gpu = int(batch_size) if isinstance(batch_size, str) else batch_size

        # Create model & tokenizer
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        # Try to place on device for single-process no-device_map cases
        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized "
                            "or a device_map was provided. If your desired device is active, you can ignore this."
                        )
            # If not launched with accelerate, rank/world already set above.

        # -------- Generation params (original) --------
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = kwargs.get("temperature", 0.1)
        self.top_p = kwargs.get("top_p", 0.9)
        self.top_k = kwargs.get("top_k", None)  # int or None
        self.alg = kwargs.get("alg", "entropy")
        self.alg_temp = kwargs.get("alg_temp", 0.0)
        self.add_bos_token = bool(kwargs.get("add_bos_token", False))
        self.mc_num = int(kwargs.get("mc_num", self.diffusion_steps))

        # -------- NEW: Early-exit knobs (Prophet + Schedule) --------
        # Prophet thresholds (phase-wise)
        self.early_threshold: float = kwargs.get("early_threshold", 7.5)
        self.mid_threshold: float = kwargs.get("mid_threshold", 5.0)
        self.late_threshold: float = kwargs.get("late_threshold", 2.5)

        # Schedule shape
        self.tau_mode: str = kwargs.get("tau_mode", "cosine")   # 'cosine' | 'linear' | 'exp'
        self.tau_high: float = kwargs.get("tau_high", 7.5)
        self.tau_low: float = kwargs.get("tau_low", 2.5)
        self.tau_k: float = kwargs.get("tau_k", 4.0)

        # Aggregation & gating
        self.stat: str = kwargs.get("stat", "mean")             # 'mean' | 'median' | 'q'
        self.q: float = kwargs.get("q", 0.20)                   # for quantile
        self.answer_region: str = kwargs.get("answer_region", "all")  # 'all' | 'last' | 'window' | 'span'
        self.answer_start: Optional[int] = kwargs.get("answer_start", None)
        self.answer_end: Optional[int] = kwargs.get("answer_end", None)
        self.average_over_masked: bool = kwargs.get("average_over_masked", True)
        self.min_progress: float = kwargs.get("min_progress", 0.0)
        self.patience_steps: int = kwargs.get("patience_steps", 0)
        self.max_change_ratio: float = kwargs.get("max_change_ratio", 0.0)

        # -------- Log-likelihood params (original) --------
        self.nll_type = nll_type
        self.log_type = log_type
        self.classifier_free_guidance = classifier_free_guidance
        self.pad_to_max_len = pad_to_max_len
        self.sampling_eps = sampling_eps

        # Ensure rank/world exist even if code path changes later
        if not hasattr(self, "_rank"):
            self._rank = 0
        if not hasattr(self, "_world_size"):
            self._world_size = 1

        # -------- Metadata logging (parity with Dream model) --------
        meta_dir = kwargs.get("meta_dir", None)
        meta_filename = kwargs.get("meta_filename", "metadata.json")
        self.meta_dir = Path(meta_dir or "eval_meta")
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.meta_filename = meta_filename or "metadata.json"
        self._meta_buffer: List[dict] = []
        self._sample_idx = 0
        self._run_header = {
            "version": 1,
            "pretrained": getattr(self.model, "name_or_path", None) or str(pretrained),
            "add_bos_token": bool(self.add_bos_token),
            "diffusion_steps": int(self.diffusion_steps),
            "max_new_tokens": int(self.max_new_tokens),
            "mc_num": int(self.mc_num),
            "parallelize": bool(parallelize),
            "alg": str(self.alg),
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
            )
            .eval()
        ).to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Apply tokenizer's chat template.
        """
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    # -------- Metadata helpers --------
    def _normalize_stats(self, alg: str, sample_stats, default_steps: int):
        """Normalize sampler stats into a shared schema."""
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
        """Persist run + sample metadata."""
        if self.meta_dir is None:
            return
        fname = f"metadata_rank_{self.rank}.json" if self.world_size > 1 else self.meta_filename
        out_path = self.meta_dir / fname
        payload = {"run": self._run_header, "samples": self._meta_buffer}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ----------------- Core generation -----------------
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        original_prompts = list(prompts)
        # Tokenize (left-pad to align decoding window)
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        )
        prompt_ids = enc.input_ids[:, -self.max_prompt_len:]
        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)

        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        # Route to early-exit samplers if requested
        if self.alg == "prophet":
            mask_id = getattr(self.model.config, "mask_token_id", None)
            if mask_id is None:
                raise ValueError(
                    "model.config.mask_token_id is None. "
                    "Set the correct diffusion mask id in the model config."
                )

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
                # request per-sample stats (ignored here but supported)
                return_stats=True,
            )

        elif self.alg == "schedule":
            mask_id = getattr(self.model.config, "mask_token_id", None)
            if mask_id is None:
                raise ValueError(
                    "model.config.mask_token_id is None. "
                    "Set the correct diffusion mask id in the model config."
                )

            # IMPORTANT: the sampler’s `alg` is the *selection rule* for the inner loop
            # ('origin' | 'maskgit_plus' | 'topk_margin' | 'entropy'). We keep 'entropy' by default.
            generation_out = diffusion_generate_schedule(
                self.model,
                prompt_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                # inner-loop selection filters
                top_p=self.top_p,
                top_k=self.top_k,
                alg="entropy",
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
                return_stats=True,
            )

        else:
            # Fallback to the model's native diffusion_generate
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

        # ---- Robust decode (align on prompt length; clamp to generation window) ----
        # Handle either .sequences or (sequences, stats)
        stats_list = None
        if hasattr(generation_out, "sequences"):
            sequences = generation_out.sequences
            stats_list = getattr(generation_out, "stats", None)
        elif isinstance(generation_out, tuple) and len(generation_out) >= 1:
            sequences = generation_out[0]
            if len(generation_out) > 1:
                stats_list = generation_out[1]
        else:
            sequences = generation_out  # last-ditch

        Lg = int(self.max_new_tokens)
        responses: List[str] = []
        for row_idx, (p, g) in enumerate(zip(prompt_ids, sequences)):
            Lp = int(p.shape[0])
            gen_tokens = g[Lp : Lp + Lg]
            text = self.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos:
                text = text.split(eos)[0]
            responses.append(text)

            # ---- Metadata logging per sample ----
            self._sample_idx += 1
            sample_stats = None
            if isinstance(stats_list, list):
                sample_stats = stats_list[row_idx] if row_idx < len(stats_list) else None
            else:
                sample_stats = stats_list

            normalized = self._normalize_stats(self.alg, sample_stats, int(self.diffusion_steps))
            prompt_text = original_prompts[row_idx] if row_idx < len(original_prompts) else None
            self._meta_buffer.append(
                {
                    "index": int(self._sample_idx),
                    "prompt": prompt_text,
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
                    "stats": normalized,
                }
            )

        self._flush_metadata()
        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res: List[str] = []

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(list(contexts))

            # Honor 'until' splitters
            for i, r in enumerate(responses):
                for s in gen_args[0].get("until", []):
                    r = r.split(s)[0]
                responses[i] = r

            if self.rank == 0 and responses:
                print(f"Context:\n{contexts[0]}\nResponse:\n{responses[0]}\n")

            res.extend(responses)
            pbar.update(len(contexts))

        self._flush_metadata()
        return res

    # ----------------- Likelihood utilities (unchanged) -----------------
    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        # always unmask bos and eos
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.tokenizer.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        """
        prompt_index : 1D bool tensor, length=batch.shape[1]
        """
        if self.classifier_free_guidance > 1.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.tokenizer.mask_token_id
            batch = torch.cat([batch, un_batch])

        if self.pad_to_max_len:
            raise NotImplementedError
        else:
            input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input, 'full').logits
            # since bos always unmask, the first logits will not be used
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        mc_num = self.diffusion_steps
        for _ in range(max(mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.tokenizer.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(
                logits[mask_indices], seq[mask_indices], reduction='none'
            ) / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

            del logits, loss, perturbed_seq, perturbed_seq_, p_mask, mask_indices
            empty_cache_by_memory(threshold_gb=70)

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0)  # 1*l1, 1*l2
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous()  # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous()  # l1*l1

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.tokenizer.mask_token_id

        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False

        if self.log_type == 'ftb':
            logits_index = torch.cat(
                [torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1
            )
        else:
            logits_index = torch.cat(
                [mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1
            )

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [self.tokenizer.eos_token_id]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
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

                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type in ('ar_ftb', 'ar_btf'):
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                is_target_greedy_dec = False  # TODO: add greedy check if needed
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError
