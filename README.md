# Fast-Decoding Diffusion Language Models via Progress‑Aware Confidence Schedules (SchED)

This repository contains the code accompanying the research paper "Fast Decoding Diffusion Language Models via Progress‑Aware Confidence Schedules". SchED is a training‑free, model‑agnostic early‑exit mechanism for diffusion language models. It accelerates decoding by monitoring an aggregated confidence signal and stopping when a smooth, progress‑dependent threshold is met.

Repository structure
- `dream_sampling/`: Schedule‑based (`sched.py`) early‑exit sampler for Dream‑style diffusion LMs.
- `eval/`: Bundled fork of lm‑evaluation‑harness with examples/configs.
  - `eval/examples/eval_dream.sh`: Example runs for Dream Base/Instruct with SchED.
  - `eval/configs/*`: Task defaults (step budgets, answer length, n-shots).

Installation
- Prerequisites
  - Python 3.9+
  - PyTorch 2.1+
- Steps
  - Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
  - Install dependencies (includes the local evaluation harness): `pip install -r requirements.txt`


Example: Schedule‑based generation
```python
import time
import torch
from transformers import AutoModel, AutoTokenizer
from dream_sampling.sched import diffusion_generate_schedule

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model_name = "Dream-org/Dream-v0-Instruct-7B"
  model = AutoModel.from_pretrained(
      model_name,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  model = model.to(device).eval()

  chat = [{"role": "user", "content": "What is the capital of France?"}]
  prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

  enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
  input_ids = enc.input_ids.to(device)
  attention_mask = enc.attention_mask.to(device) if "attention_mask" in enc else None

  steps = 128
  max_new_tokens = 128

  torch.set_grad_enabled(False)

  # 1) Plain diffusion_generate
  t0 = time.time()
  sequences_plain = model.diffusion_generate(
      input_ids,
      attention_mask=attention_mask,
      max_new_tokens=max_new_tokens,
      steps=steps,
      mask_token_id=model.config.mask_token_id,
  )
  if device == "cuda":
      torch.cuda.synchronize()
  t1 = time.time()
  elapsed_plain = t1 - t0

  # 2) Schedule-based generation
  t0 = time.time()
  sequences_sched, stats_sched = diffusion_generate_schedule(
      model,
      input_ids=input_ids,
      attention_mask=attention_mask,
      max_new_tokens=max_new_tokens,
      steps=steps,
      tau_mode="exp",
      tau_high=7.5,
      tau_low=0.0,
      tau_k=2.0,
      answer_region="all",
      stat="mean",
      mask_token_id=model.config.mask_token_id,
      return_stats=True,
  )
  if device == "cuda":
      torch.cuda.synchronize()
  t1 = time.time()
  elapsed_sched = t1 - t0

  prompt_len = input_ids.shape[1]

  gen_plain = sequences_plain[:, prompt_len:]
  text_plain = tokenizer.batch_decode(gen_plain, skip_special_tokens=True)[0].strip()

  gen_sched = sequences_sched[:, prompt_len:]
  text_sched = tokenizer.batch_decode(gen_sched, skip_special_tokens=True)[0].strip()

  print(f"\nElapsed (diffusion_generate):          {elapsed_plain:.3f} s")
  print(f"Elapsed (diffusion_generate_schedule): {elapsed_sched:.3f} s\n")
  print("Answer (plain):   ", text_plain)
  print("Answer (schedule):", text_sched)
  print("Stats (schedule):", stats_sched[0])
```

Key SchED arguments (`--model_args`)
- `alg=schedule`: select SchED (use `alg=schedule` for the phase baseline).
- `tau_mode`: `cosine` | `linear` | `exp` (progress‑aware threshold family).
- `tau_high`, `tau_low`, `tau_k`: schedule parameters (logit‑margin units).
- `stat`: aggregate over the answer region (`mean`, `median`, `quantile`, `min`).
- `answer_region`: region to aggregate confidence over (`all`, `last`, custom span).
- `min_progress`, `patience_steps`, `max_change_ratio`: stability guards for early exit.

Task defaults and reproducibility
- Step budgets, generation lengths, and few‑shot counts per task are recorded in `eval/configs`.
- Common SchED settings used in the paper include:
  - Thresholds: `tau_high=7.5`, `tau_low` in `{2.5, 0.0}`
  - Exponential curvature: `tau_k` in `{2, 4, 8, 16}`
  - Aggregation: `stat=mean`, `answer_region=all`, `average_over_masked=true`

Citation

If you find this useful, please cite:
```
@misc{mohamed2025fastdecodingdiffusionlanguagemodels,
      title={Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules}, 
      author={Amr Mohamed and Yang Zhang and Michalis Vazirgiannis and Guokan Shang},
      year={2025},
      eprint={2512.02892},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.02892}, 
}
```
