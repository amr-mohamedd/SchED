#!/usr/bin/env bash

set -euo pipefail

# 1) Dream base + SchED cosine on MMLU (5-shot)
MODEL_ARGS_MMLU="pretrained=Dream-org/Dream-v0-Base-7B,add_bos_token=true,diffusion_steps=5,max_new_tokens=5,mc_num=5,parallelize=true,enable_early_exit=true,alg=schedule,tau_mode=cosine,tau_high=7.5,tau_low=2.5,tau_k=4.0,stat=mean,answer_region=all,min_progress=0.0,average_over_masked=true,max_change_ratio=0.0"
python -m lm_eval \
  --model dream \
  --model_args "$MODEL_ARGS_MMLU" \
  --tasks mmlu_generative \
  --num_fewshot 5 \
  --batch_size 1 \
  --output_path ./results_dream/mmlu_generative/base/schedule__cosine \
  -L 100 \
  --log_samples \
  --confirm_run_unsafe_code

# 2) Dream Instruct (DiffLLM) + SchED exp (k=4) on GPQA main CoT (8-shot)
MODEL_ARGS_GPQA="pretrained=Dream-org/Dream-v0-Instruct-7B,add_bos_token=true,diffusion_steps=128,max_new_tokens=128,mc_num=128,parallelize=true,enable_early_exit=true,alg=schedule,tau_mode=exp,tau_high=7.5,tau_low=2.5,tau_k=4.0,stat=mean,answer_region=all,min_progress=0.0,average_over_masked=true,max_change_ratio=0.0,apply_chat_template=true"
python -m lm_eval \
  --model diffllm \
  --model_args "$MODEL_ARGS_GPQA" \
  --tasks gpqa_main_cot_n_shot \
  --num_fewshot 8 \
  --batch_size 1 \
  --output_path ./results_dream/gpqa_main_cot_n_shot/instruct/schedule__exp_k4 \
  -L 100 \
  --log_samples \
  --confirm_run_unsafe_code

# 3) Dream base + SchED cosine on Hellaswag (0-shot multiple-choice sanity)
MODEL_ARGS_HELLASWAG="pretrained=Dream-org/Dream-v0-Base-7B,add_bos_token=true,diffusion_steps=5,max_new_tokens=5,mc_num=5,parallelize=true,enable_early_exit=true,alg=schedule,tau_mode=cosine,tau_high=7.5,tau_low=2.5,tau_k=4.0,stat=mean,answer_region=all,min_progress=0.0,average_over_masked=true,max_change_ratio=0.0"
python -m lm_eval \
  --model dream \
  --model_args "$MODEL_ARGS_HELLASWAG" \
  --tasks hellaswag_generative \
  --num_fewshot 0 \
  --batch_size 1 \
   -L 100 \
  --confirm_run_unsafe_code
