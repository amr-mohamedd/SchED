# coding=utf-8
# dream_sampling/prophet.py

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ---------------------- Sampler helpers (mirroring Dream) ----------------------
def _top_p_logits(logits: torch.Tensor, top_p: Optional[float] = None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k = int(min(top_k, logits.size(-1)))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def _sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      confidence: (K,)  (scalar per row)
      x0:         (K,)  sampled (or greedy) token ids
    """
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        eps = 1e-10
        log_probs = torch.log(probs + eps)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


# ---------------------- Confidence & region helpers ----------------------
def _region_bounds(
    *,
    seq_len: int,
    max_new_tokens: Optional[int],
    answer_region: str,
    answer_start: Optional[int],
    answer_end: Optional[int],
) -> Tuple[int, int]:
    """
    Region of interest in [0, seq_len):
      - 'all' : [0, seq_len)
      - 'tail'/'last': last max_new_tokens
      - custom span via [answer_start, answer_end)
    """
    if answer_region in ("tail", "last"):
        if max_new_tokens is None:
            return 0, seq_len
        start = max(0, seq_len - max_new_tokens)
        return start, seq_len

    if answer_start is not None or answer_end is not None:
        s = 0 if answer_start is None else max(0, int(answer_start))
        e = seq_len if answer_end is None else min(seq_len, int(answer_end))
        s = min(s, seq_len)
        e = max(s, e)
        return s, e

    return 0, seq_len  # 'all'


def _aggregate(values: torch.Tensor, stat: str, q: float) -> torch.Tensor:
    """
    Aggregate a 1D tensor according to stat: 'mean' | 'median' | 'q'/'quantile'/'percentile'
    """
    if values.numel() == 0:
        return torch.tensor(float("-inf"), dtype=values.dtype, device=values.device)
    s = stat.lower()
    if s == "mean":
        return values.mean()
    if s == "median":
        return values.median()
    if s in ("q", "quantile", "percentile"):
        q = float(min(1.0, max(0.0, q)))
        return torch.quantile(values, q)
    return values.mean()


def _logit_margins(logits_flat: torch.Tensor) -> torch.Tensor:
    """Per-row margin = top1_logit - top2_logit."""
    top2 = torch.topk(logits_flat, k=2, dim=-1).values
    margins = top2[..., 0] - top2[..., 1]
    return margins.reshape(-1)


def _prophet_phase_threshold(progress: float, early: float, mid: float, late: float) -> float:
    """
    Piecewise-constant thresholds (Prophet):
      - early phase:   [0, 1/3)  -> early
      - mid phase:     [1/3, 2/3)-> mid
      - late phase:    [2/3, 1]  -> late
    """
    if progress < (1.0 / 3.0):
        return float(early)
    if progress < (2.0 / 3.0):
        return float(mid)
    return float(late)


# ---------------------- Main generator ----------------------
@torch.no_grad()
def diffusion_generate_prophet(
    model,
    input_ids: torch.LongTensor,
    *,
    attention_mask: Optional[torch.LongTensor] = None,
    max_new_tokens: Optional[int] = None,
    steps: int = 128,
    temperature: float = 0.0,
    # Prophet thresholds (logit-margin units)
    early_threshold: float = 7.5,
    mid_threshold: float = 5.0,
    late_threshold: float = 2.5,
    # (optional) sampling filters to mirror Dream; default keep off for stability
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    alg: str = "entropy",                # 'origin' | 'maskgit_plus' | 'topk_margin' | 'entropy'
    alg_temp: Optional[float] = None,
    eps: float = 1e-3,
    # gating/aggregation knobs (kept for symmetry with schedule sampler)
    min_progress: float = 0.0,
    patience_steps: int = 0,
    stat: str = "mean",
    q: float = 0.20,
    answer_region: str = "all",          # 'all' | 'tail'/'last' | custom via start/end
    answer_start: Optional[int] = None,
    answer_end: Optional[int] = None,
    average_over_masked: bool = True,
    max_change_ratio: float = 0.0,
    # tokens / config
    mask_token_id: Optional[int] = None,
    # output control
    return_stats: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, Any]]]]:
    """
    Self-contained generator with Prophet-style (phase) early-exit.
    Mirrors Dream's inner loop and adds an early-exit condition per step.

    Returns:
      - if return_stats=False: sequences tensor
      - if return_stats=True: (sequences, stats_list), where each stats dict has an
        "exit_info" key to match Dream._normalize_stats('prophet').
    """
    if mask_token_id is None:
        raise ValueError(
            "diffusion_generate_prophet: mask_token_id must be provided (model.config.mask_token_id)."
        )

    device = getattr(model, "device", input_ids.device)

    if max_new_tokens is None or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")
    B, Lp = input_ids.shape
    max_length = Lp + int(max_new_tokens)

    # Pad working sequence with [MASK] to target length
    x = F.pad(input_ids.to(device), (0, max_length - Lp), value=int(mask_token_id))

    # Attention mask handling (mirrors Dream)
    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask = attention_mask.to(device)
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        # broadcast to [B, 1, N, N]
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attention_mask = "full"

    # Time steps (needed for Dream's per-step transfer schedule)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    # Early-exit gating state
    patience_counter = 0
    prev_x0 = None  # previous step's argmax logits (for stability)
    steps_taken_val: int = 0
    early_exit_triggered: bool = False
    last_progress: Optional[float] = None
    last_tau: Optional[float] = None
    last_gbar: Optional[float] = None
    last_change_ratio: Optional[float] = None
    exit_decision_step: Optional[int] = None

    # Clamp bounds/params
    total_steps = int(max(1, steps))
    min_progress = float(min(1.0, max(0.0, min_progress)))
    max_change_ratio = float(min(1.0, max(0.0, max_change_ratio)))
    q = float(q)

    # For metadata timings
    t0 = time.time()

    # Main reverse loop
    for i in range(total_steps):
        mask_index = (x == mask_token_id)

        # Forward pass; Dream shifts logits right by 1 for next-token effect
        logits = model(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Progress within region: fraction of committed tokens
        N = x.shape[1]
        r_start, r_end = _region_bounds(
            seq_len=N,
            max_new_tokens=max_new_tokens,
            answer_region=answer_region,
            answer_start=answer_start,
            answer_end=answer_end,
        )
        region_len = max(0, r_end - r_start)
        if region_len == 0:
            patience_counter = 0
            # No region defined -> fall through to standard transfer
        else:
            region_mask = torch.zeros_like(mask_index, dtype=torch.bool, device=device)
            region_mask[:, r_start:r_end] = True
            masked_region = mask_index & region_mask

            total_region_positions = B * region_len
            remaining = masked_region.sum()
            if remaining.item() == 0:
                # No masked tokens remain in region -> converged
                early_exit_triggered = True
                steps_taken_val = i + 1
                exit_decision_step = i + 1
                last_progress = 1.0
                last_tau = None
                last_gbar = float("inf")
                last_change_ratio = 0.0
                # Commit everything that might still be masked anywhere
                x0_now = torch.argmax(logits, dim=-1)
                x[mask_index] = x0_now[mask_index]
                break

            progress = 1.0 - (remaining.float() / max(1, total_region_positions))
            last_progress = float(progress.item())

            # Stability: how much predictions changed on eligible (region & masked)
            x0_now = torch.argmax(logits, dim=-1)
            eligible = masked_region
            changed_ratio = 1.0
            if prev_x0 is not None and torch.any(eligible):
                diff = (x0_now != prev_x0)
                changed_ratio = float(diff[eligible].float().mean().item())
            last_change_ratio = changed_ratio
            prev_x0 = x0_now

            if last_progress >= min_progress and changed_ratio <= max_change_ratio:
                patience_counter += 1
            else:
                patience_counter = 0

            # Confidence metric: logit margin aggregated over region
            if average_over_masked:
                focus_idx = masked_region
                if not torch.any(focus_idx):
                    # Defensive: if nothing eligible, we can stop
                    early_exit_triggered = True
                    steps_taken_val = i + 1
                    exit_decision_step = i + 1
                    last_tau = None
                    last_gbar = float("inf")
                    last_change_ratio = 0.0
                    x[mask_index] = x0_now[mask_index]
                    break
                focus_logits = logits[focus_idx]  # [K, V]
                margins = _logit_margins(focus_logits)  # [K]
            else:
                region_logits = logits[:, r_start:r_end, :]  # [B, Lr, V]
                margins = _logit_margins(region_logits.reshape(-1, region_logits.size(-1)))  # [B*Lr]

            gbar = _aggregate(margins, stat=stat, q=q)
            last_gbar = float(gbar.item())

            # Prophet phase threshold τ_phase based on overall step progress
            # progress for phase partition: use (i+1)/total_steps in [0,1]
            p_phase = (i + 1) / float(total_steps)
            tau = _prophet_phase_threshold(
                p_phase, early=early_threshold, mid=mid_threshold, late=late_threshold
            )
            last_tau = float(tau)

            # Early-exit condition with patience
            if (last_progress >= min_progress) and (last_gbar >= last_tau) and (patience_counter > patience_steps):
                early_exit_triggered = True
                steps_taken_val = i + 1
                exit_decision_step = i + 1
                # Commit all remaining masks using current predictions
                x[mask_index] = x0_now[mask_index]
                break

        # -------------- Standard Dream token transfer for this step --------------
        # We follow the same update as Dream._sample (per-step transfer rate).
        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == "origin":
            p_transfer = 1 - s / t if i < total_steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + int(mask_token_id)
            transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
            _, x0[transfer_index_t_s] = _sample_tokens(
                mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k
            )
            x[mask_index] = x0.clone()
        else:
            if alg == "maskgit_plus":
                confidence, x0 = _sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            elif alg == "topk_margin":
                confidence, x0 = _sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True
                )
            elif alg == "entropy":
                confidence, x0 = _sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, neg_entropy=True
                )
            else:
                raise RuntimeError(f"Unknown alg: {alg}")

            # Decide how many tokens to transfer this step
            num_mask_token = mask_index.sum() / mask_index.shape[0]  # average per batch item
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < total_steps - 1 else int(num_mask_token)

            full_confidence = torch.full_like(x, -torch.inf, device=device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence

            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                else:
                    # temperature over selection scores
                    sel = full_confidence / alg_temp
                    sel = F.softmax(sel, dim=-1)
                    transfer_index = torch.multinomial(sel, num_samples=number_transfer_tokens)

                x_ = torch.zeros_like(x, device=device, dtype=torch.long) + int(mask_token_id)
                x_[mask_index] = x0.clone()
                row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
                x[row_indices, transfer_index] = x_[row_indices, transfer_index]

    # ---------------------- Build and return stats ----------------------
    if not early_exit_triggered:
        steps_taken_val = total_steps
        exit_decision_step = None

    elapsed = time.time() - t0

    # Prophet-style metadata wrapper (Dream._normalize_stats expects "exit_info")
    exit_info: Dict[str, Any] = {
        "early_exit_triggered": bool(early_exit_triggered),
        "exit_decision_step": int(exit_decision_step) if exit_decision_step is not None else None,
        "total_steps": int(total_steps),
        "actual_steps": int(steps_taken_val),
        "inference_time": float(elapsed),
    }

    # replicate per batch item
    stats_list: List[Dict[str, Any]] = [{"exit_info": dict(exit_info)} for _ in range(B)]

    return (x, stats_list) if return_stats else x
