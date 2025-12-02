# dream_sampling/common.py
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


class SimpleGenerateOutput:
    """Minimal HF-like container so callers can use `.sequences`."""
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Exponential-race / Gumbel-Top-1 equivalent noise injection.
    temperature=0 -> greedy.
    """
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Uniformly schedule number of token commits per step (per sample)."""
    # mask_index over the generation segment only: (B, L_gen)
    mask_num = mask_index.sum(dim=1, keepdim=True)  # (B,1)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    # distribute remainders to earliest steps per sample
    for i in range(mask_num.size(0)):
        r = int(remainder[i].item())
        if r > 0:
            num_transfer_tokens[i, :r] += 1
    return num_transfer_tokens


def tau_schedule(
    p: float,
    *,
    mode: str = "linear",
    tau_high: float = 7.5,
    tau_low: float = 2.5,
    k: float = 4.0,
) -> float:
    """
    Continuous threshold schedule τ(p), p∈[0,1].
      - linear: τ = tau_high + (tau_low - tau_high) * p
      - cosine: τ = tau_low + 0.5 * (tau_high - tau_low) * (1 + cos(pi*p))
      - exp:    τ = tau_low + (tau_high - tau_low) * exp(-k * p)
    """
    if mode == "linear":
        return tau_high + (tau_low - tau_high) * p
    if mode == "cosine":
        return tau_low + 0.5 * (tau_high - tau_low) * (1.0 + math.cos(math.pi * p))
    if mode == "exp":
        return tau_low + (tau_high - tau_low) * math.exp(-k * p)
    raise ValueError("tau_mode must be 'linear' | 'cosine' | 'exp'")


def tau_phase(p: float, thr: Dict[str, float]) -> float:
    """Legacy 3-phase thresholds to mirror Prophet's early/mid/late."""
    early = float(thr.get("early", 7.5))
    mid = float(thr.get("mid", 5.0))
    late = float(thr.get("late", 2.5))
    if p < (1.0 / 3.0):
        return early
    if p < (2.0 / 3.0):
        return mid
    return late


def confidence_statistic_batch(
    logits: torch.Tensor,          # (B, L, V)
    mask_index: torch.Tensor,      # (B, L) masked positions
    region_mask: torch.Tensor,     # (B, L) region over which to aggregate
    *,
    stat: str = "mean",
    q: float = 0.20,
    average_over_masked: bool = True,
) -> torch.Tensor:
    """Return per-sample aggregated top1-top2 logit gap; shape (B,).
    Where no eligible positions exist, returns +inf for that sample.
    """
    logits_f = logits.to(torch.float32)
    top2_vals, _ = torch.topk(logits_f, k=2, dim=-1)
    gaps = top2_vals[..., 0] - top2_vals[..., 1]  # (B, L)

    eligible = region_mask
    if average_over_masked:
        eligible = eligible & mask_index

    B = gaps.size(0)
    out = torch.full((B,), float("inf"), device=gaps.device, dtype=torch.float32)
    for b in range(B):
        idx = eligible[b]
        n = int(idx.sum().item())
        if n == 0:
            out[b] = float("inf")
            continue
        vals = gaps[b][idx]
        if stat == "mean":
            out[b] = vals.mean()
        elif stat == "median":
            out[b] = vals.median()
        elif stat == "quantile":
            out[b] = torch.quantile(vals, q)
        elif stat == "min":
            out[b] = vals.min()
        else:
            raise ValueError("stat must be 'mean'|'median'|'quantile'|'min'")
    return out