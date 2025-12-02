# dream_sampling/__init__.py
from __future__ import annotations

from typing import Callable, Dict

from .common import (
    SimpleGenerateOutput,
    add_gumbel_noise,
    get_num_transfer_tokens,
    tau_schedule,
    tau_phase,
    confidence_statistic_batch,
)
from .prophet import diffusion_generate_prophet
from .sched import diffusion_generate_schedule

__all__ = [
    "SimpleGenerateOutput",
    "add_gumbel_noise",
    "get_num_transfer_tokens",
    "tau_schedule",
    "tau_phase",
    "confidence_statistic_batch",
    "diffusion_generate_prophet",
    "diffusion_generate_schedule",
    "get_generator",
    "__version__",
]

__version__ = "0.1.0"

# Minimal registry to make integration code simple/robust.
_GENERATOR_REGISTRY: Dict[str, Callable[..., SimpleGenerateOutput]] = {
    "prophet": diffusion_generate_prophet,
    "schedule": diffusion_generate_schedule,
}


def get_generator(name: str) -> Callable[..., SimpleGenerateOutput]:
    """
    Return a generator function by name.

    Args:
        name: 'prophet' | 'schedule'

    Raises:
        KeyError if name is unknown.
    """
    key = (name or "").lower()
    if key not in _GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator '{name}'. Available: {list(_GENERATOR_REGISTRY)}")
    return _GENERATOR_REGISTRY[key]
