from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    cpu_fraction: float = 0.85
    rollout_dimensions: int = 10
    rollout_solutions: int = 8
    w_m: float = 0.25
    lambda_puct: float = 1.5
    cpw: float = 2.0
    dpw: float = 0.5
    numeric_guard_rel: float = 1.0e-12
    numeric_guard_abs: float = 1.0e-18
    lll_delta: float = 0.999
    refresh_potential_rel_tolerance: float = 1.0e-10
    status_interval_seconds: float = 2.0
