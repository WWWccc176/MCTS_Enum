from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    cpu_fraction: float = 0.85
    refresh_interval: int = 5000
    radius_global_update_interval: int = 2
    quality_gate: float = 1.2
    w_m: float = 0.25
    lambda_puct: float = 1.5
    cpw: float = 2.0
    dpw: float = 0.5
    policy_mix: float = 0.5
    visit_temperature: float = 1.0
    max_legal_actions: int = 65536
    numeric_guard_rel: float = 1.0e-12
    numeric_guard_abs: float = 1.0e-18
    recent_residual_count: int = 8
    status_interval_seconds: float = 2.0
