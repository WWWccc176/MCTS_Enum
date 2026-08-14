from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NetworkConfig:
    hidden_dim: int = 192
    level_dim: int = 64
    candidate_dim: int = 64
    dropout: float = 0.0


@dataclass(slots=True)
class TrainConfig:
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    batch_size: int = 256
    replay_capacity: int = 200_000
    gradient_steps_per_refresh: int = 64
    value_loss_weight: float = 1.0
    grad_clip_norm: float = 2.0
    max_training_samples_per_refresh: int = 50_000


@dataclass(slots=True)
class RuntimeConfig:
    inference_batch_size: int = 256
    refresh_batch_size: int = 512
    max_padded_candidates_per_device: int = 262_144
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
