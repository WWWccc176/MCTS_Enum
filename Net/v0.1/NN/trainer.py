from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .batching import collate_training, make_gso_tensor
from .config import TrainConfig
from .model import MCTSNet
from .replay import ReplayBuffer


@dataclass(slots=True)
class TrainMetrics:
    policy_loss: float
    value_loss: float
    total_loss: float
    learning_rate: float


class Trainer:
    def __init__(
        self,
        model: MCTSNet,
        device: torch.device,
        config: TrainConfig | None = None,
    ):
        self.model = model
        self.device = device
        self.config = config or TrainConfig()
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._gso_level: torch.Tensor | None = None

    def set_gso_features(self, features: Sequence[float]) -> None:
        self._gso_level = make_gso_tensor(features, self.device)

    def reset_replay(self) -> None:
        self.replay = ReplayBuffer(self.config.replay_capacity)

    def update(self, new_samples: list[dict]) -> TrainMetrics:
        if self._gso_level is None:
            raise RuntimeError("trainer GSO features were not initialized")
        self.replay.extend(new_samples)
        policy_total = 0.0
        value_total = 0.0
        total_total = 0.0
        actual_steps = 0
        self.model.train()
        for _ in range(self.config.gradient_steps_per_refresh):
            samples = self.replay.sample(self.config.batch_size)
            if not samples:
                break
            batch = collate_training(samples, self.device, self._gso_level)
            logits, values = self.model(batch)
            log_probs = torch.log_softmax(logits, dim=-1)
            safe_log_probs = torch.where(
                batch.candidate_mask,
                log_probs,
                torch.zeros_like(log_probs),
            )
            policy_loss = -(batch.policy_target * safe_log_probs).sum(dim=-1).mean()
            if batch.value_mask.any():
                value_loss = nn.functional.mse_loss(
                    values[batch.value_mask], batch.value_target[batch.value_mask]
                )
            else:
                value_loss = values.sum() * 0.0
            total_loss = policy_loss + self.config.value_loss_weight * value_loss
            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
            policy_total += float(policy_loss.detach())
            value_total += float(value_loss.detach())
            total_total += float(total_loss.detach())
            actual_steps += 1
        self.model.eval()
        divisor = max(1, actual_steps)
        return TrainMetrics(
            policy_loss=policy_total / divisor,
            value_loss=value_total / divisor,
            total_loss=total_total / divisor,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
        )
