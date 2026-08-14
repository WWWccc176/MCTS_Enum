from __future__ import annotations

import torch
from torch import nn

from .batching import ModelBatch
from .config import NetworkConfig


class MCTSNet(nn.Module):
    def __init__(self, global_dim: int, recent_dim: int, config: NetworkConfig | None = None):
        super().__init__()
        cfg = config or NetworkConfig()
        self.level_encoder = nn.Sequential(
            nn.Linear(2, cfg.level_dim),
            nn.SiLU(),
            nn.Linear(cfg.level_dim, cfg.level_dim),
            nn.SiLU(),
        )
        pooled_dim = cfg.level_dim * 2
        self.state_encoder = nn.Sequential(
            nn.Linear(global_dim + recent_dim + pooled_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(3, cfg.candidate_dim),
            nn.SiLU(),
            nn.Linear(cfg.candidate_dim, cfg.candidate_dim),
            nn.SiLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.candidate_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, batch: ModelBatch) -> tuple[torch.Tensor, torch.Tensor]:
        level = self.level_encoder(batch.gso_features)
        level_mean = level.mean(dim=1)
        level_max = level.max(dim=1).values
        state_input = torch.cat(
            [
                batch.global_features,
                batch.recent_features,
                level_mean,
                level_max,
            ],
            dim=-1,
        )
        state = self.state_encoder(state_input)
        candidate = self.candidate_encoder(batch.candidate_features)
        expanded_state = state.unsqueeze(1).expand(-1, candidate.shape[1], -1)
        policy_input = torch.cat([expanded_state, candidate], dim=-1)
        logits = self.policy_head(policy_input).squeeze(-1)
        logits = logits.masked_fill(~batch.candidate_mask, float("-inf"))
        value = self.value_head(state).squeeze(-1)
        return logits, value
