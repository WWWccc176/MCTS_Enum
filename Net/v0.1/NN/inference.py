from __future__ import annotations

import torch

from .batching import collate_requests
from .model import MCTSNet


@torch.inference_mode()
def infer_requests(
    model: MCTSNet,
    requests: list[dict],
    device: torch.device,
    gso_level: torch.Tensor,
) -> tuple[list[list[float]], list[float]]:
    batch = collate_requests(requests, device, gso_level)
    logits, values = model(batch)
    output_logits: list[list[float]] = []
    for i, request in enumerate(requests):
        count = int(request["candidate_count"])
        output_logits.append(logits[i, :count].detach().float().cpu().tolist())
    return output_logits, values.detach().float().cpu().tolist()
