from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(slots=True)
class ModelBatch:
    global_features: torch.Tensor
    gso_features: torch.Tensor
    recent_features: torch.Tensor
    candidate_features: torch.Tensor
    candidate_mask: torch.Tensor


@dataclass(slots=True)
class TrainingBatch(ModelBatch):
    policy_target: torch.Tensor
    value_target: torch.Tensor
    value_mask: torch.Tensor


def _shape_request(request: dict) -> tuple[list[float], list[float], list[list[float]]]:
    global_features = [float(x) for x in request["global"]]
    candidate_raw = [float(x) for x in request["candidates"]]
    recent = [float(x) for x in request["recent"]]
    if len(candidate_raw) % 3:
        raise ValueError("candidate feature vector is not divisible by 3")
    candidates = [candidate_raw[i : i + 3] for i in range(0, len(candidate_raw), 3)]
    return global_features, recent, candidates


def _cpu_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return tensor
    return tensor.pin_memory().to(device=device, non_blocking=True)


def make_gso_tensor(
    gso_features: Sequence[float],
    device: torch.device,
) -> torch.Tensor:
    values = [float(x) for x in gso_features]
    if not values or len(values) % 2:
        raise ValueError("static GSO feature vector must be non-empty and divisible by 2")
    cpu = torch.tensor(values, dtype=torch.float32).view(-1, 2)
    return _cpu_to_device(cpu, device)


def _expand_gso(gso_level: torch.Tensor, batch_size: int) -> torch.Tensor:
    if gso_level.ndim != 2 or gso_level.shape[-1] != 2:
        raise ValueError("GSO tensor must have shape [dimension, 2]")
    return gso_level.unsqueeze(0).expand(batch_size, -1, -1)


def collate_requests(
    requests: Iterable[dict],
    device: torch.device,
    gso_level: torch.Tensor,
) -> ModelBatch:
    rows = [_shape_request(request) for request in requests]
    if not rows:
        raise ValueError("cannot collate an empty request batch")

    max_candidates = max(len(row[2]) for row in rows)
    padded_candidates: list[list[list[float]]] = []
    masks: list[list[bool]] = []
    for _, _, candidates in rows:
        count = len(candidates)
        padded_candidates.append(
            candidates + [[0.0, 0.0, 0.0] for _ in range(max_candidates - count)]
        )
        masks.append([True] * count + [False] * (max_candidates - count))

    global_cpu = torch.tensor([row[0] for row in rows], dtype=torch.float32)
    recent_cpu = torch.tensor([row[1] for row in rows], dtype=torch.float32)
    candidate_cpu = torch.tensor(padded_candidates, dtype=torch.float32)
    mask_cpu = torch.tensor(masks, dtype=torch.bool)

    return ModelBatch(
        global_features=_cpu_to_device(global_cpu, device),
        gso_features=_expand_gso(gso_level, len(rows)),
        recent_features=_cpu_to_device(recent_cpu, device),
        candidate_features=_cpu_to_device(candidate_cpu, device),
        candidate_mask=_cpu_to_device(mask_cpu, device),
    )


def collate_training(
    samples: list[dict],
    device: torch.device,
    gso_level: torch.Tensor,
) -> TrainingBatch:
    model_batch = collate_requests(samples, device, gso_level)
    max_candidates = model_batch.candidate_features.shape[1]

    policy_rows: list[list[float]] = []
    for sample in samples:
        target = [float(x) for x in sample["policy_target"]]
        policy_rows.append(target + [0.0] * (max_candidates - len(target)))

    policy_target = _cpu_to_device(
        torch.tensor(policy_rows, dtype=torch.float32), device
    )
    value_target = _cpu_to_device(
        torch.tensor(
            [float(sample["value_target"]) for sample in samples],
            dtype=torch.float32,
        ),
        device,
    )
    value_mask = _cpu_to_device(
        torch.tensor(
            [bool(sample["has_value_target"]) for sample in samples],
            dtype=torch.bool,
        ),
        device,
    )

    return TrainingBatch(
        global_features=model_batch.global_features,
        gso_features=model_batch.gso_features,
        recent_features=model_batch.recent_features,
        candidate_features=model_batch.candidate_features,
        candidate_mask=model_batch.candidate_mask,
        policy_target=policy_target,
        value_target=value_target,
        value_mask=value_mask,
    )
