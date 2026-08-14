from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import torch

from .batching import make_gso_tensor
from .inference import infer_requests
from .model import MCTSNet


@dataclass(slots=True)
class _MicroBatch:
    indices: list[int]
    padded_cost: int


class MultiDeviceInferencePool:
    def __init__(
        self,
        model: MCTSNet,
        primary_device: torch.device,
        max_padded_candidates_per_device: int = 262_144,
    ):
        if primary_device.type == "cuda" and torch.cuda.is_available():
            self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
            if primary_device not in self.devices:
                raise ValueError(f"primary CUDA device {primary_device} is not visible")
            primary_index = self.devices.index(primary_device)
            if primary_index != 0:
                self.devices[0], self.devices[primary_index] = self.devices[primary_index], self.devices[0]
        else:
            self.devices = [torch.device("cpu")]

        self.max_padded_candidates = max(1, int(max_padded_candidates_per_device))
        self.models: list[MCTSNet] = [model.to(self.devices[0])]
        for device in self.devices[1:]:
            replica = copy.deepcopy(model).to(device)
            replica.eval()
            self.models.append(replica)
        self._gso_levels: list[torch.Tensor] = []

    @property
    def primary_model(self) -> MCTSNet:
        return self.models[0]

    @property
    def primary_device(self) -> torch.device:
        return self.devices[0]

    @property
    def gpu_count(self) -> int:
        return sum(device.type == "cuda" for device in self.devices)

    def set_gso_features(self, features: Sequence[float]) -> None:
        self._gso_levels = [make_gso_tensor(features, device) for device in self.devices]

    def sync_replicas(self) -> None:
        if len(self.models) <= 1:
            return
        state = self.primary_model.state_dict()
        for replica in self.models[1:]:
            replica.load_state_dict(state)
            replica.eval()

    def _microbatches(self, requests: list[dict]) -> list[_MicroBatch]:
        order = sorted(
            range(len(requests)),
            key=lambda index: int(requests[index]["candidate_count"]),
        )
        batches: list[_MicroBatch] = []
        current: list[int] = []
        current_max = 0
        for index in order:
            count = max(1, int(requests[index]["candidate_count"]))
            proposed_max = max(current_max, count)
            proposed_cost = proposed_max * (len(current) + 1)
            if current and proposed_cost > self.max_padded_candidates:
                batches.append(_MicroBatch(current, current_max * len(current)))
                current = []
                current_max = 0
            current.append(index)
            current_max = max(current_max, count)
            if current_max > self.max_padded_candidates and len(current) == 1:
                batches.append(_MicroBatch(current, current_max))
                current = []
                current_max = 0
        if current:
            batches.append(_MicroBatch(current, current_max * len(current)))
        return batches

    def infer(self, requests: list[dict]) -> tuple[list[list[float]], list[float]]:
        if not requests:
            return [], []
        if len(self._gso_levels) != len(self.devices):
            raise RuntimeError("inference-pool GSO features were not initialized")

        microbatches = self._microbatches(requests)
        device_jobs: list[list[_MicroBatch]] = [[] for _ in self.devices]
        device_costs = [0 for _ in self.devices]
        for batch in sorted(microbatches, key=lambda item: item.padded_cost, reverse=True):
            worker = min(range(len(self.devices)), key=lambda i: device_costs[i])
            device_jobs[worker].append(batch)
            device_costs[worker] += batch.padded_cost

        logits_out: list[list[float] | None] = [None] * len(requests)
        values_out: list[float | None] = [None] * len(requests)

        def run_device(worker: int) -> None:
            for batch in device_jobs[worker]:
                local_requests = [requests[index] for index in batch.indices]
                logits, values = infer_requests(
                    self.models[worker],
                    local_requests,
                    self.devices[worker],
                    self._gso_levels[worker],
                )
                for offset, index in enumerate(batch.indices):
                    logits_out[index] = logits[offset]
                    values_out[index] = values[offset]

        active_workers = [i for i, jobs in enumerate(device_jobs) if jobs]
        if len(active_workers) == 1:
            run_device(active_workers[0])
        else:
            with ThreadPoolExecutor(
                max_workers=len(active_workers), thread_name_prefix="mcts-gpu"
            ) as executor:
                futures = [executor.submit(run_device, worker) for worker in active_workers]
                for future in futures:
                    future.result()

        if any(row is None for row in logits_out) or any(value is None for value in values_out):
            raise RuntimeError("multi-device inference returned an incomplete batch")
        return [row for row in logits_out if row is not None], [float(v) for v in values_out if v is not None]
