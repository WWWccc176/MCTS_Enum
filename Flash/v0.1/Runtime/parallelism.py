from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    cpu_ids: tuple[int, ...]
    cpu_threads: int
    cpu_total: int


def available_cpu_ids() -> tuple[int, ...]:
    try:
        return tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return tuple(range(os.cpu_count() or 1))


def build_resource_plan(cpu_fraction: float = 0.85) -> ResourcePlan:
    if not 0.0 < cpu_fraction <= 1.0:
        raise ValueError("cpu_fraction must be in (0, 1]")
    available = available_cpu_ids()
    count = max(1, min(len(available), math.floor(len(available) * cpu_fraction)))
    selected = available[:count]
    return ResourcePlan(selected, len(selected), len(available))


def apply_cpu_plan(plan: ResourcePlan) -> None:
    try:
        os.sched_setaffinity(0, set(plan.cpu_ids))
    except (AttributeError, OSError):
        pass
    os.environ["OMP_NUM_THREADS"] = str(plan.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(plan.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(plan.cpu_threads)
