from __future__ import annotations
from .config import RuntimeConfig


def build_search_config(backend, node_budget: int, runtime: RuntimeConfig, cpu_threads: int):
    if node_budget == -1:
        unlimited_nodes = True
        native_budget = 0
    elif node_budget > 0:
        unlimited_nodes = False
        native_budget = int(node_budget)
    else:
        raise ValueError("node_budget must be -1 or a positive integer")

    config = backend.SearchConfig()
    config.node_budget = native_budget
    config.unlimited_nodes = unlimited_nodes
    config.search_threads = int(cpu_threads)
    config.rollout_dimensions = int(runtime.rollout_dimensions)
    config.rollout_solutions = int(runtime.rollout_solutions)
    config.w_m = float(runtime.w_m)
    config.lambda_puct = float(runtime.lambda_puct)
    config.cpw = float(runtime.cpw)
    config.dpw = float(runtime.dpw)
    config.numeric_guard_rel = float(runtime.numeric_guard_rel)
    config.numeric_guard_abs = float(runtime.numeric_guard_abs)
    config.lll_delta = float(runtime.lll_delta)
    config.refresh_potential_rel_tolerance = float(runtime.refresh_potential_rel_tolerance)
    return config
