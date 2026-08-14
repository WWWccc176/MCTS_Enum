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
    config.refresh_interval = int(runtime.refresh_interval)
    config.radius_global_update_interval = int(runtime.radius_global_update_interval)
    config.w_m = float(runtime.w_m)
    config.lambda_puct = float(runtime.lambda_puct)
    config.cpw = float(runtime.cpw)
    config.dpw = float(runtime.dpw)
    config.policy_mix = float(runtime.policy_mix)
    config.visit_temperature = float(runtime.visit_temperature)
    config.quality_gate = float(runtime.quality_gate)
    config.numeric_guard_rel = float(runtime.numeric_guard_rel)
    config.numeric_guard_abs = float(runtime.numeric_guard_abs)
    config.max_legal_actions = int(runtime.max_legal_actions)
    config.recent_residual_count = int(runtime.recent_residual_count)
    config.search_threads = int(cpu_threads)
    return config
