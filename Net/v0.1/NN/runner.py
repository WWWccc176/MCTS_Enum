from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .backend_loader import load_mcts_backend
from .config import NetworkConfig, RuntimeConfig, TrainConfig
from .engine_config import build_search_config
from .net_loop import NetController
from .run_meta import append_summary


@dataclass(slots=True)
class RunSpec:
    basis_path: Path
    node_budget: int
    refresh_cycles: int
    run_id: str
    result_root: Path
    result_dir: Path
    version: str
    metadata: dict


def _load_packet(backend, path: Path) -> bytes:
    return bytes(backend.encode_basis_text(path.read_text(encoding="utf-8")))


def _stop_reason(status: dict) -> str:
    if bool(status["root_closed"]):
        return "tree_exhausted"
    if bool(status["budget_reached"]):
        return "node_budget"
    return "finished"


def run_net(
    spec: RunSpec,
    cpu_threads: int,
    runtime: RuntimeConfig,
    train_config: TrainConfig,
    network_config: NetworkConfig,
    resource: dict,
) -> None:
    backend = load_mcts_backend()
    os.environ["OMP_NUM_THREADS"] = str(max(1, cpu_threads))
    torch.set_num_threads(max(1, cpu_threads))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    config = build_search_config(backend, spec.node_budget, runtime, cpu_threads)
    engine = backend.SearchEngine(_load_packet(backend, spec.basis_path), config, True)
    controller = NetController(engine, device, runtime, train_config, network_config)
    phase_limit = spec.refresh_cycles + 1

    for slot in range(phase_limit):
        start = dict(engine.status())
        print(
            f"[PHASE] {slot + 1}/{phase_limit} start phase={int(start['phase'])} "
            f"budget_per_phase={spec.node_budget} "
            f"b1/GH={float(start['initial_quality_ratio']):.8f}",
            flush=True,
        )
        controller.run_phase(engine)
        end = dict(engine.status())
        print(
            f"[PHASE] {slot + 1}/{phase_limit} end phase={int(end['phase'])} "
            f"nodes={int(end['nodes'])} stop={_stop_reason(end)} "
            f"best/GH={float(end['best_quality_ratio']):.8f}",
            flush=True,
        )

        if slot + 1 < phase_limit:
            if not bool(end["refresh_candidate_available"]):
                print(
                    f"[REFRESH] after_phase={int(end['phase'])} skipped=1 "
                    "reason=no_nonbasis_terminal; no new basis can be formed",
                    flush=True,
                )
                break

            candidate_z = list(engine.refresh_candidate_coefficients())
            candidate_text = ",".join(str(int(v)) for v in candidate_z)
            print(
                f"[REFRESH] after_phase={int(end['phase'])} method=insert-nonbasis+LLL "
                f"candidate/GH={float(end['refresh_candidate_quality_ratio']):.8f} "
                f"candidate@={int(end['refresh_candidate_found_at_node_count'])}",
                flush=True,
            )
            print(f"          z(z_1->z_n)=[{candidate_text}]", flush=True)
            changed = bool(engine.refresh_basis_with_best())
            nxt = dict(engine.status())
            print(
                f"[REFRESH] changed={int(changed)} next_phase={int(nxt['phase'])} "
                f"new_b1/GH={float(nxt['initial_quality_ratio']):.8f}",
                flush=True,
            )
            controller.reset_replay_for_new_basis()
            controller.bootstrap_new_phase(engine)

    engine.write_results(str(spec.result_root), spec.version, spec.run_id)
    append_summary(spec.result_dir, spec.metadata, resource)
    print(f"[RESULT] {spec.result_dir}", flush=True)
