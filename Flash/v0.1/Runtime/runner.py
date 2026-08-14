from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .backend_loader import load_mcts_backend
from .config import RuntimeConfig
from .engine_config import build_search_config
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


def _status_line(status: dict, completed_nodes: int, rate: float) -> str:
    nodes = int(status["nodes"])
    budget = int(status["node_budget"])
    budget_text = "inf" if budget < 0 else str(budget)
    pct = "" if budget < 0 else f" {100.0 * nodes / max(1, budget):5.1f}%"
    return (
        f"[flash] phase={int(status['phase'])} nodes={nodes}/{budget_text}{pct} "
        f"total={completed_nodes + nodes} rate={rate:,.0f}/s "
        f"best/GH={float(status['best_quality_ratio']):.8f} "
        f"best@={int(status['best_found_at_node_count'])} "
        f"rootN={int(status['root_visits'])} "
        f"Rdrop={int(status['radius_drops_since_global_update'])}"
    )


def run_flash(spec: RunSpec, cpu_threads: int, runtime: RuntimeConfig, resource: dict) -> None:
    backend = load_mcts_backend()
    os.environ["OMP_NUM_THREADS"] = str(max(1, cpu_threads))
    config = build_search_config(backend, spec.node_budget, runtime, cpu_threads)
    engine = backend.SearchEngine(_load_packet(backend, spec.basis_path), config, False)

    completed_nodes = 0
    phase_limit = spec.refresh_cycles + 1
    for slot in range(phase_limit):
        start_status = dict(engine.status())
        print(
            f"[PHASE] {slot + 1}/{phase_limit} start phase={int(start_status['phase'])} "
            f"budget_per_phase={spec.node_budget} "
            f"b1/GH={float(start_status['initial_quality_ratio']):.8f}",
            flush=True,
        )

        stop = threading.Event()
        started = time.monotonic()
        state = {"time": started, "nodes": int(start_status["nodes"])}

        def monitor() -> None:
            interval = max(0.25, float(runtime.status_interval_seconds))
            while not stop.wait(interval):
                now = time.monotonic()
                status = dict(engine.status())
                nodes = int(status["nodes"])
                dt = max(now - state["time"], 1.0e-9)
                rate = (nodes - state["nodes"]) / dt
                print(_status_line(status, completed_nodes, rate), flush=True)
                state["time"] = now
                state["nodes"] = nodes

        thread = threading.Thread(target=monitor, name="flash-status", daemon=True)
        thread.start()
        try:
            engine.run_flash()
        finally:
            stop.set()
            thread.join()

        end = dict(engine.status())
        phase_nodes = int(end["nodes"])
        elapsed = max(time.monotonic() - started, 1.0e-9)
        print(_status_line(end, completed_nodes, phase_nodes / elapsed), flush=True)
        print(
            f"[PHASE] {slot + 1}/{phase_limit} end phase={int(end['phase'])} "
            f"nodes={phase_nodes} stop={_stop_reason(end)} "
            f"best/GH={float(end['best_quality_ratio']):.8f}",
            flush=True,
        )
        completed_nodes += phase_nodes

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

    engine.write_results(str(spec.result_root), spec.version, spec.run_id)
    append_summary(spec.result_dir, spec.metadata, resource)
    print(f"[RESULT] {spec.result_dir}", flush=True)
