from __future__ import annotations

import csv
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
        return "work_budget"
    return "finished"


def _status_line(status: dict, completed_work: int, rate: float) -> str:
    work = int(status["work_nodes"])
    budget = int(status["node_budget"])
    budget_text = "inf" if budget < 0 else str(budget)
    pct = "" if budget < 0 else f" {100.0 * work / max(1, budget):5.1f}%"
    return (
        f"[flash-v0.2] phase={int(status['phase'])} work={work}/{budget_text}{pct} "
        f"total={completed_work + work} rate={rate:,.0f}/s "
        f"tree={int(status['tree_nodes'])} enum={int(status['enumeration_nodes'])} "
        f"rollouts={int(status['rollout_jobs'])} active={int(status['active_rollouts'])} "
        f"R/GH={float(status['search_radius_quality_ratio']):.8f} "
        f"best/GH={float(status['best_quality_ratio']):.8f} "
        f"Rdrop={int(status['radius_drops'])}"
    )


def _write_speed_trace(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "phase",
        "elapsed_s",
        "work_nodes",
        "tree_nodes",
        "enumeration_nodes",
        "rollout_jobs",
        "exact_candidates",
        "active_rollouts",
        "rate_work_nodes_s",
        "search_radius_over_gh",
        "best_over_gh",
        "radius_drops",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_flash(spec: RunSpec, cpu_threads: int, runtime: RuntimeConfig, resource: dict) -> None:
    backend = load_mcts_backend()
    if str(getattr(backend, "BACKEND_VERSION", "")) != spec.version:
        raise RuntimeError(
            f"compiled backend version mismatch: expected {spec.version}, "
            f"got {getattr(backend, 'BACKEND_VERSION', None)}"
        )

    os.environ["OMP_NUM_THREADS"] = str(max(1, cpu_threads))
    config = build_search_config(backend, spec.node_budget, runtime, cpu_threads)
    engine = backend.SearchEngine(_load_packet(backend, spec.basis_path), config, False)

    completed_work = 0
    phase_limit = spec.refresh_cycles + 1
    speed_rows: list[dict] = []

    for slot in range(phase_limit):
        start = dict(engine.status())
        phase = int(start["phase"])
        print(
            f"[PHASE] {slot + 1}/{phase_limit} start phase={phase} "
            f"work_budget={spec.node_budget} "
            f"R0/GH={float(start['initial_quality_ratio']):.8f} "
            f"threads={cpu_threads} rollout_dims={runtime.rollout_dimensions}",
            flush=True,
        )

        stop = threading.Event()
        started = time.monotonic()
        state = {"time": started, "work": int(start["work_nodes"])}

        def sample(status: dict, now: float, rate: float) -> None:
            speed_rows.append(
                {
                    "phase": int(status["phase"]),
                    "elapsed_s": f"{now - started:.6f}",
                    "work_nodes": int(status["work_nodes"]),
                    "tree_nodes": int(status["tree_nodes"]),
                    "enumeration_nodes": int(status["enumeration_nodes"]),
                    "rollout_jobs": int(status["rollout_jobs"]),
                    "exact_candidates": int(status["exact_candidates"]),
                    "active_rollouts": int(status["active_rollouts"]),
                    "rate_work_nodes_s": f"{rate:.6f}",
                    "search_radius_over_gh": f"{float(status['search_radius_quality_ratio']):.10f}",
                    "best_over_gh": f"{float(status['best_quality_ratio']):.10f}",
                    "radius_drops": int(status["radius_drops"]),
                }
            )

        def monitor() -> None:
            interval = max(0.25, float(runtime.status_interval_seconds))
            while not stop.wait(interval):
                now = time.monotonic()
                status = dict(engine.status())
                work = int(status["work_nodes"])
                dt = max(now - state["time"], 1.0e-9)
                rate = (work - state["work"]) / dt
                print(_status_line(status, completed_work, rate), flush=True)
                sample(status, now, rate)
                state["time"] = now
                state["work"] = work

        thread = threading.Thread(target=monitor, name="flash-v02-status", daemon=True)
        thread.start()
        try:
            engine.run_flash()
        finally:
            stop.set()
            thread.join()

        end = dict(engine.status())
        phase_work = int(end["work_nodes"])
        elapsed = max(time.monotonic() - started, 1.0e-9)
        rate = phase_work / elapsed
        print(_status_line(end, completed_work, rate), flush=True)
        sample(end, time.monotonic(), rate)
        print(
            f"[PHASE] {slot + 1}/{phase_limit} end phase={phase} "
            f"work={phase_work} tree={int(end['tree_nodes'])} "
            f"enum={int(end['enumeration_nodes'])} rollouts={int(end['rollout_jobs'])} "
            f"stop={_stop_reason(end)} best/GH={float(end['best_quality_ratio']):.8f}",
            flush=True,
        )
        completed_work += phase_work

        if slot + 1 >= phase_limit:
            break
        if not bool(end["refresh_candidate_available"]):
            print(
                f"[REFRESH] after_phase={phase} skipped=1 reason=no_nonbasis_rollout_candidate",
                flush=True,
            )
            break

        candidate_z = list(engine.refresh_candidate_coefficients())
        candidate_text = ",".join(str(int(v)) for v in candidate_z)
        print(
            f"[REFRESH] after_phase={phase} method=insert-nonbasis+LLL+potential-gate "
            f"candidate/GH={float(end['refresh_candidate_quality_ratio']):.8f} "
            f"candidate@work={int(end['refresh_candidate_found_at_work_node_count'])}",
            flush=True,
        )
        print(f"          z(z_1->z_n)=[{candidate_text}]", flush=True)
        changed = bool(engine.refresh_basis_with_best())
        if not changed:
            print(
                f"[REFRESH] after_phase={phase} accepted_change=0 stop=refresh_rejected_or_unchanged",
                flush=True,
            )
            break

        nxt = dict(engine.status())
        print(
            f"[REFRESH] accepted_change=1 next_phase={int(nxt['phase'])} "
            f"new_R0/GH={float(nxt['initial_quality_ratio']):.8f} "
            f"global_best/GH={float(nxt['best_quality_ratio']):.8f}",
            flush=True,
        )

    engine.write_results(str(spec.result_root), spec.version, spec.run_id)
    append_summary(spec.result_dir, spec.metadata, resource)
    _write_speed_trace(spec.result_dir / "speed.csv", speed_rows)
    print(f"[RESULT] {spec.result_dir}", flush=True)
