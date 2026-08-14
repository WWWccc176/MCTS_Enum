from __future__ import annotations

import argparse
from pathlib import Path

import torch

from NN.backend_loader import load_mcts_backend
from NN.config import RuntimeConfig, TrainConfig
from NN.pro_loop import ProController


def basis_text() -> str:
    return "[[2 0 0]\n[0 1 0]\n[0 0 100]]\n"


def long_zero_prefix_basis_text(dimension: int = 40) -> str:
    diagonal = [2, 1] + [100] * (dimension - 2)
    rows = []
    for i, value in enumerate(diagonal):
        row = [0] * dimension
        row[i] = value
        rows.append("[" + " ".join(str(x) for x in row) + "]")
    return "[" + "\n".join(rows) + "]\n"


def require_finished(engine, label: str) -> None:
    if not bool(engine.finished):
        raise RuntimeError(
            f"{label} did not reach a valid stop condition: "
            + str(engine.diagnostic_status())
        )


def run_pro_smoke(backend, packet: bytes, node_budget: int, device: torch.device):
    cfg = backend.SearchConfig()
    cfg.node_budget = node_budget
    cfg.quality_gate = 1.2
    cfg.search_threads = min(4, max(1, int(torch.get_num_threads() or 1)))
    engine = backend.SearchEngine(packet, cfg, True)

    # A smoke test must exercise the same bootstrap/inference/submit state
    # machine as production.  Do not pre-consume an inference request here:
    # ProController owns the root bootstrap request and immediately submits it.
    runtime = RuntimeConfig(
        inference_batch_size=64,
        refresh_interval=1_000_000,
    )
    controller = ProController(engine, device, runtime, TrainConfig())
    controller.run_phase(engine)
    require_finished(engine, "Pro smoke")
    return engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-budget", type=int, default=128)
    parser.add_argument("--result-root", type=Path, default=Path("/tmp/mcts_v01_smoke"))
    args = parser.parse_args()

    backend = load_mcts_backend()
    packet = bytes(backend.encode_basis_text(basis_text()))

    flash = backend.SearchConfig()
    flash.node_budget = args.node_budget
    flash.quality_gate = 1.2
    flash.search_threads = max(1, int((torch.get_num_threads() or 1)))
    engine = backend.SearchEngine(packet, flash, False)
    engine.run_flash()
    require_finished(engine, "Flash smoke")
    assert int(engine.node_count) > 1
    assert list(engine.best_coefficients())[1] in (-1, 1)
    engine.write_results(str(args.result_root), "v0.1flash", "smoke_flash")
    flash_files = sorted((args.result_root / "v0.1flash" / "smoke_flash").iterdir())
    assert len(flash_files) == 6
    assert all(path.is_file() and path.suffix == ".txt" for path in flash_files)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pro = run_pro_smoke(backend, packet, args.node_budget, device)
    pro.write_results(str(args.result_root), "v0.1pro", "smoke_pro")
    pro_files = sorted((args.result_root / "v0.1pro" / "smoke_pro").iterdir())
    assert len(pro_files) == 6
    assert all(path.is_file() and path.suffix == ".txt" for path in pro_files)

    regression_cfg = backend.SearchConfig()
    regression_cfg.node_budget = max(512, args.node_budget)
    regression_cfg.quality_gate = 1.2
    regression_cfg.search_threads = min(4, max(1, int(torch.get_num_threads() or 1)))
    regression = backend.SearchEngine(
        bytes(backend.encode_basis_text(long_zero_prefix_basis_text())),
        regression_cfg,
        True,
    )
    regression_runtime = RuntimeConfig(
        inference_batch_size=64,
        refresh_interval=1_000_000,
    )
    controller = ProController(
        regression,
        device,
        regression_runtime,
        TrainConfig(),
    )
    controller.run_phase(regression)
    require_finished(regression, "Long-zero-prefix Pro regression")
    assert int(regression.progress_epoch) > 0

    print("SMOKE_OK")


if __name__ == "__main__":
    main()
