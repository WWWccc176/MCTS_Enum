from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("Flash", "Net"))
    parser.add_argument("version")
    args = parser.parse_args()

    target = ROOT / args.model / args.version
    assert target.is_dir(), target
    assert (target / "VERSION").read_text().strip() == args.version

    mcts = target / "MCTSBack"
    refresh_h = (mcts / "include/mcts/basis_refresh.hpp").read_text()
    refresh_cpp = (mcts / "src/basis_refresh.cpp").read_text()
    refresh_text = (refresh_h + refresh_cpp).lower()
    search_h = (mcts / "include/mcts/search_engine.hpp").read_text()
    search_cpp = (mcts / "src/search_engine.cpp").read_text()
    pybind = (mcts / "src/pybind_module.cpp").read_text()

    assert "lll_refresh" in refresh_text
    assert "lll_bkz" not in refresh_text
    assert "bkz20" not in refresh_text
    assert "bkz_reduction" not in refresh_text
    assert "expanded[0]" in refresh_cpp
    assert "zero-row removal" in refresh_cpp
    assert "candidate must not be a single basis vector" in refresh_cpp

    assert "refresh_candidate_sq_" in search_h
    assert "refresh_candidate_z_" in search_h
    assert "is_single_basis_vector(z)" in search_cpp
    assert "lll_refresh(basis_, refresh_candidate_z_)" in search_cpp
    assert "lll_refresh(basis_, best_z_)" not in search_cpp
    assert "refresh_candidate_coefficients" in pybind
    assert "refresh_candidate_available" in pybind

    if args.model == "Flash":
        runner = (target / "Runtime/runner.py").read_text()
        assert not (target / "NN").exists()
    else:
        runner = (target / "NN/runner.py").read_text()
        assert (target / "NN/net_loop.py").is_file()
    assert "method=insert-nonbasis+LLL" in runner
    assert "reason=no_nonbasis_terminal" in runner

    print(f"SOURCE_INVARIANTS_OK model={args.model} version={args.version}")


if __name__ == "__main__":
    main()
