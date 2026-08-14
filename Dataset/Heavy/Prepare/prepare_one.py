from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HEAVY_BUILD = ROOT / "HeavyBack" / "build"
if str(HEAVY_BUILD) not in sys.path:
    sys.path.insert(0, str(HEAVY_BUILD))

QUALITY_GATE = 0.995


def prepare_basis_in_memory(raw_text: str, bkz_loops: int = 2):
    import heavy_backend

    matrix_id = heavy_backend.create_matrix_lll(raw_text)
    try:
        dimension = int(heavy_backend.quality_metrics(matrix_id)["dimension"])
        bkz = dict(
            heavy_backend.reduce_bkz2_global(
                matrix_id, min(30, dimension), max(1, int(bkz_loops))
            )
        )
        if not bool(bkz["completed"]):
            raise RuntimeError(f"BKZ30 failed: {bkz}")

        sieve = dict(heavy_backend.reduce_sieve_block(matrix_id, 0, dimension))
        if not bool(sieve["completed"]):
            raise RuntimeError(f"whole-basis BGJ stage failed: {sieve}")

        metrics = dict(heavy_backend.quality_metrics(matrix_id))
        ratio = float(metrics["b1_over_gh"])
        if ratio > QUALITY_GATE:
            raise RuntimeError(
                f"prepared basis rejected: b1/GH={ratio:.8f} > {QUALITY_GATE:.3f}"
            )

        basis_text = heavy_backend.dump_matrix(matrix_id)
        basis_packet = bytes(heavy_backend.export_matrix_packet(matrix_id))
        report = {
            "schema": "1.0",
            "quality_gate": QUALITY_GATE,
            "dimension": dimension,
            "b1_over_gh": ratio,
            "log_b1": float(metrics["log_b1"]),
            "log_gh": float(metrics["log_gh"]),
            "bkz": bkz,
            "sieve": sieve,
        }
        return basis_packet, basis_text, report
    finally:
        heavy_backend.free_matrix(matrix_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("basis", type=Path)
    parser.add_argument("--bkz-loops", type=int, default=2)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    raw_text = args.basis.read_text(encoding="utf-8")
    _, basis_text, report = prepare_basis_in_memory(raw_text, args.bkz_loops)
    name = args.name or args.basis.name
    target = ROOT / "Dataset" / "Heavy" / "Basis" / name
    target.write_text(basis_text, encoding="utf-8")
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(target)
    print(f"b1/GH={float(report['b1_over_gh']):.8f}")


if __name__ == "__main__":
    main()
