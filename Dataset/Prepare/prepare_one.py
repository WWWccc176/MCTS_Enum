from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAVY_BUILD = ROOT / "HeavyBack" / "build"
if str(HEAVY_BUILD) not in sys.path:
    sys.path.insert(0, str(HEAVY_BUILD))

LEVEL_DIRS = {
    "LLL": ROOT / "Dataset" / "LLL" / "Basis",
    "Light": ROOT / "Dataset" / "Light" / "Basis",
    "Heavy": ROOT / "Dataset" / "Heavy" / "Basis",
}


def _write_level(heavy_backend, matrix_id: int, level: str, name: str) -> Path:
    target_dir = LEVEL_DIRS[level]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    text = heavy_backend.dump_matrix(matrix_id)
    if not text.endswith("\n"):
        text += "\n"
    target.write_text(text, encoding="utf-8")
    metrics = dict(heavy_backend.quality_metrics(matrix_id))
    print(
        f"[{level:5s}] {target}  "
        f"b1/GH={float(metrics['b1_over_gh']):.8f}",
        flush=True,
    )
    return target


def prepare_all(raw_text: str, name: str, bkz20_loops: int, bkz30_loops: int):
    import heavy_backend

    # Stage 1: exact LLL(delta=0.999) inside HeavyBack.
    matrix_id = heavy_backend.create_matrix_lll(raw_text)
    outputs: dict[str, Path] = {}
    try:
        dimension = int(heavy_backend.quality_metrics(matrix_id)["dimension"])
        outputs["LLL"] = _write_level(heavy_backend, matrix_id, "LLL", name)

        # Stage 2: Light = LLL + BKZ20.
        light = dict(
            heavy_backend.reduce_bkz2_global(
                matrix_id,
                min(20, dimension),
                max(1, int(bkz20_loops)),
            )
        )
        if not bool(light["completed"]):
            raise RuntimeError(f"BKZ20 failed: {light}")
        outputs["Light"] = _write_level(heavy_backend, matrix_id, "Light", name)

        # Stage 3: Heavy = Light + BKZ30 + HeavyBack's adaptive extreme core.
        heavy_bkz = dict(
            heavy_backend.reduce_bkz2_global(
                matrix_id,
                min(30, dimension),
                max(1, int(bkz30_loops)),
            )
        )
        if not bool(heavy_bkz["completed"]):
            raise RuntimeError(f"BKZ30 failed: {heavy_bkz}")

        # Heavy stage: always use the native whole-basis BGJ sieve.
        extreme = dict(heavy_backend.reduce_sieve_block(matrix_id, 0, dimension))

        if extreme is not None:
            if not bool(extreme["completed"]):
                raise RuntimeError(f"Heavy extreme stage failed: {extreme}")
            print(
                f"[Heavy] backend={extreme['backend']} "
                f"changed={int(bool(extreme['changed']))} "
                f"stop={extreme['stop_reason']}",
                flush=True,
            )

        outputs["Heavy"] = _write_level(heavy_backend, matrix_id, "Heavy", name)
        return outputs
    finally:
        heavy_backend.free_matrix(matrix_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("basis", type=Path)
    parser.add_argument("--bkz20-loops", type=int, default=1)
    parser.add_argument("--bkz30-loops", type=int, default=2)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    raw_text = args.basis.read_text(encoding="utf-8")
    name = args.name or args.basis.name
    prepare_all(raw_text, name, args.bkz20_loops, args.bkz30_loops)


if __name__ == "__main__":
    main()
