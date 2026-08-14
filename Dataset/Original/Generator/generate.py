from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASIS_DIR = HERE.parent / "Basis"


def _ensure_generator() -> Path:
    exe = HERE / "generate_random"
    if exe.is_file():
        return exe

    makefile = HERE / "Makefile"
    source = HERE / "generate_random.cpp"
    if not makefile.is_file() or not source.is_file():
        raise FileNotFoundError(
            "generate_random was not found and the generator sources are incomplete in "
            f"{HERE}"
        )

    subprocess.run(
        ["make", "generate_random"],
        cwd=HERE,
        check=True,
    )
    if not exe.is_file():
        raise RuntimeError("make completed but generate_random was not produced")
    return exe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", "--dimension", dest="dimension", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    if args.dimension <= 0:
        raise ValueError("dimension must be positive")

    exe = _ensure_generator()
    proc = subprocess.run(
        [str(exe), "--dim", str(args.dimension), "--seed", str(args.seed)],
        cwd=HERE,
        check=True,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )

    basis_text = proc.stdout.strip()
    if not basis_text:
        raise RuntimeError("generator returned an empty basis")

    BASIS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.name or f"dim{args.dimension}_seed{args.seed}.txt"
    target = BASIS_DIR / name
    target.write_text(basis_text + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
