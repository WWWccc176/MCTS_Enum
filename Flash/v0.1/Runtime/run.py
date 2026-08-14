from __future__ import annotations

import argparse
from pathlib import Path

from .config import RuntimeConfig
from .parallelism import apply_cpu_plan, build_resource_plan
from .run_meta import build_identity
from .runner import RunSpec, run_flash
from .version import MODEL, VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-budget", type=int, required=True,
                        help="node budget for EACH phase; -1 means unlimited")
    parser.add_argument("--refresh-cycles", type=int, default=0,
                        help="number of LLL refreshes; total phases = refresh-cycles + 1")
    parser.add_argument("basis", nargs="+", type=Path)
    args = parser.parse_args()
    if args.node_budget == 0 or args.node_budget < -1:
        parser.error("--node-budget must be -1 or a positive integer")
    if args.refresh_cycles < 0:
        parser.error("--refresh-cycles must be >= 0")

    runtime = RuntimeConfig()
    plan = build_resource_plan(runtime.cpu_fraction)
    apply_cpu_plan(plan)
    print(f"resource plan: cpu={plan.cpu_threads}/{plan.cpu_total} selected", flush=True)

    for basis in args.basis:
        basis = basis.resolve()
        tag, metadata = build_identity(
            MODEL, VERSION, basis, args.node_budget, args.refresh_cycles, runtime
        )
        run_id = f"{basis.stem}/{tag}"
        result_root = PROJECT_ROOT / "Result" / MODEL
        result_dir = result_root / VERSION / basis.stem / tag
        resource = {
            "cpu_threads": plan.cpu_threads,
            "cpu_total": plan.cpu_total,
        }
        print(
            f"[RUN] model={MODEL} version={VERSION} basis={basis.name} "
            f"parameter_id={metadata['parameter_id']} phases={args.refresh_cycles + 1}",
            flush=True,
        )
        run_flash(
            RunSpec(
                basis_path=basis,
                node_budget=args.node_budget,
                refresh_cycles=args.refresh_cycles,
                run_id=run_id,
                result_root=result_root,
                result_dir=result_dir,
                version=VERSION,
                metadata=metadata,
            ),
            plan.cpu_threads,
            runtime,
            resource,
        )


if __name__ == "__main__":
    main()
