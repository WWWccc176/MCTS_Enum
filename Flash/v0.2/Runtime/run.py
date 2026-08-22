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
    parser.add_argument(
        "--node-budget",
        type=int,
        required=True,
        help=(
            "per-phase work budget = MCTS tree nodes + fplll enumeration nodes; "
            "parallel atomic rollouts may overshoot the boundary"
        ),
    )
    parser.add_argument("--refresh-cycles", type=int, default=0)
    parser.add_argument(
        "--reduction-level",
        choices=("Original", "LLL", "Light", "Heavy"),
        required=True,
    )
    parser.add_argument("--cpu-fraction", type=float, default=0.85)
    parser.add_argument("--rollout-dimensions", type=int, default=10)
    parser.add_argument("--rollout-solutions", type=int, default=8)
    parser.add_argument("--w-m", type=float, default=0.25)
    parser.add_argument("--lambda-puct", type=float, default=1.5)
    parser.add_argument("--c-pw", type=float, default=2.0)
    parser.add_argument("--d-pw", type=float, default=0.5)
    parser.add_argument("--lll-delta", type=float, default=0.999)
    parser.add_argument("basis", nargs="+", type=Path)
    args = parser.parse_args()

    if args.node_budget == 0 or args.node_budget < -1:
        parser.error("--node-budget must be -1 or a positive integer")
    if args.refresh_cycles < 0:
        parser.error("--refresh-cycles must be >= 0")
    if args.rollout_dimensions <= 0:
        parser.error("--rollout-dimensions must be positive")
    if args.rollout_solutions <= 0:
        parser.error("--rollout-solutions must be positive")

    runtime = RuntimeConfig(
        cpu_fraction=args.cpu_fraction,
        rollout_dimensions=args.rollout_dimensions,
        rollout_solutions=args.rollout_solutions,
        w_m=args.w_m,
        lambda_puct=args.lambda_puct,
        cpw=args.c_pw,
        dpw=args.d_pw,
        lll_delta=args.lll_delta,
    )
    plan = build_resource_plan(runtime.cpu_fraction)
    apply_cpu_plan(plan)
    print(f"resource plan: cpu={plan.cpu_threads}/{plan.cpu_total} selected", flush=True)

    for basis in args.basis:
        basis = basis.resolve()
        tag, metadata = build_identity(
            MODEL,
            VERSION,
            basis,
            args.node_budget,
            args.refresh_cycles,
            runtime,
            {"reduction_level": args.reduction_level},
        )
        run_id = f"{args.reduction_level}/{basis.stem}/{tag}"
        result_root = PROJECT_ROOT / "Result" / MODEL
        result_dir = result_root / VERSION / args.reduction_level / basis.stem / tag
        resource = {"cpu_threads": plan.cpu_threads, "cpu_total": plan.cpu_total}
        print(
            f"[RUN] model={MODEL} version={VERSION} reduction={args.reduction_level} "
            f"basis={basis.name} parameter_id={metadata['parameter_id']} "
            f"phases={args.refresh_cycles + 1}",
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
