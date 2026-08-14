from __future__ import annotations

import hashlib
import json
import shlex
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_identity(model, version, basis, node_budget, refresh_cycles, runtime, extra=None):
    basis = basis.resolve()
    algorithm_params = {
        "basis_sha256": _sha256(basis),
        "node_budget_per_phase": node_budget,
        "refresh_cycles": refresh_cycles,
        "runtime": asdict(runtime),
        "extra": extra or {},
    }
    canonical = json.dumps(algorithm_params, sort_keys=True, separators=(",", ":"))
    parameter_id = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    now = datetime.now().astimezone()
    tag = now.strftime("%Y%m%d-%H%M%S-%f")[:-3] + "_" + parameter_id
    metadata = {
        "model": model,
        "code_version": version,
        "basis_path": str(basis),
        "basis_sha256": algorithm_params["basis_sha256"],
        "parameter_id": parameter_id,
        "started_at": now.isoformat(timespec="seconds"),
        "node_budget_per_phase": node_budget,
        "refresh_cycles": refresh_cycles,
        "phase_limit": refresh_cycles + 1,
        "refresh_method": "LLL",
        "command": shlex.join(sys.argv),
        "runtime": asdict(runtime),
    }
    if extra:
        metadata.update(extra)
    return tag, metadata


def append_summary(result_dir: Path, metadata: dict, resource: dict) -> None:
    def emit(prefix, value, out):
        if isinstance(value, dict):
            for key in sorted(value):
                emit(f"{prefix}.{key}" if prefix else key, value[key], out)
        elif isinstance(value, (list, tuple)):
            out.append(f"{prefix}=" + ",".join(str(x) for x in value))
        elif isinstance(value, bool):
            out.append(f"{prefix}={int(value)}")
        else:
            out.append(f"{prefix}={value}")

    lines = ["", "[run_parameters]"]
    for key in sorted(metadata):
        emit(key, metadata[key], lines)
    for key in sorted(resource):
        emit(f"resource.{key}", resource[key], lines)
    with (result_dir / "summary.txt").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
