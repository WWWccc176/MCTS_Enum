from __future__ import annotations

import csv
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

def root_dir():
    for p in (HERE, *HERE.parents):
        if (p / "Dataset").is_dir() and (p / "Result").is_dir():
            return p
    raise RuntimeError("cannot find project root")

ROOT = root_dir()
LLL = ROOT / "Dataset/LLL/Basis"
HEAVY = ROOT / "Dataset/Heavy/Basis"
MCTS = ROOT / "Result/Flash/v0.1"
OUT = HERE / "all_sum.csv"

def load_basis(path):
    rows = []
    text = path.read_text(encoding="utf-8")
    for body in re.findall(r"\[([^\[\]]+)\]", text, re.S):
        values = body.split()
        if values:
            rows.append([int(x) for x in values])
    if not rows or any(len(r) != len(rows) for r in rows):
        raise ValueError(f"invalid basis: {path}")
    return rows

def det_abs(matrix):
    a = [r[:] for r in matrix]
    n = len(a)
    if n == 1:
        return abs(a[0][0])
    prev = 1
    sign = 1
    for k in range(n - 1):
        if a[k][k] == 0:
            q = next((i for i in range(k + 1, n) if a[i][k] != 0), None)
            if q is None:
                return 0
            a[k], a[q] = a[q], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, n):
            aik = a[i][k]
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * pivot - aik * a[k][j]) // prev
            a[i][k] = 0
        prev = pivot
    return abs(sign * a[-1][-1])

def log_int(x):
    if x <= 0:
        raise ValueError("non-positive integer")
    bits = x.bit_length()
    if bits <= 53:
        return math.log(float(x))
    shift = bits - 53
    return math.log(float(x >> shift)) + shift * math.log(2.0)

def log_gh(matrix):
    n = len(matrix)
    d = det_abs(matrix)
    if d == 0:
        raise ValueError("singular basis")
    return (
        math.lgamma(n / 2.0 + 1.0)
        - (n / 2.0) * math.log(math.pi)
        + log_int(d)
    ) / n

def shortest_over_gh(matrix):
    lg = log_gh(matrix)
    sq = min(sum(x * x for x in row) for row in matrix if any(x != 0 for x in row))
    return math.exp(0.5 * log_int(sq) - lg)

def parse_summary(path):
    global_data = {}
    phases = {}
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.fullmatch(r"\[phase\s+(\d+)\]", line)
        if m:
            idx = int(m.group(1))
            current = phases.setdefault(idx, {"phase": idx})
            continue
        if line.startswith("[") and line.endswith("]"):
            current = None
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if current is None:
            global_data[k] = v
        else:
            current[k] = v
    return global_data, [phases[k] for k in sorted(phases)]

def latest_summary(stem):
    files = []
    for d in (MCTS / "LLL" / stem, MCTS / stem):
        if d.is_dir():
            files.extend(d.glob("*/summary.txt"))
    return max(files, key=lambda p: p.parent.name) if files else None

def mcts_stats(summary, initial_ratio):
    g, phases = parse_summary(summary)

    best_ratio = initial_ratio
    real_nodes = 0
    cycles = 1
    before = 0

    for phase in phases:
        ratio = phase.get("best_over_gh")
        if ratio is not None:
            ratio = float(ratio)
            if ratio < best_ratio - 1e-12:
                best_ratio = ratio
                real_nodes = before + int(phase.get("best_found_at_node_count", "0"))
                cycles = int(phase["phase"]) + 1
        before += int(phase.get("total_nodes", "0"))

    if "final_best_over_gh" in g:
        ratio = float(g["final_best_over_gh"])
        if ratio < best_ratio - 1e-12:
            best_ratio = ratio
            real_nodes = int(g.get("final_best_found_at_total_node_count", "0"))
            phase = g.get("final_best_found_in_phase", g.get("final_best_found_phase", "0"))
            cycles = int(phase) + 1

    total_nodes = sum(int(p.get("total_nodes", "0")) for p in phases)
    if not phases:
        total_nodes = int(g.get("total_nodes_all_phases", "0"))

    return best_ratio, real_nodes, cycles, total_nodes

def order(path):
    m = re.fullmatch(r"dim(\d+)_seed(-?\d+)\.txt", path.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 10**9, 10**9

rows = []

for lll_path in sorted(LLL.glob("*.txt"), key=order):
    summary = latest_summary(lll_path.stem)
    if summary is None:
        continue

    lll_ratio = shortest_over_gh(load_basis(lll_path))

    heavy_path = HEAVY / lll_path.name
    heavy_ratio = ""
    if heavy_path.is_file():
        heavy_ratio = shortest_over_gh(load_basis(heavy_path))

    mcts_ratio, real_nodes, cycles, total_nodes = mcts_stats(summary, lll_ratio)
    shorten = max(0.0, lll_ratio - mcts_ratio)

    m = re.fullmatch(r"dim(\d+)_seed(.+)\.txt", lll_path.name)
    rows.append({
        "file": lll_path.name,
        "dimension": m.group(1) if m else "",
        "seed": m.group(2) if m else "",
        "norm/GH(LLL)": f"{lll_ratio:.5f}",
        "norm/GH(Heavy)": "" if heavy_ratio == "" else f"{heavy_ratio:.5f}",
        "norm/GH(MCTS)": f"{mcts_ratio:.5f}",
        "shorten": f"{shorten:.5f}",
        "real nodes": real_nodes,
        "cycles": cycles,
        "Total nodes": total_nodes,
    })

fields = [
    "file",
    "dimension",
    "seed",
    "norm/GH(LLL)",
    "norm/GH(Heavy)",
    "norm/GH(MCTS)",
    "shorten",
    "real nodes",
    "cycles",
    "Total nodes",
]

with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(OUT)
print(f"rows={len(rows)}")
