# MCTS-in-Enumeration SVP v0.1

This repository deliberately separates **basis preparation** from **coefficient search**.

- `HeavyBack`: independent heavy native backend. It owns LLL/BKZ/Enumeration/BGJ/CUDA and prepares a high-quality basis. Handoff requires `b1/GH <= 0.995`.
- `MCTSBack`: independent lightweight MCTS-in-enumeration backend. It has no BGJ, sieve, or CUDA code. The basis is immutable inside one MCTS phase.
- `NN`: the learning plane only. PyTorch supplies dynamic-action policy/value inference, training, multi-GPU model replicas, and batching.

All **runtime module handoff** is in memory. `HeavyBack.export_matrix_packet()` emits an exact binary `MCTSBAS1` packet and `MCTSBack` consumes the bytes directly. Dataset and Result files are deliberate persistent experiment records, never temporary inter-backend IPC.

## Versions

- Algorithm: `v0.1`
- `v0.1flash`: no neural network. C++ MCTS uses a Schnorr-Euchner-centred heuristic prior.
- `v0.1pro`: policy/value network enabled.
- Result record schema: `1.0`
- In-memory basis protocol: `MCTSBAS1`

In Pro, every 5000 newly created nodes trigger: training -> primary model update -> synchronization to every visible GPU replica -> **full-tree** prior/value refresh. There is no lazy refresh and no per-node model versioning.

## Directory layout

```text
MCTS_SVP_v0.1/
├── Dataset/
│   ├── Original/
│   │   ├── Generator/
│   │   └── Basis/
│   └── Heavy/
│       ├── Prepare/
│       └── Basis/
├── Generator/
├── HeavyBack/
├── MCTSBack/
├── NN/
├── Result/
│   ├── v0.1flash/
│   └── v0.1pro/
├── Tests/
├── run_pipeline.py
└── build_all.sh
```

## Computational ownership

### Python

Python owns only the learning/control plane:

- policy/value network;
- batching and candidate masks;
- optimizer/replay/training;
- multi-GPU inference dispatch;
- thin process orchestration.

### C++

C++ owns all search mathematics and exact state:

- basis/GSO;
- exact determinant and exact terminal integer norm;
- `c_k`, `rho`, `Delta_k` and legal integer intervals;
- MCTS tree and node accounting;
- PUCT and `M` extreme-value statistic;
- progressive widening;
- Schnorr-Euchner + policy proposal order;
- global incumbent/radius tightening;
- LLL+BKZ20 phase refresh;
- common Flash/Pro result recording.

### CUDA

Custom CUDA exists only in `HeavyBack` for BGJ/DH sieve. `MCTSBack` is CPU-native. Pro GPU work is neural inference/training through PyTorch.

## Numerical design

A fixed-basis phase computes GSO once. The basis is globally scaled by one power of two before two-pass modified Gram-Schmidt, avoiding per-row scaling that would change enumeration geometry. GH uses an exact GMP Bareiss determinant in the log domain. Terminal vectors and squared norms are exact GMP integers.

Enumeration boundaries are handled conservatively: the floating radius is expanded, accumulated `rho` and `g_k` are biased outward, `Delta` is moved outward with `nextafter`, and only then is the strict integer interval derived. Thus the numerical guard may retain an uncertain boundary action but is designed not to prune one merely because of roundoff.

After any exact incumbent improvement, the shared radius is updated immediately and all initialized nodes are rescanned. Edges outside the new legal interval are disabled globally. Pending NN evaluations whose branch became mathematically invalid are discarded when they return and cannot back up stale values.

## MCTS v0.1 decisions

Fixed design choices:

- heavy-basis gate: `b1/GH <= 0.995`;
- basis fixed inside one MCTS phase;
- optional refresh: insert best exact vector -> LLL -> BKZ20 -> LLL -> rebuild tree;
- refresh implementation is isolated to exactly `basis_refresh.hpp/.cpp`;
- Pro network update every 5000 new nodes;
- value target: best exact terminal score observed below the node;
- selection: `(1-w_M)Q + w_M M + U`, `w_M = 0.25`;
- proposal: SE + policy mixture, with every legal action remaining reachable;
- search budget: node count;
- CPU plan: 85% of CPUs visible through process affinity;
- Pro: one network replica per visible GPU;
- HeavyBack build: visible GPU count is auto-detected unless `LATTICE_GPU_NUM` is explicitly supplied.

Configuration-only starting values, not research conclusions:

- `lambda_puct = 1.5`;
- `C_PW = 2.0`;
- `d_PW = 0.5`;
- policy/SE mix = `0.5`;
- visit target temperature = `1.0`.

## Dataset preparation

Generate one raw lattice:

```bash
python Dataset/Original/Generator/generate.py --dimension 80 --bits 32 --seed 0
```

Prepare one basis with HeavyBack:

```bash
python Dataset/Heavy/Prepare/prepare_one.py Dataset/Original/Basis/dim80_seed0.txt
```

Heavy preparation performs initial LLL, whole-basis BKZ30, then the native whole-basis GPU sieve stage. The inherited HeavyBack routes BGJ3 at its supported high-dimension threshold and BGJ2 below that threshold; this preserves the stable native engine rather than forcing BGJ3 into unsupported small blocks. Only a basis passing `0.995` is written to `Dataset/Heavy/Basis`.

For actual end-to-end experiments, `run_pipeline.py` does **not** write the Heavy basis as an intermediate file unless `--record-heavy` is explicitly requested.

## Run directly from raw basis

Flash:

```bash
python run_pipeline.py Dataset/Original/Basis/dim80_seed0.txt \
  --mode flash --node-budget 1000000
```

Pro:

```bash
python run_pipeline.py Dataset/Original/Basis/dim80_seed0.txt \
  --mode pro --node-budget 1000000
```

One optional post-search LLL+BKZ20 phase refresh:

```bash
python run_pipeline.py Dataset/Original/Basis/dim80_seed0.txt \
  --mode pro --node-budget 1000000 --refresh-cycles 1
```

## Result schema

Flash and Pro use exactly the same C++ writer and directory schema:

```text
Result/<version>/<run_id>/
├── run.json
├── search_config.json
├── nn_metrics.csv
├── network.json
├── network.pt                 # Pro only when checkpoint supplied
├── final_basis.txt
├── final_coefficients.txt
├── final_vector.txt
├── phase_000/
│   ├── initial_basis.txt
│   ├── summary.json
│   ├── nodes_per_depth.csv
│   ├── best_coefficients.txt
│   ├── best_vector.txt
│   └── best_path.csv
└── phase_001/ ...
```

If phase `p` ends with LLL+BKZ20 refresh it additionally contains `refreshed_basis.txt`; phase `p+1` starts from that basis and records an independent tree.

`best_path.csv` follows the coefficient order `z_n -> ... -> z_1`. Each row records the coefficient/depth, node IDs, action, total nodes at that depth, GSO scale exponent, parent/child accumulated bound, incremental GSO cost, radius/budget, center, `Delta`, legal interval, normalized offset, and edge `P/N/Q/M` statistics.

Scaled squared quantities reconstruct as

```text
real_squared_quantity = recorded_scaled_quantity * 2^(2 * gso_scale_exp2)
```

## Build and validation

```bash
PYTHON_BIN=/home/amax/.conda/envs/drl_env/bin/python ./build_all.sh
python Tests/validate_source.py
python Tests/smoke.py
```

`Tests/validate_source.py` enforces the MCTSBack/no-sieve boundary, version constants, fixed v0.1 decisions, and the exactly-two-file basis-refresh rule.

See `FILE_GUIDE.md` for the per-file implementation map and `DEPENDENCIES.md` for the toolchain.
