# MCTS_Enum

MCTS-guided exact lattice enumeration experiments. The repository keeps basis preparation, search, and neural learning as separate layers.

## Current versions

- `Flash/v0.1`: full-depth persistent MCTS baseline. Every coefficient level is represented by the MCTS tree.
- `Flash/v0.2`: hybrid CPU search. MCTS schedules upper subtrees; fplll Schnorr-Euchner/KFP enumeration completes the lower subtree.
- `Net/v0.1`: policy/value version of the v0.1 search interface. It is intentionally not rewritten as v0.2 until the Flash hybrid executor is validated experimentally.

Flash is CPU-only. CUDA/BGJ belongs to `HeavyBack`; Net uses PyTorch CUDA for neural inference/training.

## Dataset levels

`Dataset/Prepare/prepare_one.py` is the source of truth for preparation levels:

- `Original`: raw generated basis.
- `LLL`: `LLL(delta=0.999)`.
- `Light`: LLL followed by one configurable global BKZ20 stage.
- `Heavy`: Light followed by configurable BKZ30 and the native whole-basis BGJ sieve.

**BKZ20 is a dataset-preparation stage for Light/Heavy. Flash v0.2 basis refresh does not run BKZ20.**

## Flash v0.2 search

For every phase, v0.2 scans every basis row and initializes the exact search radius with

```text
R0 = min_i ||b_i||.
```

LLL does not sort raw Euclidean row lengths, so the shortest basis row does not have to be `b1`.

The search is split at `--rollout-dimensions`:

```text
upper coefficients                 lower coefficients
z_n -> ... -> z_(m+1)              z_m -> ... -> z_1
        MCTS                 ->     fplll SE/KFP subtree enumeration
```

The upper tree uses PUCT, progressive widening, and lazy Schnorr-Euchner action generation. Once a selected path reaches the rollout frontier, one worker enumerates that complete lower subtree with its own fplll GSO/enumeration state outside the global tree lock. Returned coefficient vectors are re-evaluated with exact GMP arithmetic before they can change the incumbent.

The bottom enumerator uses fplll `MatGSO<..., long double>` with `GSO_ROW_EXPO` and fplll's native enumeration node counters. The persistent tree therefore stores only upper prefixes instead of millions of deep enumeration states.

### Radius and refresh are different objects

- `phase_best` / `R`: shortest exact vector known in the current phase. A shorter exact rollout result tightens the radius immediately.
- `overall_best`: shortest exact vector seen across all phases. Basis refresh can never erase an earlier best result.
- `refresh_candidate`: shortest non-basis rollout vector retained for basis restructuring. It is maintained independently from the incumbent logic.

When a refresh is requested, v0.2 performs:

```text
insert non-basis candidate
-> LLL on n+1 rows
-> remove the dependent zero row
-> LLL on n rows
-> exact determinant check
-> non-worsening GSO-potential transaction gate
-> commit or rollback
```

There is no BKZ20 call in this refresh path.

## Flash v0.2 budget

In v0.1, `--node-budget` counted persistent MCTS tree nodes. In v0.2 the same CLI name denotes an approximate per-phase **work budget**:

```text
work_nodes = persistent MCTS tree nodes + fplll enumeration nodes.
```

This makes the accounting much closer to actual search work. Because multiple subtree rollouts run concurrently and a native fplll call is atomic from the scheduler's point of view, a phase may finish slightly above the requested work boundary.

## Flash v0.2 parallelism

`--cpu-fraction` selects a fraction of CPUs visible to the process; the default is `0.85`. Each search worker owns an independent fplll GSO/enumeration context. The shared mutex protects short MCTS selection/commit operations, while the expensive lower-subtree enumeration runs outside that mutex.

The current Flash v0.2 path does not use a GPU.

## Build

The server environment needs Python/pybind11, CMake, C++17, GMP/GMPXX and fplll development files.

Build only Flash v0.2:

```bash
cd ~/projects/MCTS_Enum
conda activate drl_env
rm -rf Flash/v0.2/build
cd Flash/v0.2/MCTSBack
PYTHON_BIN="$CONDA_PREFIX/bin/python" BUILD_DIR="../build" ./build.sh
cd ~/projects/MCTS_Enum
```

The generic repository build also works, but additionally rebuilds shared preparation components:

```bash
cd ~/projects/MCTS_Enum
conda activate drl_env
PYTHON_BIN="$CONDA_PREFIX/bin/python" bash build_all.zsh Flash v0.2 --clean
```

## Run

Small smoke run:

```bash
cd ~/projects/MCTS_Enum
conda activate drl_env
./run.zsh Flash v0.2 \
  --reduction-level LLL \
  --node-budget 200000 \
  --refresh-cycles 0 \
  --rollout-dimensions 10 \
  Dataset/LLL/Basis/dim35_seed0.txt
```

A useful v0.1/v0.2 scaling check is:

```bash
./run.zsh Flash v0.2 \
  --reduction-level LLL \
  --node-budget 9000000 \
  --refresh-cycles 2 \
  --rollout-dimensions 10 \
  Dataset/LLL/Basis/dim37_seed0.txt \
  Dataset/LLL/Basis/dim40_seed0.txt
```

Run every LLL basis:

```bash
./run.zsh Flash v0.2 \
  --reduction-level LLL \
  --node-budget 9000000 \
  --refresh-cycles 2 \
  --rollout-dimensions 10 \
  Dataset/LLL/Basis/*.txt
```

## Flash v0.2 result records

Results are written under:

```text
Result/Flash/v0.2/<reduction-level>/<basis>/<run-id>/
```

Important files:

- `summary.txt`: tree/enumeration/work counts, rollout count, best ratio, phase and refresh statistics.
- `depth.txt`: persistent-tree depth counts plus native fplll enumeration nodes by coefficient level.
- `best.txt`: exact per-phase best coefficients/vector/norm.
- `path.txt`: geometry and MCTS statistics for the best path when available.
- `basis.txt`: phase input bases and accepted/rejected refresh outputs.
- `speed.csv`: sampled work rate, tree/enum split, rollout count, current radius and global best.
- `params.txt`: Python-side run identity and resolved parameters.

## Numerical boundary

All accepted candidate vectors are reconstructed as exact integers and their squared norms are recomputed with GMP. The fplll lower enumerator itself uses long-double GSO with row-exponent scaling; v0.2 therefore uses a mature high-performance enumeration kernel, but it is not an MPFR formal error-bound certificate.

See `FILE_GUIDE.md` for the implementation map and `DEPENDENCIES.md` for dependencies.


## SVP Challenge generator compatibility

`Dataset/Original/Generator/generate_random` is built against the private
`mcts_env/ntl-9.3.0` installation. This is required for compatibility with
the official SVP Challenge: NTL 9.4 changed the pseudorandom generator, so
using NTL 9.4+ with the same dimension and seed produces a different lattice.

Dataset levels follow the implementation:
- `Original`: official-compatible raw SVP Challenge basis.
- `LLL`: `LLL(delta=0.999)`.
- `Light`: LLL followed by BKZ20.
- `Heavy`: Light followed by BKZ30 and the whole-basis BGJ sieve.

BKZ20 is a dataset preprocessing stage for `Light`/`Heavy`; it is not an
implicit Flash search-cycle operation.
