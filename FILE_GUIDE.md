# File Guide — MCTS_Enum

## Repository root

- `README.md` — current architecture and commands. This is the authoritative human-readable description.
- `DEPENDENCIES.md` — native/Python/CUDA dependencies.
- `build_all.zsh` — generic versioned build entry point. On a server without zsh, invoke it with `bash build_all.zsh ...`.
- `run.zsh` — generic versioned launcher; runs Python in module mode and loads the backend from the selected version directory.

## Dataset

- `Dataset/Original/Basis` — raw bases.
- `Dataset/LLL/Basis` — LLL(delta=0.999) bases.
- `Dataset/Light/Basis` — LLL + BKZ20 bases.
- `Dataset/Heavy/Basis` — Light + BKZ30 + whole-basis native BGJ sieve bases.
- `Dataset/Prepare/prepare_one.py` — source of truth for the four preparation levels. BKZ20 here is preprocessing; it is not a Flash v0.2 refresh operation.

## HeavyBack

Independent C++/CUDA preparation backend. It owns LLL/BKZ/BGJ and CUDA code used to create Light/Heavy datasets. Flash v0.2 does not link the HeavyBack sieve into its search engine.

## Flash/v0.1

Preserved baseline. It represents the whole coefficient path as a persistent MCTS tree and is retained for direct experimental comparison with v0.2.

## Flash/v0.2

### Runtime

- `Runtime/run.py` — CLI and per-basis dispatch. Exposes work budget, refresh cycles, CPU fraction, rollout depth and MCTS parameters.
- `Runtime/config.py` — v0.2 defaults.
- `Runtime/engine_config.py` — transfers Runtime configuration to native `SearchConfig`.
- `Runtime/parallelism.py` — CPU affinity plan; default 85% of visible logical CPUs.
- `Runtime/backend_loader.py` — loads `Flash/v0.2/build/mcts_enum_backend*.so`.
- `Runtime/runner.py` — phase loop, status monitor, refresh orchestration and `speed.csv` writer.
- `Runtime/run_meta.py` — run fingerprint and `params.txt` metadata.
- `Runtime/version.py` — canonical Flash/v0.2 version string.

### MCTSBack build/API

- `MCTSBack/CMakeLists.txt` — C++17/pybind11/fplll/GMP/pthreads Release backend. No CUDA/BGJ dependency.
- `MCTSBack/build.sh` — version-local native build wrapper.
- `MCTSBack/include/mcts/types.hpp` — v0.2 configuration, edge/path/status/phase schemas.
- `MCTSBack/src/pybind_module.cpp` — Python binding for the Flash v0.2 native engine.

### Exact basis and GSO

- `include/mcts/basis.hpp`, `src/basis.cpp` — exact MPZ basis, packet/text serialization, exact `Bz`, exact squared norm and determinant.
- `include/mcts/gso.hpp`, `src/gso.cpp` — fplll long-double `MatGSO` with `GSO_ROW_EXPO`; exposes `mu`, scaled diagonal, GH and log-potential.

### Upper MCTS geometry/tree

- `include/mcts/enumeration_geometry.hpp`, `src/enumeration_geometry.cpp` — exact-search center/radius geometry with conservative floating guards. It no longer materializes or sorts the whole legal action interval.
- `include/mcts/tree.hpp`, `src/tree.cpp` — persistent upper-prefix tree. Each node stores its selected high-to-low coefficient prefix, lazy SE cursor and MCTS edge statistics.
- `include/mcts/search_engine.hpp`, `src/search_engine.cpp` — hybrid scheduler. A single selection traverses/creates upper nodes until the rollout frontier, then dispatches a complete lower-subtree enumeration. It maintains phase radius, cross-phase global best and the independent non-basis refresh candidate.

### fplll lower-subtree executor

- `include/mcts/fplll_rollout.hpp`, `src/fplll_rollout.cpp` — one private fplll GSO/enumerator per worker. Uses the fplll subtree interface, SE/KFP enumeration and native per-level node counters. The expensive call runs outside the global tree mutex.

### Basis refresh

- `include/mcts/basis_refresh.hpp`, `src/basis_refresh.cpp` — insert retained non-basis vector, LLL on n+1 rows, remove dependency zero row, final LLL, exact determinant check and GSO-potential non-worsening transaction gate.
- There is deliberately **no BKZ20 call** in Flash v0.2 refresh. BKZ20 belongs to Light/Heavy dataset preparation unless a future version explicitly adds and documents another refresh mode.

### Results

- `include/mcts/result_recorder.hpp`, `src/result_recorder.cpp` — writes `summary.txt`, `basis.txt`, `best.txt`, `path.txt` and `depth.txt`.
- `Runtime/runner.py` additionally writes `speed.csv` and parameter metadata.

## Net/v0.1

Current policy/value implementation remains on the v0.1 search interface. It is intentionally separated from Flash v0.2 so the hybrid executor can first be benchmarked without NN/GPU confounders.

## Result

- `Result/Flash/v0.1` — v0.1 baseline results.
- `Result/Flash/v0.2` — generated hybrid-search results.
- `Result/Net/v0.1` — generated neural-search results when used.


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
