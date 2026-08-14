# File Guide — MCTS-in-Enumeration SVP v0.1

This file is the implementation map for the repository. Runtime responsibility is intentionally split into Heavy preparation, lightweight exact MCTS search, and neural learning.

## Repository root

- `.gitignore` — excludes native build products, Python caches, generated datasets/results, and local transient files.
- `README.md` — architecture, invariants, run commands, numerical design, versioning, and result schema.
- `VERSION.json` — machine-readable algorithm/variant/record/protocol version identifiers.
- `DEPENDENCIES.md` — Python/C++/CUDA toolchain and server build requirements.
- `FILE_GUIDE.md` — this per-file implementation map.
- `MANIFEST.sha256` — SHA-256 manifest for every shipped file except the manifest itself.
- `build_all.sh` — builds Generator, HeavyBack, and MCTSBack using one selected Python interpreter.
- `run_pipeline.py` — same-process raw-basis -> HeavyBack -> in-memory `MCTSBAS1` -> MCTSBack pipeline. It never needs a Heavy basis file for runtime handoff; `--record-heavy` is an explicit research-record option only.

## Dataset

### `Dataset/Original`

- `Dataset/Original/Generator/generate.py` — thin Python dataset wrapper around the native generator; parses CLI arguments and deliberately writes a raw-basis record plus schema/seed metadata.
- `Dataset/Original/Basis/.gitkeep` — keeps the raw dataset directory in Git before generated data exist.

### `Dataset/Heavy`

- `Dataset/Heavy/Prepare/prepare_one.py` — thin Python wrapper around HeavyBack. Heavy computation remains native: initial LLL, BKZ30, whole-basis GPU sieve, quality gate. Its function returns packet/text/report in memory; CLI mode deliberately writes a prepared dataset record.
- `Dataset/Heavy/Basis/.gitkeep` — keeps the prepared-basis dataset directory in Git.

## Generator

- `Generator/CMakeLists.txt` — builds the native `lattice_generator` pybind11 module against GMP/GMPXX.
- `Generator/build.sh` — reproducible Release build wrapper using `PYTHON_BIN`, `BUILD_DIR`, and `BUILD_JOBS`.
- `Generator/src/generator.cpp` — GMP random dense integer lattice generator. A modular Gaussian-elimination rank test rejects singular candidates before returning the basis.

## HeavyBack — independent heavy preparation backend

### Build / public backend

- `HeavyBack/CMakeLists.txt` — builds the CUDA/C++ heavy backend, detects AVX-512/native host path, links fplll/NTL/GMP/OpenMP/CUDA/optional NUMA, and compiles the inherited BGJ engine.
- `HeavyBack/build.sh` — server-oriented CMake build wrapper; auto-detects visible GPU count unless overridden and exposes CUDA arch, memory budgets, NUMA, Python, and build parallelism through environment variables.
- `HeavyBack/src/backend/types.hpp` — shared backend constants/result structures, including action limits, sieve/BGJ routing thresholds, and reduction result semantics.
- `HeavyBack/src/backend/matrix_utils.hpp` — declarations for exact matrix parsing/serialization, scaled extraction, GSO/statistical helpers, and transaction metrics.
- `HeavyBack/src/backend/matrix_utils.cpp` — implementation of matrix parsing, exact/scale-aware matrix utilities, GSO metrics, and validation helpers used by all heavy reducers.
- `HeavyBack/src/backend/bkz2_engine.hpp` — global/local BKZ2 engine interface.
- `HeavyBack/src/backend/bkz2_engine.cpp` — fplll BKZ2 execution and strategy setup with explicit completion/result semantics.
- `HeavyBack/src/backend/enumeration_engine.hpp` — local enumeration engine interface.
- `HeavyBack/src/backend/enumeration_engine.cpp` — exact/local enumeration reduction path, stopping/acceptance logic, and basis update handling.
- `HeavyBack/src/backend/extreme_reducer.hpp` — dispatcher interface for heavy reduction modes.
- `HeavyBack/src/backend/extreme_reducer.cpp` — chooses/coordinates enumeration, BKZ, and sieve reductions and enforces non-worsening transaction rules.
- `HeavyBack/src/backend/sieve_bridge.hpp` — C++ bridge from exact MPZ blocks into the BGJ/DH engine.
- `HeavyBack/src/backend/sieve_bridge.cpp` — converts between fplll matrices and sieve-engine representations, selects BGJ2/BGJ3 according to supported dimension, executes one sieve call, and recovers exact candidate blocks.
- `HeavyBack/src/backend/lattice_backend.cpp` — pybind11 module `heavy_backend`; owns the in-process matrix pool and public APIs for LLL/BKZ/sieve/evaluation. It also implements exact determinant quality metrics and the binary `MCTSBAS1` in-memory export used by MCTSBack.
- `HeavyBack/src/cuda/lattice_cuda.h` — declarations for small generic CUDA feature kernels used by the heavy Python-facing backend.
- `HeavyBack/src/cuda/lattice_cuda.cu` — CUDA implementations for Gram/cosine feature calculations where GPU use is worthwhile.

### BGJ/DH engine configuration and data structures

- `HeavyBack/src/sieve_engine/include/config.h` — compile-time hardware/cache/kernel constants for the inherited sieve engine; local build definitions override the important resource budgets.
- `HeavyBack/src/sieve_engine/include/vec.h` — aligned vector primitive declarations used by CPU sieve/GSO code.
- `HeavyBack/src/sieve_engine/include/quad.h` — NTL `quad_float` vector/matrix primitive declarations for higher-precision lattice arithmetic.
- `HeavyBack/src/sieve_engine/include/lattice.h` — `Lattice_QP` representation, GSO/LLL/dual/lattice transformation interfaces.
- `HeavyBack/src/sieve_engine/include/utils.h` — aligned allocation, low-level memory, conversion, and general sieve utility declarations/macros.
- `HeavyBack/src/sieve_engine/include/sampler.h` — random/discrete-Gaussian sampling declarations.
- `HeavyBack/src/sieve_engine/include/UidTable.h` — concurrent UID deduplication table used to reject duplicate sieve vectors.
- `HeavyBack/src/sieve_engine/include/pool_hd.h` — central host-side vector pool, cache managers, logging/profiling, chunk state, and BGJ pool interfaces.
- `HeavyBack/src/sieve_engine/include/bgj_hd.h` — BGJ host-side bucket/solution cache manager types and profiling wrappers.
- `HeavyBack/src/sieve_engine/include/common_device.h` — common CUDA error checks and device helper primitives.
- `HeavyBack/src/sieve_engine/include/random_device.h` — cuRAND state and random center-generation kernel declarations.
- `HeavyBack/src/sieve_engine/include/pool_hd_device.h` — CUDA-resident pool state, packed vector layouts, device-side conversion and pool operation interfaces.
- `HeavyBack/src/sieve_engine/include/bgj_hd_device.h` — BGJ2/BGJ3 GPU bucketing/reduction/filtering declarations and reducer/bucketer orchestration structures.
- `HeavyBack/src/sieve_engine/include/dh_device.h` — DH bucketer/reducer buffers, kernel parameters, synchronization state, and atomic device-initialization state.

### BGJ/DH engine host implementations

- `HeavyBack/src/sieve_engine/src/vec.cpp` — generic aligned SIMD vector arithmetic implementation retained from the source engine.
- `HeavyBack/src/sieve_engine/src/vec_native.cpp` — `-march=native` vector arithmetic implementation selected when the dedicated AVX-512 path is unavailable.
- `HeavyBack/src/sieve_engine/src/vec_avx512.cpp` — explicit AVX-512 vector arithmetic implementation selected only when CPU and compiler support all required subsets.
- `HeavyBack/src/sieve_engine/src/quad.cpp` — `quad_float` vector/matrix arithmetic primitives.
- `HeavyBack/src/sieve_engine/src/utils.cpp` — aligned allocations and general host utility implementation.
- `HeavyBack/src/sieve_engine/src/lattice.cpp` — `Lattice_QP` allocation, conversion, serialization, and general lattice-state implementation.
- `HeavyBack/src/sieve_engine/src/lll.cpp` — high-precision GSO/LLL operations for the sieve engine lattice representation.
- `HeavyBack/src/sieve_engine/src/dual.cpp` — dual-lattice construction routines.
- `HeavyBack/src/sieve_engine/src/sampler.cpp` — discrete-Gaussian/random sampling implementation.
- `HeavyBack/src/sieve_engine/src/UidTable.cpp` — concurrent UID table insertion/check/erase and inherited spill/load support.
- `HeavyBack/src/sieve_engine/src/pool_hd.cpp` — generic host vector-pool/cache implementation and basis-consistency logic.
- `HeavyBack/src/sieve_engine/src/pool_hd_native.cpp` — native-SIMD host vector-pool implementation chosen on non-AVX512 builds.
- `HeavyBack/src/sieve_engine/src/pool_hd_avx512.cpp` — AVX-512 host vector-pool implementation.
- `HeavyBack/src/sieve_engine/src/bgj_hd.cpp` — BGJ bucket-cache and solution-cache host managers.
- `HeavyBack/src/sieve_engine/src/pool_hd_device.cu` — CUDA pool allocation, pinned/device memory, packing/conversion, cache transfers, and GPU-facing pool operations.
- `HeavyBack/src/sieve_engine/src/bgj_hd_device.cu` — main BGJ GPU orchestration for bucketer/reducer streams, BGJ2/BGJ3 strategies, device buffers, and safe host-side output-count validation.
- `HeavyBack/src/sieve_engine/src/dh_device.cu` — DH host/device orchestration. Includes bounded waits, hard time checks, atomic initialization, reduced logging frequency, synchronized counter copies, and defensive output-count validation.

### CUDA kernels

- `HeavyBack/src/sieve_engine/src/kernel_buc.cu` — BGJ bucket construction kernels.
- `HeavyBack/src/sieve_engine/src/kernel_ctr.cu` — cuRAND initialization and BGJ center-generation kernels.
- `HeavyBack/src/sieve_engine/src/kernel_dhb.cu` — DH bucketing kernels and DH device constants.
- `HeavyBack/src/sieve_engine/src/kernel_dhr.cu` — DH reduction kernels. Output slots use saturating/CAS reservation so a full buffer cannot drive the signed counter through `INT_MAX`.
- `HeavyBack/src/sieve_engine/src/kernel_red.cu` — standard BGJ reduction kernels with capacity-bounded output reservation.
- `HeavyBack/src/sieve_engine/src/kernel_mred.cu` — multi-reduction BGJ kernels with the same bounded output-counter discipline.
- `HeavyBack/src/sieve_engine/src/kernel_flt.cu` — filtering/compaction stage; safely handles a full global output buffer and clears non-committable accumulator tails.
- `HeavyBack/src/sieve_engine/src/kernel_packf.cu` — packed-vector device pack/unpack kernels.
- `HeavyBack/src/sieve_engine/src/kernel_pop.cu` — GPU population/vector-consistency checking helpers.
- `HeavyBack/src/sieve_engine/src/kernel_v2h.cu` — vector-to-DH-head/device conversion helpers used by the DH path.

### Vendored dependencies

- `HeavyBack/src/sieve_engine/dep/thread_pool/thread_pool.hpp` — vendored lightweight C++ thread pool used by the inherited sieve engine.
- `HeavyBack/src/sieve_engine/dep/phmap/LICENSE` — Parallel Hashmap license.
- `HeavyBack/src/sieve_engine/dep/phmap/btree.h` — vendored phmap B-tree containers.
- `HeavyBack/src/sieve_engine/dep/phmap/meminfo.h` — vendored process-memory inspection helpers.
- `HeavyBack/src/sieve_engine/dep/phmap/phmap.h` — main Parallel Hashmap public containers.
- `HeavyBack/src/sieve_engine/dep/phmap/phmap_base.h` — phmap hash-table implementation base.
- `HeavyBack/src/sieve_engine/dep/phmap/phmap_bits.h` — low-level phmap bit/control-byte helpers.
- `HeavyBack/src/sieve_engine/dep/phmap/phmap_config.h` — phmap platform/compiler configuration.
- `HeavyBack/src/sieve_engine/dep/phmap/phmap_dump.h` — phmap binary serialization helpers inherited by UID spill support.
- `HeavyBack/src/sieve_engine/dep/phmap/phmap_fwd_decl.h` — phmap forward declarations.
- `HeavyBack/src/sieve_engine/dep/phmap/phmap_utils.h` — phmap utility traits/helpers.

## MCTSBack — independent lightweight coefficient-search backend

- `MCTSBack/CMakeLists.txt` — C++17/pybind/fplll/GMP/OpenMP build; intentionally declares no CUDA, BGJ, or sieve dependency.
- `MCTSBack/build.sh` — Release build wrapper using the selected Python/Conda environment.
- `MCTSBack/include/mcts/types.hpp` — algorithm versions, fixed v0.1 defaults, search configuration, edge statistics, path-record schema, and NN metric schema.
- `MCTSBack/include/mcts/basis.hpp` — exact MPZ basis abstraction and `MCTSBAS1` protocol interface.
- `MCTSBack/src/basis.cpp` — text/packet parsing, exact `Bz`, exact squared norm, Bareiss determinant, and log/power-of-two scale helpers.
- `MCTSBack/include/mcts/gso.hpp` — fixed-phase GSO data interface (`mu`, scaled `g`, GH, static normalized NN features).
- `MCTSBack/src/gso.cpp` — one-global-scale two-pass modified Gram-Schmidt; exact determinant GH and static log-GSO normalization.
- `MCTSBack/include/mcts/enumeration_geometry.hpp` — exact-search geometry API for center, legal interval, candidates, and next accumulated bound.
- `MCTSBack/src/enumeration_geometry.cpp` — computes `c_k`, conservatively guarded `Delta_k`, strict integer legal interval, SE ordering, and incremental `rho`.
- `MCTSBack/include/mcts/tree.hpp` — compact node/tree structures; nodes store parent/action/rho/statistics instead of a full coefficient vector.
- `MCTSBack/src/tree.cpp` — root reset, compact child creation, and per-depth node counting.
- `MCTSBack/include/mcts/search_engine.hpp` — public search engine plus inference/training request schemas and internal pending-evaluation state.
- `MCTSBack/src/search_engine.cpp` — complete MCTS engine: dynamic legal actions, PUCT+M, progressive widening, SE/policy proposal order, exact terminal GMP evaluation, global radius rescan, Flash CPU parallelism, Pro inference requests, training targets, 5000-node full-tree refresh support, result snapshots, and phase rebuild.
- `MCTSBack/include/mcts/basis_refresh.hpp` — the single public update interface for optional basis refresh.
- `MCTSBack/src/basis_refresh.cpp` — the single implementation file for refresh: insert best exact vector, LLL dependency removal, BKZ20 scan, final LLL, return a new basis. A successful change ends the old phase and requires a fresh tree/GSO.
- `MCTSBack/include/mcts/result_recorder.hpp` — common Flash/Pro C++ result schema and phase snapshot structures.
- `MCTSBack/src/result_recorder.cpp` — the only core experiment-result writer. Writes common version/config/NN metadata, phase summaries, node counts, exact best vector/coefficient records, complete best-path rows, refreshed bases, and optional NN checkpoint bytes.
- `MCTSBack/src/pybind_module.cpp` — pybind module `mcts_enum_backend`; exposes search configuration, in-memory basis construction, Flash search, Pro inference/training/full-refresh calls, refresh API, exact best result access, and C++ result writer.

## NN — Python learning plane

- `NN/__init__.py` — package marker.
- `NN/version.py` — canonical `v0.1flash` and `v0.1pro` strings.
- `NN/config.py` — neural architecture, optimizer/replay settings, inference sizes, 5000-node refresh interval, 0.995 gate, `w_M=0.25`, numerical guards, 85% CPU fraction, and candidate-padding cap.
- `NN/backend_loader.py` — imports the already-built native `heavy_backend` / `mcts_enum_backend` modules from their independent build directories.
- `NN/engine_config.py` — copies Python runtime configuration into the C++ `SearchConfig` object; no search mathematics are implemented here.
- `NN/parallelism.py` — detects CPU affinity and all visible CUDA devices, reserves 85% of visible logical CPUs, and applies thread-affinity environment settings.
- `NN/batching.py` — converts only NN feature payloads into float32 tensors, pads dynamic candidate sets, builds masks, caches fixed-phase GSO tensors, and performs pinned nonblocking H2D transfers.
- `NN/model.py` — dynamic-action policy/value network. GSO levels are encoded and pooled; state and each legal candidate are encoded separately; policy produces one logit per candidate and value produces one scalar.
- `NN/inference.py` — no-grad model inference adapter returning variable-length policy logits and scalar values to C++.
- `NN/inference_pool.py` — creates one model replica per visible GPU, performs candidate-count-aware microbatching, load-balances batches by padded candidate cost, runs GPUs concurrently, and synchronizes replicas after training.
- `NN/replay.py` — bounded in-memory replay buffer for training samples; no replay dataset is written to disk.
- `NN/trainer.py` — AdamW training, visit-count cross-entropy policy loss, masked best-terminal value loss, gradient clipping, and static-GSO cache management.
- `NN/pro_loop.py` — Pro control loop: inference -> C++ backup/search -> every 5000 new nodes collect training samples -> train primary -> synchronize all GPU replicas -> full-tree refresh. Checkpoint serialization uses in-memory `BytesIO` and is handed to the C++ recorder as bytes.
- `NN/runner.py` — Flash/Pro execution helpers for already-prepared basis files; all search work remains C++.
- `NN/run.py` — multi-basis CLI; applies the 85% CPU resource plan and executes jobs sequentially so each Pro run may use every visible GPU.
- `NN/run_flash.py` — Flash CLI alias.
- `NN/run_pro.py` — Pro CLI alias.

## Result

- `Result/v0.1flash/.gitkeep` — keeps the Flash result root in Git before runs exist.
- `Result/v0.1pro/.gitkeep` — keeps the Pro result root in Git before runs exist.

Actual result files are generated by `MCTSBack/src/result_recorder.cpp`, so Flash and Pro use one schema and one code path.

## Tests

- `Tests/validate_source.py` — static architectural invariants: MCTSBack contains no BGJ/CUDA/sieve code, fixed version/0.995/5000/0.25 decisions are present, and basis refresh remains exactly two files.
- `Tests/smoke.py` — after native build, runs a tiny Flash search and a tiny Pro inference loop, verifies a shorter vector is found, and exercises common result writing.
