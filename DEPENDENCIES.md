# Dependencies

## Python plane

- Python 3.13-compatible environment.
- PyTorch for the policy/value network, optimizer, CUDA inference, and model checkpoint serialization.
- pybind11 for the three native Python modules.

Python does not implement GSO, enumeration geometry, legality, MCTS selection, progressive widening, exact terminal evaluation, basis refresh, or BGJ.

## Native plane

- C++17 compiler.
- CMake >= 3.24.
- GMP / GMPXX for exact integer vectors, exact terminal squared norms, exact determinant, and packet serialization.
- fplll for LLL/BKZ in HeavyBack and the isolated LLL+BKZ20 refresh in MCTSBack.
- OpenMP / pthreads for CPU parallelism.
- NTL for the inherited HeavyBack BGJ engine.
- libnuma is optional in HeavyBack and enabled when present.

## CUDA plane

Only `HeavyBack` requires the CUDA toolkit. `MCTSBack` deliberately has no CUDA source and no sieve dependency. In `v0.1pro`, GPU work is PyTorch neural inference/training: all visible GPUs receive a model replica automatically.

## Server build

```bash
cd /home/amax/projects/MCTS_SVP_v0.1
PYTHON_BIN=/home/amax/.conda/envs/drl_env/bin/python ./build_all.sh
```

`build_all.sh` builds `Generator`, `HeavyBack`, then `MCTSBack` with the same Python interpreter.
