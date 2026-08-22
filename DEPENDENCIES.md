# Dependencies

## Flash v0.2 native search

- C++17 compiler
- CMake >= 3.24
- Python 3.13-compatible environment
- pybind11 available to that Python interpreter
- GMP / GMPXX development libraries
- fplll development library and headers
- pthreads / C++ standard threads

Flash v0.2 directly uses fplll `MatGSO` and Schnorr-Euchner/KFP enumeration for lower-subtree execution. It does not require CUDA, NTL or the HeavyBack sieve at runtime.

## Dataset preparation / HeavyBack

HeavyBack additionally requires its existing fplll/NTL/OpenMP/CUDA toolchain and optional NUMA support. It creates the preparation levels:

```text
LLL   = LLL(delta=0.999)
Light = LLL + BKZ20
Heavy = Light + BKZ30 + whole-basis BGJ sieve
```

BKZ20 is not part of Flash v0.2 basis refresh.

## Net v0.1

Net v0.1 additionally requires PyTorch. CUDA is used for neural inference/training when available.

## Server build example

```bash
cd ~/projects/MCTS_Enum
conda activate drl_env
rm -rf Flash/v0.2/build
cd Flash/v0.2/MCTSBack
PYTHON_BIN="$CONDA_PREFIX/bin/python" BUILD_DIR="../build" ./build.sh
```
