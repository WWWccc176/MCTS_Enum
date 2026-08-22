#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
  -DPython3_EXECUTABLE="$PYTHON_BIN"
)
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  CMAKE_ARGS+=( -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" )
fi
cmake -S . -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$BUILD_JOBS"
