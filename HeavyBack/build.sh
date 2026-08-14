#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PYBIND11_DIR="$($PYTHON_BIN -m pybind11 --cmakedir)"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

echo "Python:   $PYTHON_BIN"
echo "pybind11: $PYBIND11_DIR"
echo "Build:    $BUILD_DIR"

if [[ -z "${LATTICE_GPU_NUM:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
        if [[ -z "${CUDA_VISIBLE_DEVICES}" || "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
            echo "No CUDA device is visible to HeavyBack" >&2
            exit 1
        fi
        IFS=',' read -r -a _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
        LATTICE_GPU_NUM="${#_visible_gpus[@]}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        LATTICE_GPU_NUM="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)"
    else
        LATTICE_GPU_NUM=1
    fi
fi
if (( LATTICE_GPU_NUM < 1 || LATTICE_GPU_NUM > 8 )); then
    echo "Invalid auto-detected LATTICE_GPU_NUM=$LATTICE_GPU_NUM (supported: 1..8)" >&2
    exit 1
fi

echo "Visible HeavyBack GPUs: $LATTICE_GPU_NUM"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
    -DPython3_EXECUTABLE="$PYTHON_BIN"
    -Dpybind11_DIR="$PYBIND11_DIR"
    -DLATTICE_GPU_NUM="$LATTICE_GPU_NUM"
    -DLATTICE_CUDA_ARCH="${LATTICE_CUDA_ARCH:-native}"
    -DLATTICE_PWC_DRAM_GB="${LATTICE_PWC_DRAM_GB:-3}"
    -DLATTICE_BWC_DRAM_GB="${LATTICE_BWC_DRAM_GB:-6}"
    -DLATTICE_SWC_DRAM_GB="${LATTICE_SWC_DRAM_GB:-2}"
    -DLATTICE_UT_TABLE_DRAM_GB="${LATTICE_UT_TABLE_DRAM_GB:-6}"
    -DLATTICE_UT_BUFFER_DRAM_GB="${LATTICE_UT_BUFFER_DRAM_GB:-3}"
    -DLATTICE_ENABLE_NUMA="${LATTICE_ENABLE_NUMA:-ON}"
)

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CMAKE_ARGS+=( -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" )
fi

cmake -S . -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j"$BUILD_JOBS"
