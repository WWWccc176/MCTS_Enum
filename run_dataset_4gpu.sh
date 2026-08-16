#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

GPU_COUNT=4
CPU_TOTAL="$(nproc)"
CPU_PER_WORKER=$(( CPU_TOTAL / GPU_COUNT ))

(( CPU_PER_WORKER < 4 )) && CPU_PER_WORKER=4
(( CPU_PER_WORKER > 32 )) && CPU_PER_WORKER=32

LOG_DIR="$ROOT/Result/dataset_generation"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Dataset generation"
echo "dimensions             = 30..40"
echo "seeds                  = 0..6"
echo "instances              = 77"
echo "parallel workers       = 4"
echo "GPU per worker         = 1"
echo "sieve threads / worker = $CPU_PER_WORKER"
echo "Python                 = $PYTHON_BIN"
echo "============================================================"

nvidia-smi -L
echo

fifo="$(mktemp -u)"
mkfifo "$fifo"
exec 9<>"$fifo"
rm -f "$fifo"

for gpu in 0 1 2 3; do
    printf '%s\n' "$gpu" >&9
done

pids=()

run_one() {
    local dim="$1"
    local seed="$2"

    local gpu
    read -r gpu <&9

    (
        trap 'printf "%s\n" "$gpu" >&9' EXIT

        name="dim${dim}_seed${seed}.txt"
        raw="$ROOT/Dataset/Original/Basis/$name"
        log="$LOG_DIR/dim${dim}_seed${seed}.log"

        {
            echo
            echo "============================================================"
            echo "START dim=$dim seed=$seed GPU=$gpu"
            echo "============================================================"
            date

            CUDA_VISIBLE_DEVICES="$gpu" \
            "$PYTHON_BIN" \
                Dataset/Original/Generator/generate.py \
                --dimension "$dim" \
                --seed "$seed" \
                --name "$name"

            CUDA_VISIBLE_DEVICES="$gpu" \
            LATTICE_SIEVE_THREADS="$CPU_PER_WORKER" \
            "$PYTHON_BIN" \
                Dataset/Prepare/prepare_one.py \
                "$raw" \
                --name "$name"

            rm -f "Dataset/Heavy/Basis/${name}.json"

            for level in Original LLL Light Heavy; do
                file="Dataset/$level/Basis/$name"

                if [[ ! -f "$file" ]]; then
                    echo "ERROR: missing $file"
                    exit 1
                fi
            done

            echo
            echo "============================================================"
            echo "DONE dim=$dim seed=$seed GPU=$gpu"
            echo "============================================================"
            date
        } 2>&1 | tee "$log"
    ) &

    pids+=("$!")
}

for dim in $(seq 30 40); do
    for seed in $(seq 0 6); do
        run_one "$dim" "$seed"
    done
done

failed=0

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

exec 9>&-

echo
echo "============================================================"
echo "Checking dataset"
echo "============================================================"

missing=0

for level in Original LLL Light Heavy; do
    count=0

    for dim in $(seq 30 40); do
        for seed in $(seq 0 6); do
            file="Dataset/$level/Basis/dim${dim}_seed${seed}.txt"

            if [[ -f "$file" ]]; then
                count=$((count + 1))
            else
                echo "MISSING: $file"
                missing=1
            fi
        done
    done

    echo "$level: $count / 77"
done

if (( failed != 0 || missing != 0 )); then
    echo "DATASET GENERATION FINISHED WITH ERRORS"
    exit 1
fi

echo
echo "DATASET GENERATION COMPLETE"
