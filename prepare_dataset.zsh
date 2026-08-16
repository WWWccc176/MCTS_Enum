#!/usr/bin/env zsh
set -euo pipefail
ROOT="${0:A:h}"
cd "$ROOT"
[[ $# -eq 2 ]] || {
  print -u2 "usage: ./prepare_dataset.zsh <dimension> <seed>"
  exit 2
}
DIM="$1"
SEED="$2"
NAME="dim${DIM}_seed${SEED}.txt"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/mcts_env/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"

# Build generator when needed.
if [[ ! -x Dataset/Original/Generator/generate_random ]]; then
  (cd Dataset/Original/Generator && ./build.sh)
fi

# HeavyBack must be built before preparation.
if ! ls HeavyBack/build/heavy_backend*.so >/dev/null 2>&1; then
  (cd HeavyBack && PYTHON_BIN="$PYTHON_BIN" ./build.sh)
fi

print "==> Original: dim=$DIM seed=$SEED"
"$PYTHON_BIN" Dataset/Original/Generator/generate.py \
  --dimension "$DIM" --seed "$SEED" --name "$NAME"

RAW="Dataset/Original/Basis/$NAME"
print "==> Reduced datasets: LLL / Light / Heavy"
"$PYTHON_BIN" Dataset/Prepare/prepare_one.py "$RAW" --name "$NAME"

# Old preparation code created Heavy .json sidecars. They are not part of the
# new four-file dataset contract.
rm -f "Dataset/Heavy/Basis/${NAME}.json"

print "==> READY"
for level in Original LLL Light Heavy; do
  file="Dataset/$level/Basis/$NAME"
  [[ -f "$file" ]] || { print -u2 "missing: $file"; exit 1; }
  print "    $file"
done
