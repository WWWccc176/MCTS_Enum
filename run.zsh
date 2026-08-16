#!/usr/bin/env zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 3 ]] || {
  echo "usage: ./run.zsh <Flash|Net> <version> --reduction-level <Original|LLL|Light|Heavy> [run options...] <basis...>"
  echo "example: ./run.zsh Flash v0.1 --reduction-level LLL --node-budget 5000000 --refresh-cycles 2 Dataset/LLL/Basis/dim35_seed0.txt"
  exit 2
}
MODEL="$1"; VERSION="$2"; shift 2
case "$MODEL" in
  Flash|flash) MODEL="Flash"; MODULE="Runtime.run" ;;
  Net|net) MODEL="Net"; MODULE="NN.run" ;;
  *) echo "model must be Flash or Net" >&2; exit 2 ;;
esac
TARGET="$ROOT/$MODEL/$VERSION"
[[ -d "$TARGET" ]] || { echo "version not found: $TARGET" >&2; exit 2; }
ls "$TARGET"/build/mcts_enum_backend*.so >/dev/null 2>&1 || {
  echo "backend is not built; run: ./build_all.zsh $MODEL $VERSION" >&2
  exit 2
}
PYTHON_BIN="${PYTHON_BIN:-$ROOT/mcts_env/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
export PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -u -m "$MODULE" "$@"
