#!/usr/bin/env zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ $# -ge 2 ]] || { echo "usage: ./build_all.zsh <Flash|Net> <version> [--clean]"; exit 2; }
MODEL="$1"; VERSION="$2"; CLEAN="${3:-}"
case "$MODEL" in
  Flash|flash) MODEL="Flash" ;;
  Net|net) MODEL="Net" ;;
  *) echo "model must be Flash or Net" >&2; exit 2 ;;
esac
TARGET="$ROOT/$MODEL/$VERSION"
[[ -d "$TARGET/MCTSBack" ]] || { echo "version not found: $TARGET" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-$ROOT/mcts_env/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"

if [[ "$CLEAN" == "--clean" ]]; then
  rm -rf "$TARGET/build"
fi

if [[ -x "$ROOT/Dataset/Original/Generator/build.sh" ]]; then
  echo "==> Dataset/Original/Generator"
  (cd "$ROOT/Dataset/Original/Generator" && ./build.sh)
fi

if [[ -x "$ROOT/HeavyBack/build.sh" ]]; then
  echo "==> HeavyBack (shared preparation backend)"
  (cd "$ROOT/HeavyBack" && PYTHON_BIN="$PYTHON_BIN" ./build.sh)
fi

echo "==> $MODEL/$VERSION/MCTSBack -> $TARGET/build"
(cd "$TARGET/MCTSBack" && \
  BUILD_DIR="$TARGET/build" \
  PYTHON_BIN="$PYTHON_BIN" \
  ./build.sh)

echo "BUILD_OK model=$MODEL version=$VERSION"
