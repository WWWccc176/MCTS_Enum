#!/usr/bin/env zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ $# -eq 3 ]] || { echo "usage: ./new_version.zsh <Flash|Net> <from_version> <to_version>"; exit 2; }
MODEL="$1"; FROM="$2"; TO="$3"
case "$MODEL" in
  Flash|flash) MODEL="Flash" ;;
  Net|net) MODEL="Net" ;;
  *) echo "model must be Flash or Net" >&2; exit 2 ;;
esac
SRC="$ROOT/$MODEL/$FROM"; DST="$ROOT/$MODEL/$TO"
[[ -d "$SRC" ]] || { echo "source version not found: $SRC" >&2; exit 2; }
[[ ! -e "$DST" ]] || { echo "destination already exists: $DST" >&2; exit 2; }
cp -a "$SRC" "$DST"
rm -rf "$DST/build"
printf '%s\n' "$TO" > "$DST/VERSION"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/mcts_env/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
"$PYTHON_BIN" - "$DST" "$FROM" "$TO" <<'PY'
from pathlib import Path
import sys
root, old, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for path in root.rglob("*.py"):
    text = path.read_text()
    text = text.replace(f'VERSION = "{old}"', f'VERSION = "{new}"')
    path.write_text(text)
for path in [root / "MCTSBack/include/mcts/types.hpp", root / "MCTSBack/src/pybind_module.cpp"]:
    if path.exists():
        text = path.read_text().replace(old, new)
        path.write_text(text)
PY
echo "CREATED $MODEL/$TO from $MODEL/$FROM"
echo "Edit only algorithm/code under $DST, then build with: ./build_all.zsh $MODEL $TO"
