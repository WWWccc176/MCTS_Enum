#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/DRL/MCTS_Enum"
PYTHON_BIN="$ROOT/mcts_env/bin/python"
NTL_PREFIX="$ROOT/mcts_env/ntl-9.3.0"
NTL_SRC_ROOT="$ROOT/mcts_env/src"
NTL_ARCHIVE="$NTL_SRC_ROOT/ntl-9.3.0.tar.gz"
NTL_SRC="$NTL_SRC_ROOT/ntl-9.3.0"
GEN_DIR="$ROOT/Dataset/Original/Generator"

cd "$ROOT"

test -x "$PYTHON_BIN"
command -v curl >/dev/null
command -v g++ >/dev/null
command -v make >/dev/null

mkdir -p "$NTL_SRC_ROOT"

if [ ! -f "$NTL_PREFIX/include/NTL/version.h" ]; then
    rm -rf "$NTL_SRC"
    curl -fL --retry 3 --retry-delay 2 \
        https://libntl.org/ntl-9.3.0.tar.gz \
        -o "$NTL_ARCHIVE"
    tar -xzf "$NTL_ARCHIVE" -C "$NTL_SRC_ROOT"

    cd "$NTL_SRC/src"
    ./configure \
        DEF_PREFIX="$NTL_PREFIX" \
        NTL_GMP_LIP=on \
        CXX=g++ \
        CXXFLAGS="-O3 -DNDEBUG -std=gnu++11"
    make
    make check
    make install
fi

cd "$ROOT"

grep -Eq 'NTL_VERSION.*9.*3|NTL_MAJOR_VERSION.*9' "$NTL_PREFIX/include/NTL/version.h"

NTL_STATIC="$(find "$NTL_PREFIX" -type f -name 'libntl.a' -print -quit)"
test -n "$NTL_STATIC"

cat > "$GEN_DIR/build.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
NTL_PREFIX="${NTL93_PREFIX:-$ROOT/mcts_env/ntl-9.3.0}"
CXX="${CXX:-g++}"

NTL_STATIC="$(find "$NTL_PREFIX" -type f -name 'libntl.a' -print -quit)"
test -f "$NTL_STATIC"
test -d "$NTL_PREFIX/include"

cd "$HERE"

"$CXX" \
    -O3 -DNDEBUG -std=gnu++11 -Wall -Wextra \
    -I"$NTL_PREFIX/include" \
    -o generate_random \
    generate_random.cpp \
    "$NTL_STATIC" \
    -lgmp -lm -pthread
EOF

chmod +x "$GEN_DIR/build.sh"

cat > "$GEN_DIR/readme.txt" <<'EOF'
SVP Challenge generator.

The official SVP Challenge instances require the pre-NTL-9.4 pseudorandom
generator. This project therefore builds generate_random against a private
NTL 9.3.0 installation at:

    mcts_env/ntl-9.3.0

Do not link this generator against the system NTL if the system version is
9.4 or newer. NTL 9.4 changed the PRG, so the same dimension/seed would
produce a different lattice.

Build:
    ./build.sh

Run:
    ./generate_random --dim 40 --seed 5778
EOF

"$GEN_DIR/build.sh"

if ldd "$GEN_DIR/generate_random" 2>/dev/null | grep -q 'libntl'; then
    echo "generate_random unexpectedly depends on a shared system libntl" >&2
    exit 1
fi

"$GEN_DIR/generate_random" --dim 40 --seed 5778 > /tmp/mcts_enum_ntl93_dim40_seed5778.txt
test -s /tmp/mcts_enum_ntl93_dim40_seed5778.txt

"$PYTHON_BIN" - <<'PY'
from pathlib import Path

for name in ("README.md", "FILE_GUIDE.md"):
    p = Path(name)
    if not p.is_file():
        continue
    s = p.read_text(encoding="utf-8")
    marker = "## SVP Challenge generator compatibility"
    if marker not in s:
        s += f"""

{marker}

`Dataset/Original/Generator/generate_random` is built against the private
`mcts_env/ntl-9.3.0` installation. This is required for compatibility with
the official SVP Challenge: NTL 9.4 changed the pseudorandom generator, so
using NTL 9.4+ with the same dimension and seed produces a different lattice.

Dataset levels follow the implementation:
- `Original`: official-compatible raw SVP Challenge basis.
- `LLL`: `LLL(delta=0.999)`.
- `Light`: LLL followed by BKZ20.
- `Heavy`: Light followed by BKZ30 and the whole-basis BGJ sieve.

BKZ20 is a dataset preprocessing stage for `Light`/`Heavy`; it is not an
implicit Flash search-cycle operation.
"""
        p.write_text(s, encoding="utf-8")
PY

mapfile -t NAMES < <(
    find Dataset/Original/Basis \
        -maxdepth 1 -type f \
        -name 'dim*_seed*.txt' \
        -printf '%f\n' | sort -V
)

if [ "${#NAMES[@]}" -eq 0 ]; then
    echo "No existing Original dataset filenames found" >&2
    exit 1
fi

rm -rf HeavyBack/build

cd HeavyBack
PYTHON_BIN="$PYTHON_BIN" BUILD_DIR=build ./build.sh
cd "$ROOT"

BACKUP="$(mktemp -d "$ROOT/Dataset/.ntl_prg_backup.XXXXXX")"
COMMITTED=0

restore_old_dataset() {
    if [ "$COMMITTED" -eq 0 ]; then
        rm -rf \
            "$ROOT/Dataset/Original/Basis" \
            "$ROOT/Dataset/LLL/Basis" \
            "$ROOT/Dataset/Light/Basis" \
            "$ROOT/Dataset/Heavy/Basis"

        for level in Original LLL Light Heavy; do
            if [ -d "$BACKUP/$level" ]; then
                mkdir -p "$ROOT/Dataset/$level"
                mv "$BACKUP/$level" "$ROOT/Dataset/$level/Basis"
            fi
        done
    fi
}

trap restore_old_dataset ERR INT TERM

for level in Original LLL Light Heavy; do
    mkdir -p "$BACKUP"
    if [ -d "$ROOT/Dataset/$level/Basis" ]; then
        mv "$ROOT/Dataset/$level/Basis" "$BACKUP/$level"
    fi
    mkdir -p "$ROOT/Dataset/$level/Basis"
done

for name in "${NAMES[@]}"; do
    if [[ ! "$name" =~ ^dim([0-9]+)_seed(-?[0-9]+)\.txt$ ]]; then
        echo "Unexpected dataset filename: $name" >&2
        exit 1
    fi

    dim="${BASH_REMATCH[1]}"
    seed="${BASH_REMATCH[2]}"

    "$PYTHON_BIN" Dataset/Original/Generator/generate.py \
        --dimension "$dim" \
        --seed "$seed" \
        --name "$name"

    "$PYTHON_BIN" Dataset/Prepare/prepare_one.py \
        "Dataset/Original/Basis/$name" \
        --name "$name"

    rm -f "Dataset/Heavy/Basis/$name.json"
done

expected="${#NAMES[@]}"

for level in Original LLL Light Heavy; do
    actual="$(find "Dataset/$level/Basis" -maxdepth 1 -type f -name 'dim*_seed*.txt' | wc -l)"
    if [ "$actual" -ne "$expected" ]; then
        echo "$level count mismatch: expected=$expected actual=$actual" >&2
        exit 1
    fi
done

for name in "${NAMES[@]}"; do
    for level in Original LLL Light Heavy; do
        test -s "Dataset/$level/Basis/$name"
    done
done

COMMITTED=1
trap - ERR INT TERM
rm -rf "$BACKUP"

rm -rf Flash/v0.1/build Flash/v0.2/build

cd Flash/v0.1/MCTSBack
PYTHON_BIN="$PYTHON_BIN" BUILD_DIR="../build" ./build.sh

cd "$ROOT/Flash/v0.2/MCTSBack"
PYTHON_BIN="$PYTHON_BIN" BUILD_DIR="../build" ./build.sh

cd "$ROOT"

printf '%s\n' \
    "NTL93=$NTL_PREFIX" \
    "generator=$GEN_DIR/generate_random" \
    "datasets=$expected files per level" \
    "Original=$(find Dataset/Original/Basis -maxdepth 1 -type f -name 'dim*_seed*.txt' | wc -l)" \
    "LLL=$(find Dataset/LLL/Basis -maxdepth 1 -type f -name 'dim*_seed*.txt' | wc -l)" \
    "Light=$(find Dataset/Light/Basis -maxdepth 1 -type f -name 'dim*_seed*.txt' | wc -l)" \
    "Heavy=$(find Dataset/Heavy/Basis -maxdepth 1 -type f -name 'dim*_seed*.txt' | wc -l)"

echo "READY"
