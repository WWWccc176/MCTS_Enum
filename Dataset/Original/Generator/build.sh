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
