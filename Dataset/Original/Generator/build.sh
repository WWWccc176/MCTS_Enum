#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O3 -DNDEBUG -Wall -Wextra}"
$CXX $CXXFLAGS -o generate_random generate_random.cpp -lntl -lgmp -lm
