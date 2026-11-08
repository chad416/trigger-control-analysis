#!/usr/bin/env bash
# Build the C library (shared, for the Python cross-check) and run the C unit tests.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
CFLAGS="-std=c99 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror"
gcc $CFLAGS -c trgctl/trgctl.c -o build/trgctl.o
gcc $CFLAGS -fPIC -shared trgctl/trgctl.c -o build/libtrgctl.so -lm
gcc $CFLAGS tests/test_trgctl.c trgctl/trgctl.c -o build/test_trgctl -lm
./build/test_trgctl
size build/trgctl.o
