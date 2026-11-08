#!/usr/bin/env bash
# Build, run the C unit tests, the Python cross-check tests and the evaluation (writes results/).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
./scripts/build.sh
"$PY" -m pytest -q tests
"$PY" tools/evaluate.py
