#!/bin/bash
#
# Run openap-top benchmarks.
#
# Usage:
#   ./benchmark.sh                    Benchmark HEAD (local dev code)
#   ./benchmark.sh 2.0.0              Benchmark PyPI version 2.0.0
#   ./benchmark.sh 1.11.0 2.0.0       Benchmark multiple versions sequentially
#
# Reports are written to tests/benchmarks/<version>.txt.
# Grid cost tests use tests/fixtures/synthetic_4d.casadi (committed fixture).

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ $# -eq 0 ]; then
    uv run tests/benchmark.py
else
    for v in "$@"; do
        echo
        echo "=== Benchmarking openap-top ${v} ==="
        uv run tests/benchmark.py --version "$v"
    done
fi
