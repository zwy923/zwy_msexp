#!/usr/bin/env bash
# Formal three-way MBPP run + summarize + verify
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/run_three_way_experiments.py "$@"
