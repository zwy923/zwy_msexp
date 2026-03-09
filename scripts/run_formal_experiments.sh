#!/bin/bash
# Formal experiment run: full re-run, no resume.
# Step 1: Direct + EF-leaky (full)
# Step 2: EF-no-answer config exists
# Step 3: Three-way comparison (Direct | EF-leaky | EF-no-answer)

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "=== Deleting old results ==="
rm -rf results/sanitized_mbpp_direct_pilot
rm -rf results/sanitized_mbpp_execution_feedback_pilot
rm -rf results/sanitized_mbpp_execution_feedback_no_leakage_pilot

echo "=== Step 1+2+3: Running Direct (full) ==="
python scripts/run_experiment.py --config configs/sanitized_mbpp_direct_pilot.yaml

echo "=== Running EF-leaky (full) ==="
python scripts/run_experiment.py --config configs/sanitized_mbpp_execution_feedback_pilot.yaml

echo "=== Running EF-no-answer (full) ==="
python scripts/run_experiment.py --config configs/sanitized_mbpp_execution_feedback_no_leakage_pilot.yaml

echo "=== Three-way comparison (summarize) ==="
python scripts/summarize_results.py \
  --results-jsonl results/sanitized_mbpp_direct_pilot/results.jsonl \
  --results-jsonl results/sanitized_mbpp_execution_feedback_pilot/results.jsonl \
  --results-jsonl results/sanitized_mbpp_execution_feedback_no_leakage_pilot/results.jsonl \
  --output-dir results/three_way_comparison

echo "=== Verification (integrity check) ==="
python scripts/verify_comparison.py

echo "=== Done. See results/three_way_comparison/table_3_execution_feedback_comparison.csv ==="
