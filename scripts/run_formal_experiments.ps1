# Formal experiment run: full re-run, no resume.
# Step 1: Direct + EF-leaky (full)
# Step 2: EF-no-answer config exists
# Step 3: Three-way comparison (Direct | EF-leaky | EF-no-answer)
# record_id has no config_hash; resume does not validate parser/prompt version.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

# Ensure thesis_exp is importable (scripts use src/ via bootstrap, but PYTHONPATH helps)
$env:PYTHONPATH = "src"

Write-Host "=== Deleting old results ==="
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue results/sanitized_mbpp_direct_pilot
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue results/sanitized_mbpp_execution_feedback_pilot
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue results/sanitized_mbpp_execution_feedback_no_leakage_pilot

Write-Host "=== Step 1+2+3: Running Direct (full) ==="
python scripts/run_experiment.py --config configs/sanitized_mbpp_direct_pilot.yaml

Write-Host "=== Running EF-leaky (full) ==="
python scripts/run_experiment.py --config configs/sanitized_mbpp_execution_feedback_pilot.yaml

Write-Host "=== Running EF-no-answer (full) ==="
python scripts/run_experiment.py --config configs/sanitized_mbpp_execution_feedback_no_leakage_pilot.yaml

Write-Host "=== Three-way comparison (summarize) ==="
python scripts/summarize_results.py `
  --results-jsonl results/sanitized_mbpp_direct_pilot/results.jsonl `
  --results-jsonl results/sanitized_mbpp_execution_feedback_pilot/results.jsonl `
  --results-jsonl results/sanitized_mbpp_execution_feedback_no_leakage_pilot/results.jsonl `
  --output-dir results/three_way_comparison

Write-Host "=== Verification (integrity check) ==="
python scripts/verify_comparison.py

Write-Host "=== Done. See results/three_way_comparison/table_3_execution_feedback_comparison.csv ==="
