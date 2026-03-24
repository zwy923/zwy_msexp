# Formal three-way MBPP run (Direct | EF-leaky | EF-no-leakage) + summarize + verify.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
python scripts/run_three_way_experiments.py @args
