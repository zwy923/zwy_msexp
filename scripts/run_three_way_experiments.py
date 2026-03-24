#!/usr/bin/env python3
"""Run Direct + EF-leaky + EF-no-leakage, then summarize and verify (cross-platform)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

THREE_WAY_CONFIGS: tuple[str, ...] = (
    "configs/sanitized_mbpp_direct_pilot.yaml",
    "configs/sanitized_mbpp_execution_feedback_pilot.yaml",
    "configs/sanitized_mbpp_execution_feedback_no_leakage_pilot.yaml",
)


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SRC)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_ROOT), env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing results dirs for the three conditions.",
    )
    parser.add_argument("--skip-summarize", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from thesis_exp.runners.experiment_runner import RunnerExperimentConfig  # noqa: E402

    if not args.no_clean:
        for rel in THREE_WAY_CONFIGS:
            cfg_path = _ROOT / rel
            out = Path(RunnerExperimentConfig.from_file(str(cfg_path)).output.results_dir)
            if out.exists():
                shutil.rmtree(out)
                print(f"Removed {out}", flush=True)

    for rel in THREE_WAY_CONFIGS:
        _run([sys.executable, "scripts/run_experiment.py", "--config", rel])

    if not args.skip_summarize:
        _run(
            [
                sys.executable,
                "scripts/summarize_results.py",
                "--results-jsonl",
                "results/sanitized_mbpp_direct_pilot/results.jsonl",
                "--results-jsonl",
                "results/sanitized_mbpp_execution_feedback_pilot/results.jsonl",
                "--results-jsonl",
                "results/sanitized_mbpp_execution_feedback_no_leakage_pilot/results.jsonl",
                "--output-dir",
                "results/three_way_comparison",
            ]
        )

    if not args.skip_verify:
        _run([sys.executable, "scripts/verify_comparison.py"])

    print("Done. See results/three_way_comparison/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
