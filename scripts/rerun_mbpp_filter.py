#!/usr/bin/env python3
"""Re-run sanitized MBPP filtering (same rules as batch experiments).

Uses ``configs/sanitized_mbpp_direct_pilot.yaml`` (via extends) for dataset path,
``filter_output_dir``, and ``evaluation`` so reference checks match ``run_experiment.py``.

By default also writes ``data/sanitized_mbpp_filter_latest`` (common path for
``--accepted-jsonl`` in single-problem scripts).

Examples::

  set PYTHONPATH=src
  python scripts/rerun_mbpp_filter.py
  python scripts/rerun_mbpp_filter.py --config configs/sanitized_mbpp_direct_pilot.yaml
  python scripts/rerun_mbpp_filter.py --no-extra-latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from thesis_exp.datasets.mbpp_filter import (  # noqa: E402
    SanitizedMbppFilterConfig,
    filter_sanitized_mbpp_samples,
)
from thesis_exp.runners.experiment_runner import RunnerExperimentConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/sanitized_mbpp_direct_pilot.yaml",
        help="YAML with dataset.path, dataset.filter_output_dir, evaluation block.",
    )
    parser.add_argument(
        "--no-extra-latest",
        action="store_true",
        help="Do not also write data/sanitized_mbpp_filter_latest.",
    )
    args = parser.parse_args()

    cfg_path = _ROOT / args.config
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1

    cfg = RunnerExperimentConfig.from_file(str(cfg_path))
    ds = Path(cfg.dataset.path)
    if not ds.is_file():
        print(
            f"Dataset not found: {ds}\n"
            "Place sanitized-mbpp.json at that path or edit dataset.path in YAML.",
            file=sys.stderr,
        )
        return 1

    fconf = SanitizedMbppFilterConfig(evaluation_config=cfg.evaluation)
    out_main = cfg.dataset.filter_output_dir or str(_ROOT / "data/filtered/sanitized_mbpp_direct_pilot")

    print(f"[1/2] Filtering -> {out_main}", flush=True)
    filter_sanitized_mbpp_samples(str(ds), out_main, fconf)

    if not args.no_extra_latest:
        extra = _ROOT / "data/sanitized_mbpp_filter_latest"
        print(f"[2/2] Filtering -> {extra}", flush=True)
        filter_sanitized_mbpp_samples(str(ds), str(extra), fconf)

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
