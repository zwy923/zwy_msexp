"""CLI entry point for running experiments."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

# Suppress SyntaxWarning from MBPP reference/test code with invalid escape sequences (e.g. "\w", "\.")
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")


def _bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from thesis_exp.common.env import load_env_file  # noqa: E402
from thesis_exp.runners import ExperimentRunner  # noqa: E402


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full bug-diagnosis experiment pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment config file in JSON or YAML format.",
    )
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root)
    runner = ExperimentRunner.from_file(args.config)
    results = runner.run()
    logging.getLogger("thesis_exp.scripts.run_experiment").info(
        "Experiment completed with %s new evaluation result(s).",
        len(results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
