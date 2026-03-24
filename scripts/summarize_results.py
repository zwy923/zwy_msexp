"""CLI entry point for summarizing experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from thesis_exp.analysis import generate_analysis_artifacts  # noqa: E402
from thesis_exp.runners import RunnerExperimentConfig  # noqa: E402


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tables and figures from experiment results.")
    parser.add_argument(
        "--config",
        help="Path to an experiment config file. When provided, results paths are derived automatically.",
    )
    parser.add_argument(
        "--results-jsonl",
        action="append",
        help="Path(s) to runner results JSONL. Multiple paths are merged for comparison. Optional if --config is provided.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for generated analysis artifacts. Optional if --config is provided.",
    )
    parser.add_argument(
        "--no-paired-subset",
        action="store_true",
        help="Disable paired subset filtering. By default, table_1/table_3 use only samples present in all three conditions.",
    )
    return parser


def _resolve_paths(
    *,
    config_path: str | None,
    results_jsonl: list[str] | None,
    output_dir: str | None,
) -> tuple[str | list[str], str]:
    if config_path:
        config = RunnerExperimentConfig.from_file(config_path)
        results_dir = Path(config.output.results_dir)
        resolved_results_jsonl = str(results_dir / config.output.jsonl_filename)
        resolved_output_dir = str(Path(output_dir) if output_dir else (results_dir / "analysis"))
        return resolved_results_jsonl, resolved_output_dir

    if not results_jsonl or not output_dir:
        raise ValueError("Either provide --config or provide both --results-jsonl and --output-dir.")

    return results_jsonl, output_dir


def main() -> int:
    args = _build_argument_parser().parse_args()
    results_jsonl_path, output_dir = _resolve_paths(
        config_path=args.config,
        results_jsonl=args.results_jsonl,
        output_dir=args.output_dir,
    )
    generate_analysis_artifacts(
        results_jsonl_path,
        output_dir,
        use_paired_subset=not args.no_paired_subset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
