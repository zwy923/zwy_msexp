#!/usr/bin/env python3
"""Export sample_ids for fair three-way experiments.

Two modes:

1) **Intersection of three finished runs** (default): ``sample_id`` must appear in all three
   ``results.jsonl`` files. Loads are **deduped by sample_id** (last line wins).

2) **From accepted JSONL** (``--from-accepted``): Enumerates the same ``sample_id`` values the batch
   runner would emit for each problem × injector, using ``validate_buggy_sample`` with the
   given YAML ``generation`` / ``evaluation`` blocks. Use this after re-filtering, **before**
   re-running the three API experiments.

Usage::

  set PYTHONPATH=src
  python scripts/export_paired_sample_ids.py --output data/paired_sample_ids.txt

  python scripts/export_paired_sample_ids.py \\
    --from-accepted data/sanitized_mbpp_filter_latest/accepted_samples.jsonl \\
    --config configs/sanitized_mbpp_direct_pilot.yaml \\
    --output data/paired_sample_ids.txt

Add to experiment YAML when restricting the run::

  dataset:
    sample_ids_path: data/paired_sample_ids.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thesis_exp.datasets.loader import (  # noqa: E402
    dataset_record_to_problem,
    validate_dataset_problem_record,
)
from thesis_exp.quality import SampleQualityConfig, validate_buggy_sample  # noqa: E402
from thesis_exp.runners.experiment_runner import (  # noqa: E402
    RunnerExperimentConfig,
    build_buggy_sample,
)
from thesis_exp.schemas.sample import ProgrammingProblem  # noqa: E402


def _problem_from_accepted_line(line: dict[str, Any]) -> ProgrammingProblem:
    """Build ``ProgrammingProblem`` from one ``accepted_samples.jsonl`` object."""

    tests = line.get("tests")
    if not tests and isinstance(line.get("raw_record"), dict):
        tr = line["raw_record"].get("test_list") or []
        ti = line["raw_record"].get("test_imports") or []
        prefix = "\n".join(x for x in ti if isinstance(x, str) and x.strip())
        tests = [f"{prefix}\n{t}" if prefix else t for t in tr if isinstance(t, str)]
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"accepted line {line.get('problem_id')!r}: missing non-empty tests")

    payload: dict[str, Any] = {
        "problem_id": line["problem_id"],
        "prompt": (line.get("prompt") or "").strip() or "(empty)",
        "entry_point": (line.get("entry_point") or "").strip(),
        "reference_code": (line.get("reference_code") or "").strip(),
        "tests": tests,
        "programming_language": line.get("programming_language") or "python",
        "starter_code": line.get("starter_code") or "",
        "problem_title": line.get("problem_title") or line["problem_id"],
        "difficulty": line.get("difficulty") or "",
        "topic": line.get("topic") or "",
        "source_dataset_name": line.get("source_dataset_name") or "",
        "source_split_name": line.get("source_split_name") or "",
    }
    if not payload["entry_point"] or not payload["reference_code"]:
        raise ValueError(f"accepted line {line.get('problem_id')!r}: missing entry_point or reference_code")
    return dataset_record_to_problem(validate_dataset_problem_record(payload))


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _keys_from_results_jsonl(path: Path) -> set[str]:
    """Deduplicate by sample_id (last occurrence wins)."""

    by_sid: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl_records(path):
        sid = str(record.get("sample_id", "")).strip()
        if sid:
            by_sid[sid] = record
    return set(by_sid.keys())


def _enumerate_from_accepted(
    accepted_path: Path,
    cfg: RunnerExperimentConfig,
) -> list[str]:
    lines = _load_jsonl_records(accepted_path)
    max_p = cfg.dataset.max_problems
    if max_p is not None:
        lines = lines[: int(max_p)]

    quality_config = SampleQualityConfig(
        max_changed_lines=cfg.generation.max_changed_lines,
        require_reference_pass_all_tests=cfg.generation.require_reference_pass_all_tests,
        require_buggy_fail_some_tests=cfg.generation.require_buggy_fail_some_tests,
        require_buggy_syntax_valid=cfg.generation.require_buggy_syntax_valid,
        require_buggy_executable=cfg.generation.require_buggy_executable,
    )

    sample_ids: list[str] = []
    for line in lines:
        if not line.get("accepted", True):
            continue
        try:
            problem = _problem_from_accepted_line(line)
        except Exception as exc:  # noqa: BLE001
            print(f"Skip line (parse error): {exc}", file=sys.stderr)
            continue

        for injector_type in cfg.generation.injector_types:
            sample = build_buggy_sample(problem, injector_type)
            if sample is None:
                continue
            report = validate_buggy_sample(sample, quality_config, cfg.evaluation)
            if not report.accepted:
                continue
            sample_ids.append(sample.sample_id)

    return sorted(set(sample_ids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/paired_sample_ids.txt",
        help="Output path for sample_ids (one per line).",
    )
    parser.add_argument(
        "--config",
        default="configs/sanitized_mbpp_direct_pilot.yaml",
        help="YAML for --from-accepted (generation.injector_types, max_problems, evaluation).",
    )
    parser.add_argument(
        "--from-accepted",
        default="",
        help="If set, enumerate sample_ids from this accepted_samples.jsonl (no results.jsonl needed).",
    )
    parser.add_argument(
        "--direct",
        default="results/sanitized_mbpp_direct_pilot/results.jsonl",
        help="[results mode] Path to direct diagnosis results.",
    )
    parser.add_argument(
        "--ef-leaky",
        default="results/sanitized_mbpp_execution_feedback_pilot/results.jsonl",
        help="[results mode] Path to EF-leaky results.",
    )
    parser.add_argument(
        "--ef-no-answer",
        default="results/sanitized_mbpp_execution_feedback_no_leakage_pilot/results.jsonl",
        help="[results mode] Path to EF-no-answer results.",
    )
    args = parser.parse_args()

    project_root = _ROOT
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.from_accepted.strip():
        acc = project_root / args.from_accepted.strip()
        if not acc.is_file():
            print(f"Accepted JSONL not found: {acc}", file=sys.stderr)
            return 1
        cfg_path = project_root / args.config
        if not cfg_path.is_file():
            print(f"Config not found: {cfg_path}", file=sys.stderr)
            return 1
        cfg = RunnerExperimentConfig.from_file(str(cfg_path))
        paired = _enumerate_from_accepted(acc, cfg)
        if not paired:
            print("No sample_ids enumerated (check accepted file and quality gates).", file=sys.stderr)
            return 1
        output_path.write_text("\n".join(paired) + "\n", encoding="utf-8")
        print(f"Exported {len(paired)} sample_ids from {acc} (enumeration + quality, max_problems={cfg.dataset.max_problems})")
        print(f"  -> {output_path}")
        return 0

    direct_path = project_root / args.direct
    ef_leaky_path = project_root / args.ef_leaky
    ef_no_answer_path = project_root / args.ef_no_answer

    keys_direct = _keys_from_results_jsonl(direct_path)
    keys_ef_leaky = _keys_from_results_jsonl(ef_leaky_path)
    keys_ef_no_answer = _keys_from_results_jsonl(ef_no_answer_path)

    paired = sorted(keys_direct & keys_ef_leaky & keys_ef_no_answer)
    if not paired:
        print("No paired samples found (intersection of three deduped results.jsonl).", file=sys.stderr)
        print(
            "  Hint: use --from-accepted path/to/accepted_samples.jsonl after re-filtering.",
            file=sys.stderr,
        )
        return 1

    output_path.write_text("\n".join(paired) + "\n", encoding="utf-8")

    print(f"Exported {len(paired)} paired sample_ids to {output_path}")
    print(f"  direct: {len(keys_direct)}, ef_leaky: {len(keys_ef_leaky)}, ef_no_answer: {len(keys_ef_no_answer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
