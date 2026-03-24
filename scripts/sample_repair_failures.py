"""Randomly sample failing repair rows and print why they failed.

Reads experiment ``results.jsonl`` (same format as ``ExperimentRunner`` output).
Failure reasons come from ``mutation_adequacy_status`` and ``evaluation_notes``;
use ``--rerun-exec`` to re-run patched code locally and see per-test outcomes.

Usage (from repo root, after ``pip install -e .`` or with PYTHONPATH=src)::

    python scripts/sample_repair_failures.py --results results/sanitized_mbpp_direct_pilot/results.jsonl
    python scripts/sample_repair_failures.py --results a.jsonl --results b.jsonl -n 8 --seed 42 --rerun-exec
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

# Human-readable labels for mutation_adequacy_status / repair pipeline
_STATUS_DESCRIPTIONS: dict[str, str] = {
    "repair_patch_no_tests_selected": "No tests selected for repair evaluation (config or data issue).",
    "repair_patch_tests_failed": "Patch did not pass all selected tests, or syntax/exec error, timeout, or empty patch.",
    "mutation_adequacy_disabled": "Mutation adequacy disabled (should not co-occur with repair_success=0).",
    "mutation_adequacy_skipped_no_reference": "Mutation check skipped (no reference solution).",
    "mutation_adequacy_not_assessable_no_mutants": "No mutants generated; patch not penalized.",
    "mutation_adequacy_failed_no_mutants_killed": "All tests passed but mutation adequacy failed: mutants were not killed.",
    "mutation_adequacy_passed": "Mutation adequacy passed (should not co-occur with repair_success=0).",
    "mutation_adequacy_error_reference_failed_tests": "Reference did not pass tests; cannot assess mutants.",
    "mutation_adequacy_error_empty_reference": "Empty reference code.",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _patch_passes_value(ev: dict[str, Any]) -> float:
    raw = ev.get("patch_passes_selected_tests")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0
    notes = str(ev.get("evaluation_notes", "") or "")
    if "patch_passes_selected_tests=" in notes:
        frag = notes.split("patch_passes_selected_tests=")[1].split("|")[0].strip()
        try:
            return 1.0 if int(frag) == 1 else 0.0
        except ValueError:
            pass
    if "repair_tests_passed=" in notes:
        frag = notes.split("repair_tests_passed=")[1].split("|")[0].strip()
        if "/" in frag:
            a, b = frag.split("/", 1)
            try:
                ai, bi = int(a.strip()), int(b.strip())
                if bi == 0:
                    return 0.0
                return 1.0 if ai == bi else 0.0
            except ValueError:
                pass
    return 0.0


def _repair_failed(record: dict[str, Any], *, failure_layer: str) -> bool:
    ev = record.get("evaluation_result") or {}
    try:
        strict = float(ev.get("repair_success", 0.0))
    except (TypeError, ValueError):
        strict = 0.0
    patch_ok = _patch_passes_value(ev)
    if failure_layer == "strict":
        return strict < 1.0
    if failure_layer == "patch_tests":
        return patch_ok < 1.0
    return strict < 1.0 or patch_ok < 1.0


def _short_code(s: str | None, max_len: int = 400) -> str:
    if not s:
        return "(none)"
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _print_record(i: int, rec: dict[str, Any], *, rerun_exec: bool) -> None:
    ev = rec.get("evaluation_result") or {}
    diag = rec.get("diagnosis_output") or {}
    bug = rec.get("buggy_sample") or {}
    inj = bug.get("bug_injection_record") or {}

    status = str(ev.get("mutation_adequacy_status", "") or "")
    desc = _STATUS_DESCRIPTIONS.get(status, f"(unknown status: {status})")

    print("=" * 72)
    print(f"Sample #{i + 1}")
    print(f"  record_id:     {rec.get('record_id', '')}")
    print(f"  sample_id:     {rec.get('sample_id', '')}")
    print(f"  prompt:        {diag.get('prompt_template_name', '')}")
    pp = _patch_passes_value(ev)
    try:
        strict_v = float(ev.get("repair_success") or 0.0)
    except (TypeError, ValueError):
        strict_v = 0.0
    print(f"  patch_passes_selected_tests: {pp}  |  repair_success (strict): {ev.get('repair_success')}")
    if pp >= 1.0 and strict_v < 1.0:
        print(
            "  (Note: selected tests all passed but strict repair_success=0 — often mutation adequacy or "
            "another strict gate.)"
        )
    print(f"  failure_reason (status): {desc}")
    print(f"  mutation_adequacy_status: {status or '(empty)'}")
    print(f"  evaluation_notes: {ev.get('evaluation_notes', '')}")
    print(f"  bug_type_accuracy: {ev.get('bug_type_accuracy')} | localization: {ev.get('localization_accuracy')}")
    print(f"  GT bug_type:   {inj.get('bug_type')}")
    print(f"  Pred bug_type: {diag.get('parsed_bug_type')}")
    if diag.get("parsing_error_message"):
        print(f"  parse_error:   {diag.get('parsing_error_message')}")
    print()
    print("  --- injection line (snippet) ---")
    print(_short_code(str(inj.get("buggy_code_snippet", "")), 300))
    print()
    print("  --- patched_code (truncated) ---")
    print(_short_code(diag.get("parsed_repaired_code"), 600))
    print()

    if rerun_exec:
        from thesis_exp.evaluators.diagnosis_evaluator import EvaluationConfig, execute_patched_code_safely
        from thesis_exp.schemas.sample import BuggyProgramSample, ModelDiagnosisOutput

        sample = BuggyProgramSample.from_dict(bug)
        diagnosis = ModelDiagnosisOutput.from_dict(diag)
        cfg = EvaluationConfig()
        exec_res = execute_patched_code_safely(diagnosis.parsed_repaired_code, sample.test_cases, cfg)
        print("  --- local re-run execute_patched_code_safely ---")
        print(
            f"  syntax_valid={exec_res.syntax_valid} timed_out={exec_res.timed_out} "
            f"passed={exec_res.passed_test_count}/{exec_res.total_test_count}"
        )
        if exec_res.execution_error_message:
            print(f"  execution_error: {exec_res.execution_error_message}")
        for o in exec_res.test_outcomes:
            mark = "OK" if o.passed else "FAIL"
            extra = ""
            if not o.passed:
                extra = f"  ({o.exception_type}: {o.exception_message})"
            print(f"    [{mark}] {o.test_case_id}{extra}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample failing repair rows; use --failure-layer to separate strict vs test-layer failures."
    )
    parser.add_argument(
        "--results",
        action="append",
        required=True,
        help="Path(s) to results.jsonl (repeat flag for multiple files).",
    )
    parser.add_argument("-n", type=int, default=5, help="Number of samples to show (default 5).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--failure-layer",
        choices=("strict", "patch_tests", "either"),
        default="strict",
        help=(
            "strict: repair_success<1 (includes mutation gate failures); "
            "patch_tests: not all selected tests passed; "
            "either: failed on at least one layer."
        ),
    )
    parser.add_argument(
        "--rerun-exec",
        action="store_true",
        help="Re-execute patch locally and print per-assert outcomes (slower; uses spawn subprocess).",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for p in args.results:
        path = Path(p)
        if not path.is_file():
            print(f"Warning: file not found, skipping: {path}", file=sys.stderr)
            continue
        all_rows.extend(_load_jsonl(path))

    failed = [r for r in all_rows if _repair_failed(r, failure_layer=args.failure_layer)]
    if not failed:
        print(f"No rows matching failure_layer={args.failure_layer!r} among {len(all_rows)} loaded records.")
        print("Try another results.jsonl or use --failure-layer patch_tests / either.")
        return

    if args.seed is not None:
        random.seed(args.seed)
    k = min(args.n, len(failed))
    picked = random.sample(failed, k) if k < len(failed) else list(failed)

    print(
        f"failure_layer={args.failure_layer!r}: {len(failed)} failing / {len(all_rows)} total; showing {len(picked)} samples.\n"
    )

    for i, rec in enumerate(picked):
        _print_record(i, rec, rerun_exec=args.rerun_exec)


if __name__ == "__main__":
    main()
