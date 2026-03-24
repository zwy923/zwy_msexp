#!/usr/bin/env python3
"""Print detailed evaluation info for one sample_id from results JSONL.

Use this to see *why* a row failed (patch tests vs mutation gate, timeouts, etc.).

Examples::

  set PYTHONPATH=src
  python scripts/inspect_sample_result.py \\
    --results results/sanitized_mbpp_direct_pilot/results.jsonl \\
    --sample-id mbpp_20::condition_inversion::acaa557a2f9de591

  # Re-run patched code with the same EvaluationConfig as experiments (recommended):
  python scripts/inspect_sample_result.py \\
    --results results/sanitized_mbpp_direct_pilot/results.jsonl \\
    --sample-id mbpp_57::premature_return::e028f292aa0250de \\
    --config configs/sanitized_mbpp_direct_pilot.yaml \\
    --rerun-exec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_src_path()

_STATUS_HELP: dict[str, str] = {
    "repair_patch_no_tests_selected": "No tests were selected for repair scoring (check EvaluationConfig / test_case types).",
    "repair_patch_tests_failed": "Patch did not pass all selected tests, or syntax error, exec error, timeout, or missing patched_code.",
    "mutation_adequacy_disabled": "Mutation adequacy is off; strict repair_success should equal patch_passes_selected_tests.",
    "mutation_adequacy_skipped_no_reference": "No reference code; mutation gate skipped (strict may still follow patch layer).",
    "mutation_adequacy_not_assessable_no_mutants": "Mutator produced zero mutants; patch not penalized for adequacy.",
    "mutation_adequacy_failed_no_mutants_killed": "All tests passed on the patch, but the suite did not kill enough mutants of the reference.",
    "mutation_adequacy_passed": "Tests passed and mutation adequacy passed.",
    "mutation_adequacy_error_reference_failed_tests": "Reference failed tests; adequacy check aborted.",
    "mutation_adequacy_error_empty_reference": "Empty reference; adequacy check aborted.",
}


def _load_first_match(paths: list[Path], sample_id: str) -> dict[str, Any] | None:
    for path in paths:
        if not path.is_file():
            print(f"Warning: missing file, skipping: {path}", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("sample_id", "")).strip() == sample_id:
                return rec
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one results.jsonl row by sample_id.")
    parser.add_argument("--sample-id", required=True, help="Exact sample_id string.")
    parser.add_argument(
        "--results",
        action="append",
        required=True,
        help="Path to results.jsonl (first file containing the sample wins).",
    )
    parser.add_argument(
        "--config",
        help="Experiment YAML/JSON to load EvaluationConfig for --rerun-exec (optional).",
    )
    parser.add_argument(
        "--rerun-exec",
        action="store_true",
        help="Re-execute patched_code in the sandbox and print per-test outcomes.",
    )
    parser.add_argument(
        "--full-code",
        action="store_true",
        help="Print full buggy_code and patched_code (can be long).",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.results]
    rec = _load_first_match(paths, args.sample_id.strip())
    if rec is None:
        print(f"No record found for sample_id={args.sample_id!r} in given --results files.", file=sys.stderr)
        return 1

    ev = rec.get("evaluation_result") or {}
    diag = rec.get("diagnosis_output") or {}
    bug = rec.get("buggy_sample") or {}
    inj = bug.get("bug_injection_record") or {}

    print("=== Record ===")
    print("record_id:", rec.get("record_id"))
    print("experiment_id:", rec.get("experiment_id"))
    print("problem_id:", rec.get("problem_id"))
    print("prompt_template:", diag.get("prompt_template_name"))
    print()

    print("=== Evaluation (numeric) ===")
    for key in (
        "bug_type_accuracy",
        "bug_type_accuracy_coarse",
        "localization_accuracy",
        "localization_accuracy_exact",
        "patch_passes_selected_tests",
        "repair_success",
        "diagnosis_hallucination_rate",
        "narrative_hallucination_rate",
    ):
        if key in ev:
            print(f"  {key}: {ev.get(key)}")
    print()

    print("=== Ground truth vs prediction ===")
    print("  ground_truth_bug_type:", ev.get("ground_truth_bug_type"), "| predicted:", ev.get("predicted_bug_type"))
    print("  injection lines:", ev.get("ground_truth_injection_line_start"), "-", ev.get("ground_truth_injection_line_end"))
    print("  predicted lines:", ev.get("predicted_bug_line_start"), "-", ev.get("predicted_bug_line_end"))
    print("  injector:", inj.get("injection_operator_name"), "| injected bug_type (record):", inj.get("bug_type"))
    print()

    status = str(ev.get("mutation_adequacy_status") or "")
    print("=== Why pass/fail (repair pipeline) ===")
    print("mutation_adequacy_status:", status or "(empty)")
    print("explained:", _STATUS_HELP.get(status, "(no built-in explanation for this token)"))
    print()
    print("evaluation_notes (full):")
    print(ev.get("evaluation_notes") or "(empty)")
    print()

    if diag.get("parsing_error_message"):
        print("=== Parsing ===")
        print(diag.get("parsing_error_message"))
        print()

    print("=== Snippets ===")
    print("bug_description:", inj.get("bug_description"))
    print("buggy_code_snippet:\n", inj.get("buggy_code_snippet") or "(none)")
    if args.full_code:
        print("\n--- full buggy_code ---\n", bug.get("buggy_code") or "(none)")
        print("\n--- full patched_code ---\n", diag.get("parsed_repaired_code") or "(none)")
    else:
        pc = diag.get("parsed_repaired_code") or ""
        prev = (pc[:1200] + "\n... [truncated]") if len(pc) > 1200 else pc
        print("patched_code (first 1200 chars):\n", prev or "(none)")
    print()

    if args.rerun_exec:
        from thesis_exp.evaluators.diagnosis_evaluator import EvaluationConfig, execute_patched_code_safely
        from thesis_exp.runners.experiment_runner import RunnerExperimentConfig
        from thesis_exp.schemas.sample import BuggyProgramSample, ModelDiagnosisOutput

        if args.config:
            cfg = RunnerExperimentConfig.from_file(args.config).evaluation
        else:
            cfg = EvaluationConfig()
            print(
                "Note: using default EvaluationConfig (use --config for YAML parity).\n",
                file=sys.stderr,
            )

        sample = BuggyProgramSample.from_dict(bug)
        diagnosis = ModelDiagnosisOutput.from_dict(diag)
        exec_res = execute_patched_code_safely(diagnosis.parsed_repaired_code, sample.test_cases, cfg)
        print("=== Rerun execute_patched_code_safely ===")
        print(
            f"syntax_valid={exec_res.syntax_valid} timed_out={exec_res.timed_out} "
            f"passed={exec_res.passed_test_count}/{exec_res.total_test_count}"
        )
        if exec_res.execution_error_message:
            print("execution_error_message:", exec_res.execution_error_message)
        print("Per test:")
        for o in exec_res.test_outcomes:
            mark = "PASS" if o.passed else "FAIL"
            line = f"  [{mark}] {o.test_case_id} hidden={o.is_hidden}"
            if not o.passed:
                line += f" | {o.exception_type}: {o.exception_message}"
            print(line)
        if not exec_res.test_outcomes and exec_res.total_test_count:
            print("  (no per-test rows; failure before test loop — see execution_error_message)")
        print()
        print("Selected tests only (hidden/public per config). Entry point:", bug.get("entry_point"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
