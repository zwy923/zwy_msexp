"""Preprocess and filter sanitized MBPP samples for controlled experiments."""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Suppress SyntaxWarning from MBPP reference/test code with invalid escape sequences
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

from thesis_exp.datasets import (
    DatasetProtocolError,
    DatasetProblemRecord,
    dataset_record_to_problem,
    load_sanitized_mbpp_raw_payloads,
    validate_sanitized_mbpp_record,
)
from thesis_exp.evaluators.diagnosis_evaluator import EvaluationConfig, select_tests_for_evaluation
from thesis_exp.evaluators.reference_validation import (
    reference_execution_filter_fields,
    validate_reference_solution,
)
from thesis_exp.execution.safe_exec import ALLOWED_IMPORT_ROOT_MODULES
from thesis_exp.injectors.python import (
    find_top_level_function,
    function_has_accumulator_init_site,
    function_has_condition_inversion_site,
    function_has_off_by_one_site,
    function_has_premature_return_site,
    function_has_wrong_comparison_operator_site,
    function_has_wrong_loop_bound_site,
)
from thesis_exp.schemas.sample import ProgrammingProblem, TestCase

_DISALLOWED_IMPORT_MODULES = {
    "os",
    "pathlib",
    "shutil",
    "subprocess",
    "socket",
    "sys",
    "tempfile",
}


@dataclass(slots=True)
class SanitizedMbppFilterConfig:
    """Explicit filtering rules for sanitized MBPP problems."""

    max_reference_line_count: int = 30
    max_top_level_function_count: int = 3
    execution_timeout_seconds: float = 2.0
    """Used only when ``evaluation_config`` is None (CLI default). Mirrors ``EvaluationConfig.repair_timeout_seconds``."""

    evaluation_config: EvaluationConfig | None = None
    """When set, reference test execution uses this (timeout, hidden/public tests) — same as batch experiments."""

    require_supported_injector_pattern: bool = True
    allow_classes: bool = False
    allow_file_io: bool = False


def evaluation_config_for_sanitized_filter(config: SanitizedMbppFilterConfig) -> EvaluationConfig:
    """Resolve the ``EvaluationConfig`` used for reference validation during filtering."""

    if config.evaluation_config is not None:
        return config.evaluation_config
    return EvaluationConfig(repair_timeout_seconds=config.execution_timeout_seconds)


@dataclass(slots=True)
class SanitizedMbppFilterMetadata:
    """Computed metadata used for filtering decisions."""

    line_count: int
    top_level_function_count: int
    top_level_function_names: list[str]
    has_top_level_function: bool
    entry_point_present: bool
    class_count: int
    has_classes: bool
    imported_modules: list[str]
    has_disallowed_imports: bool
    has_unknown_imports: bool
    file_io_keywords_present: bool
    has_loops: bool
    has_comparisons: bool
    has_assignments: bool
    supported_injector_patterns: list[str]
    reference_code_executable: bool
    tests_executable: bool
    reference_passes_all_tests: bool
    execution_error_message: str = ""


@dataclass(slots=True)
class SanitizedMbppFilterDecision:
    """Decision record for one sanitized MBPP problem."""

    record_index: int
    task_id: str
    problem_id: str
    accepted: bool
    rejection_reasons: list[str]
    filter_metadata: SanitizedMbppFilterMetadata
    normalized_record: DatasetProblemRecord | None = None
    raw_record: dict[str, Any] | None = None


@dataclass(slots=True)
class SanitizedMbppFilterArtifacts:
    """Paths for exported filtering artifacts."""

    accepted_samples_jsonl: str
    rejected_samples_jsonl: str
    filtering_summary_json: str


def _collect_imported_modules(module: ast.Module) -> list[str]:
    imported_modules: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module.split(".")[0])
    return sorted(set(imported_modules))


def _has_file_io_keywords(module: ast.Module) -> bool:
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            return True
        if isinstance(node, ast.Attribute) and node.attr in {"read", "write", "readlines", "writelines"}:
            return True
    return False


def _supported_injector_patterns(module: ast.Module, entry_point: str) -> list[str]:
    """Injector names that can actually apply to the entry-point function (aligned with injectors/python.py)."""

    target_fn = find_top_level_function(module, entry_point)
    if target_fn is None:
        return []

    supported: set[str] = set()
    if function_has_off_by_one_site(target_fn) or function_has_wrong_loop_bound_site(target_fn):
        supported.add("loop_boundary_error")
    if function_has_accumulator_init_site(target_fn):
        supported.add("accumulator_init_error")
    if function_has_condition_inversion_site(target_fn) or function_has_wrong_comparison_operator_site(
        target_fn
    ):
        supported.add("conditional_logic_error")
    if function_has_premature_return_site(target_fn):
        supported.add("premature_return")

    return sorted(supported)


def analyze_sanitized_mbpp_problem(
    problem: ProgrammingProblem,
    config: SanitizedMbppFilterConfig,
) -> SanitizedMbppFilterMetadata:
    """Compute filtering metadata for one normalized MBPP problem."""

    try:
        module = ast.parse(problem.reference_solution.reference_code)
    except SyntaxError as exc:
        return SanitizedMbppFilterMetadata(
            line_count=len(problem.reference_solution.reference_code.splitlines()),
            top_level_function_count=0,
            top_level_function_names=[],
            has_top_level_function=False,
            entry_point_present=False,
            class_count=0,
            has_classes=False,
            imported_modules=[],
            has_disallowed_imports=False,
            has_unknown_imports=False,
            file_io_keywords_present=False,
            has_loops=False,
            has_comparisons=False,
            has_assignments=False,
            supported_injector_patterns=[],
            reference_code_executable=False,
            tests_executable=False,
            reference_passes_all_tests=False,
            execution_error_message=f"SyntaxError: {exc.msg}",
        )

    top_level_function_names = [
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    imported_modules = _collect_imported_modules(module)
    has_disallowed_imports = any(module_name in _DISALLOWED_IMPORT_MODULES for module_name in imported_modules)
    has_unknown_imports = any(
        module_name not in ALLOWED_IMPORT_ROOT_MODULES and module_name not in _DISALLOWED_IMPORT_MODULES
        for module_name in imported_modules
    )
    eval_cfg = evaluation_config_for_sanitized_filter(config)
    if not select_tests_for_evaluation(problem.test_cases, eval_cfg):
        reference_code_executable = False
        tests_executable = False
        reference_passes_all_tests = False
        execution_error_message = "No executable tests were found."
    else:
        ref_result = validate_reference_solution(
            problem.reference_solution.reference_code,
            problem.test_cases,
            eval_cfg,
        )
        (
            reference_code_executable,
            tests_executable,
            reference_passes_all_tests,
            execution_error_message,
        ) = reference_execution_filter_fields(ref_result)

    return SanitizedMbppFilterMetadata(
        line_count=len(problem.reference_solution.reference_code.splitlines()),
        top_level_function_count=len(top_level_function_names),
        top_level_function_names=top_level_function_names,
        has_top_level_function=bool(top_level_function_names),
        entry_point_present=problem.entry_point in top_level_function_names,
        class_count=sum(1 for node in ast.walk(module) if isinstance(node, ast.ClassDef)),
        has_classes=any(isinstance(node, ast.ClassDef) for node in ast.walk(module)),
        imported_modules=imported_modules,
        has_disallowed_imports=has_disallowed_imports,
        has_unknown_imports=has_unknown_imports,
        file_io_keywords_present=_has_file_io_keywords(module),
        has_loops=any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(module)),
        has_comparisons=any(isinstance(node, ast.Compare) for node in ast.walk(module)),
        has_assignments=any(isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for node in ast.walk(module)),
        supported_injector_patterns=_supported_injector_patterns(module, problem.entry_point),
        reference_code_executable=reference_code_executable,
        tests_executable=tests_executable,
        reference_passes_all_tests=reference_passes_all_tests,
        execution_error_message=execution_error_message,
    )


def decide_sanitized_mbpp_problem(
    normalized_record: DatasetProblemRecord,
    config: SanitizedMbppFilterConfig,
) -> SanitizedMbppFilterDecision:
    """Apply explicit filtering rules to one normalized MBPP record."""

    problem = dataset_record_to_problem(normalized_record)
    metadata = analyze_sanitized_mbpp_problem(problem, config)
    rejection_reasons: list[str] = []

    if not metadata.has_top_level_function:
        rejection_reasons.append("No top-level function was found in the reference solution.")
    if not metadata.entry_point_present:
        rejection_reasons.append("The normalized entry_point was not found among top-level functions.")
    if metadata.top_level_function_count > config.max_top_level_function_count:
        rejection_reasons.append(
            f"Reference solution defines {metadata.top_level_function_count} top-level functions, exceeding the limit."
        )
    if metadata.line_count > config.max_reference_line_count:
        rejection_reasons.append(
            f"Reference solution has {metadata.line_count} lines, exceeding the limit of {config.max_reference_line_count}."
        )
    if not config.allow_classes and metadata.has_classes:
        rejection_reasons.append("Reference solution uses classes and is not a simple function-level problem.")
    if not config.allow_file_io and metadata.file_io_keywords_present:
        rejection_reasons.append("Reference solution appears to use file I/O.")
    if metadata.has_disallowed_imports:
        rejection_reasons.append("Reference solution imports disallowed modules.")
    if metadata.has_unknown_imports:
        rejection_reasons.append("Reference solution imports modules outside the conservative allowlist.")
    if not metadata.reference_code_executable:
        rejection_reasons.append("Reference solution could not be executed safely.")
    if not metadata.tests_executable:
        rejection_reasons.append("Tests could not be executed safely.")
    if not metadata.reference_passes_all_tests:
        rejection_reasons.append("Reference solution did not pass all executable tests.")
    if config.require_supported_injector_pattern and not metadata.supported_injector_patterns:
        rejection_reasons.append("No supported bug-injection pattern was detected.")

    return SanitizedMbppFilterDecision(
        record_index=-1,
        task_id=normalized_record.problem_id.removeprefix("mbpp_"),
        problem_id=normalized_record.problem_id,
        accepted=not rejection_reasons,
        rejection_reasons=rejection_reasons,
        filter_metadata=metadata,
        normalized_record=normalized_record,
        raw_record=None,
    )


def _decision_to_export_record(decision: SanitizedMbppFilterDecision) -> dict[str, Any]:
    normalized_payload = asdict(decision.normalized_record) if decision.normalized_record is not None else {}
    return {
        **normalized_payload,
        "record_index": decision.record_index,
        "task_id": decision.task_id,
        "accepted": decision.accepted,
        "rejection_reasons": decision.rejection_reasons,
        "filter_metadata": asdict(decision.filter_metadata),
        "raw_record": decision.raw_record,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def filter_sanitized_mbpp_samples(
    input_path: str,
    output_dir: str,
    config: SanitizedMbppFilterConfig | None = None,
) -> SanitizedMbppFilterArtifacts:
    """Filter sanitized MBPP samples and export accepted/rejected subsets."""

    effective_config = config or SanitizedMbppFilterConfig()
    raw_payloads = load_sanitized_mbpp_raw_payloads(input_path)

    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    rejection_reason_counts: Counter[str] = Counter()

    for record_index, raw_payload in enumerate(raw_payloads, start=1):
        try:
            normalized_record = validate_sanitized_mbpp_record(raw_payload)
            decision = decide_sanitized_mbpp_problem(normalized_record, effective_config)
        except DatasetProtocolError as exc:
            task_id = str(raw_payload.get("task_id", f"record_{record_index}"))
            decision = SanitizedMbppFilterDecision(
                record_index=record_index,
                task_id=task_id,
                problem_id=f"mbpp_{task_id}" if task_id else f"record_{record_index}",
                accepted=False,
                rejection_reasons=[str(exc)],
                filter_metadata=SanitizedMbppFilterMetadata(
                    line_count=0,
                    top_level_function_count=0,
                    top_level_function_names=[],
                    has_top_level_function=False,
                    entry_point_present=False,
                    class_count=0,
                    has_classes=False,
                    imported_modules=[],
                    has_disallowed_imports=False,
                    has_unknown_imports=False,
                    file_io_keywords_present=False,
                    has_loops=False,
                    has_comparisons=False,
                    has_assignments=False,
                    supported_injector_patterns=[],
                    reference_code_executable=False,
                    tests_executable=False,
                    reference_passes_all_tests=False,
                    execution_error_message=str(exc),
                ),
                normalized_record=None,
                raw_record=raw_payload,
            )
        decision.record_index = record_index
        decision.raw_record = raw_payload

        export_record = _decision_to_export_record(decision)
        if decision.accepted:
            accepted_records.append(export_record)
        else:
            rejected_records.append(export_record)
            rejection_reason_counts.update(decision.rejection_reasons)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    accepted_path = output_path / "accepted_samples.jsonl"
    rejected_path = output_path / "rejected_samples.jsonl"
    summary_path = output_path / "filtering_summary.json"

    _write_jsonl(accepted_path, accepted_records)
    _write_jsonl(rejected_path, rejected_records)

    summary_payload = {
        "input_path": str(Path(input_path)),
        "output_dir": str(output_path),
        "total_records": len(raw_payloads),
        "accepted_count": len(accepted_records),
        "rejected_count": len(rejected_records),
        "acceptance_rate": (len(accepted_records) / len(raw_payloads)) if raw_payloads else 0.0,
        "rejection_reason_counts": dict(rejection_reason_counts),
        "filter_config": asdict(effective_config),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return SanitizedMbppFilterArtifacts(
        accepted_samples_jsonl=str(accepted_path),
        rejected_samples_jsonl=str(rejected_path),
        filtering_summary_json=str(summary_path),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter sanitized MBPP samples for the thesis experiment.")
    parser.add_argument("--input-path", required=True, help="Path to the sanitized MBPP file.")
    parser.add_argument("--output-dir", required=True, help="Directory for accepted/rejected outputs.")
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()
    filter_sanitized_mbpp_samples(args.input_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
