"""Single source of truth: whether reference code passes the evaluation test subset.

Filter, ``validate_buggy_sample``, and batch runners must all use this path so the same
problem + ``EvaluationConfig`` always yields the same pass/fail.
"""

from __future__ import annotations

from thesis_exp.evaluators.diagnosis_evaluator import (
    EvaluationConfig,
    RepairExecutionResult,
    execute_patched_code_safely,
    select_tests_for_evaluation,
)
from thesis_exp.schemas.sample import TestCase


def validate_reference_solution(
    reference_code: str,
    test_cases: list[TestCase],
    evaluation_config: EvaluationConfig,
) -> RepairExecutionResult:
    """Run reference code against the same test subset and sandbox as repair/quality evaluation."""

    return execute_patched_code_safely(reference_code, test_cases, evaluation_config)


def reference_passes_all_selected_tests(result: RepairExecutionResult) -> bool:
    """True iff reference executed and every selected test passed (matches quality gate semantics)."""

    return (
        result.syntax_valid
        and not result.timed_out
        and result.total_test_count > 0
        and result.passed_test_count == result.total_test_count
    )


def reference_execution_filter_fields(
    result: RepairExecutionResult,
) -> tuple[bool, bool, bool, str]:
    """Map execution result to ``SanitizedMbppFilterMetadata`` reference fields.

    Returns ``(reference_code_executable, tests_executable, reference_passes_all_tests, execution_error_message)``.
    """

    if result.total_test_count == 0:
        return False, False, False, "No executable tests were found."

    reference_code_executable = result.syntax_valid and not result.timed_out and result.execution_error_message == ""
    tests_executable = reference_code_executable and result.total_test_count > 0
    reference_passes_all_tests = reference_passes_all_selected_tests(result)
    execution_error_message = result.execution_error_message
    return reference_code_executable, tests_executable, reference_passes_all_tests, execution_error_message
