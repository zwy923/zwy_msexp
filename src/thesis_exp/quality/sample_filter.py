"""Quality filters for injected buggy samples."""

from __future__ import annotations

from dataclasses import dataclass, field

from thesis_exp.evaluators.diagnosis_evaluator import EvaluationConfig, execute_patched_code_safely
from thesis_exp.evaluators.reference_validation import reference_passes_all_selected_tests
from thesis_exp.schemas.sample import BuggyProgramSample


@dataclass(slots=True)
class SampleQualityConfig:
    """Configuration for sample-level quality filtering."""

    max_changed_lines: int = 3
    require_reference_pass_all_tests: bool = True
    require_buggy_fail_some_tests: bool = True
    require_buggy_syntax_valid: bool = True
    require_buggy_executable: bool = True


@dataclass(slots=True)
class BuggySampleQualityReport:
    """Validation report for one injected buggy sample."""

    accepted: bool
    sample_id: str
    changed_line_count: int
    selected_test_count: int
    reference_passes_all_tests: bool
    buggy_code_syntax_valid: bool
    buggy_code_executable: bool
    buggy_fails_at_least_one_test: bool
    changed_lines_within_limit: bool
    failure_reasons: list[str] = field(default_factory=list)


def validate_buggy_sample(
    sample: BuggyProgramSample,
    quality_config: SampleQualityConfig,
    evaluation_config: EvaluationConfig | None = None,
) -> BuggySampleQualityReport:
    """Validate that an injected buggy sample is clean and usable."""

    effective_evaluation_config = evaluation_config or EvaluationConfig()
    reference_execution = execute_patched_code_safely(
        sample.reference_code,
        sample.test_cases,
        effective_evaluation_config,
    )
    buggy_execution = execute_patched_code_safely(
        sample.buggy_code,
        sample.test_cases,
        effective_evaluation_config,
    )

    reference_passes_all_tests = reference_passes_all_selected_tests(reference_execution)
    buggy_code_syntax_valid = buggy_execution.syntax_valid
    buggy_code_executable = (
        buggy_execution.syntax_valid
        and not buggy_execution.timed_out
        and (
            bool(buggy_execution.test_outcomes)
            or not buggy_execution.execution_error_message
        )
    )
    buggy_fails_at_least_one_test = (
        buggy_execution.total_test_count > 0
        and buggy_execution.passed_test_count < buggy_execution.total_test_count
    )
    changed_lines_within_limit = (
        sample.bug_injection_record.changed_line_count <= quality_config.max_changed_lines
    )

    failure_reasons: list[str] = []
    if quality_config.require_reference_pass_all_tests and not reference_passes_all_tests:
        failure_reasons.append("Reference code did not pass all selected tests.")
    if quality_config.require_buggy_syntax_valid and not buggy_code_syntax_valid:
        failure_reasons.append("Injected buggy code is not syntactically valid.")
    if quality_config.require_buggy_executable and not buggy_code_executable:
        failure_reasons.append("Injected buggy code is not executable under the selected tests.")
    if quality_config.require_buggy_fail_some_tests and not buggy_fails_at_least_one_test:
        failure_reasons.append("Injected buggy code did not fail any selected test.")
    if not changed_lines_within_limit:
        failure_reasons.append(
            f"Injected buggy code changed {sample.bug_injection_record.changed_line_count} lines, exceeding the limit."
        )

    return BuggySampleQualityReport(
        accepted=not failure_reasons,
        sample_id=sample.sample_id,
        changed_line_count=sample.bug_injection_record.changed_line_count,
        selected_test_count=buggy_execution.total_test_count,
        reference_passes_all_tests=reference_passes_all_tests,
        buggy_code_syntax_valid=buggy_code_syntax_valid,
        buggy_code_executable=buggy_code_executable,
        buggy_fails_at_least_one_test=buggy_fails_at_least_one_test,
        changed_lines_within_limit=changed_lines_within_limit,
        failure_reasons=failure_reasons,
    )
