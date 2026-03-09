"""Base interfaces and validation helpers for program transforms."""

from __future__ import annotations

import io
import sys
import warnings
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from dataclasses import dataclass, field

from thesis_exp.schemas.sample import BuggyProgramSample, TestCase, TransformedSample


@dataclass(slots=True)
class TestExecutionOutcome:
    """Execution result for one test case."""

    test_case_id: str
    test_case_type: str
    supported: bool
    passed: bool
    exception_type: str = ""
    exception_message: str = ""
    stdout_text: str = ""


@dataclass(slots=True)
class TransformationValidationReport:
    """Validation report for one transformation."""

    syntax_valid: bool
    supported_test_case_count: int
    behavior_preserved: bool | None
    original_test_outcomes: list[TestExecutionOutcome] = field(default_factory=list)
    transformed_test_outcomes: list[TestExecutionOutcome] = field(default_factory=list)
    validation_summary: str = ""


@dataclass(slots=True)
class TransformationResult:
    """Transformed sample plus explicit metadata."""

    transformed_sample: TransformedSample
    transformation_name: str
    changed_lines: list[int]
    transformation_description: str
    bug_type_preserved: bool
    behavior_preserved: bool | None
    validation_report: TransformationValidationReport


def execute_test_case(source_code: str, test_case: TestCase) -> TestExecutionOutcome:
    """Execute one supported test case against source code."""

    if test_case.test_case_type not in {"unit_assertion", "custom"} or not test_case.test_code:
        return TestExecutionOutcome(
            test_case_id=test_case.test_case_id,
            test_case_type=test_case.test_case_type,
            supported=False,
            passed=False,
            exception_message="Unsupported test case for behavior comparison.",
        )

    namespace: dict[str, object] = {"__name__": "__main__"}
    stdout_buffer = io.StringIO()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
            with redirect_stdout(stdout_buffer):
                exec(source_code, namespace, namespace)
                exec(test_case.test_code, namespace, namespace)
    except Exception as exc:  # noqa: BLE001
        return TestExecutionOutcome(
            test_case_id=test_case.test_case_id,
            test_case_type=test_case.test_case_type,
            supported=True,
            passed=False,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            stdout_text=stdout_buffer.getvalue(),
        )

    return TestExecutionOutcome(
        test_case_id=test_case.test_case_id,
        test_case_type=test_case.test_case_type,
        supported=True,
        passed=True,
        stdout_text=stdout_buffer.getvalue(),
    )


def validate_behavior_equivalence(
    original_code: str,
    transformed_code: str,
    test_cases: list[TestCase],
) -> TransformationValidationReport:
    """Compare original and transformed behavior on supported tests."""

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
            compile(transformed_code, "<transformed_code>", "exec")
    except SyntaxError as exc:
        return TransformationValidationReport(
            syntax_valid=False,
            supported_test_case_count=0,
            behavior_preserved=False,
            validation_summary=f"Syntax validation failed: {exc.msg}",
        )

    original_outcomes = [execute_test_case(original_code, test_case) for test_case in test_cases]
    transformed_outcomes = [execute_test_case(transformed_code, test_case) for test_case in test_cases]

    comparable_pairs = [
        (original_outcome, transformed_outcome)
        for original_outcome, transformed_outcome in zip(original_outcomes, transformed_outcomes, strict=True)
        if original_outcome.supported and transformed_outcome.supported
    ]

    if not comparable_pairs:
        return TransformationValidationReport(
            syntax_valid=True,
            supported_test_case_count=0,
            behavior_preserved=None,
            original_test_outcomes=original_outcomes,
            transformed_test_outcomes=transformed_outcomes,
            validation_summary="No supported tests were available for behavior comparison.",
        )

    behavior_preserved = all(
        (
            original_outcome.passed == transformed_outcome.passed
            and original_outcome.exception_type == transformed_outcome.exception_type
            and original_outcome.exception_message == transformed_outcome.exception_message
            and original_outcome.stdout_text == transformed_outcome.stdout_text
        )
        for original_outcome, transformed_outcome in comparable_pairs
    )

    summary = "Behavior matched on all supported tests." if behavior_preserved else "Behavior changed on supported tests."
    return TransformationValidationReport(
        syntax_valid=True,
        supported_test_case_count=len(comparable_pairs),
        behavior_preserved=behavior_preserved,
        original_test_outcomes=original_outcomes,
        transformed_test_outcomes=transformed_outcomes,
        validation_summary=summary,
    )


class BaseProgramTransformer(ABC):
    """Apply a semantic-preserving program transformation."""

    transformation_name: str
    transformation_version: str = "v1"

    @abstractmethod
    def transform(
        self,
        sample: BuggyProgramSample,
        validate_behavior: bool = True,
    ) -> TransformationResult | None:
        """Return one transformed sample or None if not applicable."""


# Backward-compatible alias for the initial scaffold.
BaseTransformer = BaseProgramTransformer
