"""Automatic evaluation utilities for diagnosis experiments."""

from __future__ import annotations

import io
import multiprocessing
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

from thesis_exp.schemas.sample import (
    BugInjectionRecord,
    BuggyProgramSample,
    EvaluationResult,
    ModelDiagnosisOutput,
    TestCase,
)

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "AssertionError": AssertionError,
    "Exception": Exception,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "ValueError": ValueError,
}

_HALLUCINATION_PHRASES = (
    "multiple bugs",
    "two bugs",
    "several bugs",
    "another bug",
    "another issue",
    "in addition",
    "also has",
)

_KNOWN_BUG_PATTERNS: dict[str, str] = {
    r"\boff[\s_-]?by[\s_-]?one\b": "off_by_one",
    r"\bwrong[\s_-]?loop[\s_-]?bound\b": "wrong_loop_bound",
    r"\baccumulator[\s_-]?(init|initialization)[\s_-]?error\b": "accumulator_init_error",
    r"\bcondition[\s_-]?inversion\b": "condition_inversion",
    r"\bpremature[\s_-]?return\b": "premature_return",
    r"\bwrong[\s_-]?comparison[\s_-]?operator\b": "wrong_comparison_operator",
}


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for automatic diagnosis evaluation."""

    localization_tolerance_lines: int = 1
    repair_timeout_seconds: float = 2.0
    use_hidden_tests_for_repair: bool = True
    use_public_tests_for_repair: bool = True


@dataclass(slots=True)
class RepairTestOutcome:
    """Execution outcome for one repair test."""

    test_case_id: str
    is_hidden: bool
    passed: bool
    exception_type: str = ""
    exception_message: str = ""
    stdout_text: str = ""
    stderr_text: str = ""


@dataclass(slots=True)
class RepairExecutionResult:
    """Aggregated result for patched-code execution."""

    syntax_valid: bool
    timed_out: bool
    passed_test_count: int
    total_test_count: int
    test_outcomes: list[RepairTestOutcome] = field(default_factory=list)
    execution_error_message: str = ""


def _repair_worker(source_code: str, test_cases: list[TestCase], queue: multiprocessing.Queue) -> None:
    """Execute patched code and selected tests in a subprocess."""

    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "__name__": "__main__"}
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
            compile(source_code, "<patched_code>", "exec")
    except SyntaxError as exc:
        queue.put(
            {
                "syntax_valid": False,
                "timed_out": False,
                "passed_test_count": 0,
                "total_test_count": len(test_cases),
                "execution_error_message": f"SyntaxError: {exc.msg}",
                "test_outcomes": [],
            }
        )
        return

    outcomes: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                exec(source_code, namespace, namespace)
            except Exception as exc:  # noqa: BLE001
                queue.put(
                    {
                        "syntax_valid": True,
                        "timed_out": False,
                        "passed_test_count": 0,
                        "total_test_count": len(test_cases),
                        "execution_error_message": f"{type(exc).__name__}: {exc}",
                        "test_outcomes": [],
                    }
                )
                return

            passed_test_count = 0
            for test_case in test_cases:
                try:
                    exec(test_case.test_code, namespace, namespace)
                    outcomes.append(
                        {
                            "test_case_id": test_case.test_case_id,
                            "is_hidden": test_case.is_hidden,
                            "passed": True,
                            "stdout_text": stdout_buffer.getvalue(),
                            "stderr_text": stderr_buffer.getvalue(),
                        }
                    )
                    passed_test_count += 1
                except Exception as exc:  # noqa: BLE001
                    outcomes.append(
                        {
                            "test_case_id": test_case.test_case_id,
                            "is_hidden": test_case.is_hidden,
                            "passed": False,
                            "exception_type": type(exc).__name__,
                            "exception_message": str(exc),
                            "stdout_text": stdout_buffer.getvalue(),
                            "stderr_text": stderr_buffer.getvalue(),
                        }
                    )

    queue.put(
        {
            "syntax_valid": True,
            "timed_out": False,
            "passed_test_count": passed_test_count,
            "total_test_count": len(test_cases),
            "execution_error_message": "",
            "test_outcomes": outcomes,
        }
    )


def _select_repair_tests(test_cases: list[TestCase], config: EvaluationConfig) -> list[TestCase]:
    """Select public and hidden tests according to config."""

    selected = []
    for test_case in test_cases:
        if test_case.test_case_type not in {"unit_assertion", "custom"} or not test_case.test_code:
            continue
        if test_case.is_hidden and not config.use_hidden_tests_for_repair:
            continue
        if not test_case.is_hidden and not config.use_public_tests_for_repair:
            continue
        selected.append(test_case)
    return selected


def execute_patched_code_safely(
    patched_code: str | None,
    test_cases: list[TestCase],
    config: EvaluationConfig,
) -> RepairExecutionResult:
    """Execute patched code in a subprocess with a timeout."""

    selected_test_cases = _select_repair_tests(test_cases, config)
    if not patched_code:
        return RepairExecutionResult(
            syntax_valid=False,
            timed_out=False,
            passed_test_count=0,
            total_test_count=len(selected_test_cases),
            execution_error_message="No patched_code was provided by the model.",
        )

    context = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = context.Queue()
    process = context.Process(target=_repair_worker, args=(patched_code, selected_test_cases, queue))
    process.start()
    process.join(config.repair_timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return RepairExecutionResult(
            syntax_valid=False,
            timed_out=True,
            passed_test_count=0,
            total_test_count=len(selected_test_cases),
            execution_error_message="Patched code execution timed out.",
        )

    if queue.empty():
        return RepairExecutionResult(
            syntax_valid=False,
            timed_out=False,
            passed_test_count=0,
            total_test_count=len(selected_test_cases),
            execution_error_message="No repair execution result was returned by the worker.",
        )

    payload = queue.get()
    return RepairExecutionResult(
        syntax_valid=bool(payload["syntax_valid"]),
        timed_out=bool(payload["timed_out"]),
        passed_test_count=int(payload["passed_test_count"]),
        total_test_count=int(payload["total_test_count"]),
        execution_error_message=str(payload.get("execution_error_message", "")),
        test_outcomes=[
            RepairTestOutcome(
                test_case_id=item["test_case_id"],
                is_hidden=bool(item["is_hidden"]),
                passed=bool(item["passed"]),
                exception_type=str(item.get("exception_type", "")),
                exception_message=str(item.get("exception_message", "")),
                stdout_text=str(item.get("stdout_text", "")),
                stderr_text=str(item.get("stderr_text", "")),
            )
            for item in payload.get("test_outcomes", [])
        ],
    )


def compute_bug_type_accuracy(
    ground_truth: BugInjectionRecord,
    diagnosis: ModelDiagnosisOutput,
) -> float:
    """Return 1.0 when the predicted bug type matches the injected bug type."""

    return float(diagnosis.parsed_bug_type == ground_truth.bug_type)


def compute_localization_accuracy(
    ground_truth: BugInjectionRecord,
    diagnosis: ModelDiagnosisOutput,
    tolerance_lines: int = 1,
) -> float:
    """Return 1.0 when the predicted line falls within a tolerant gold window."""

    predicted_line = diagnosis.parsed_bug_line_start
    if predicted_line is None:
        return 0.0

    lower_bound = ground_truth.injection_line_start - tolerance_lines
    upper_bound = ground_truth.injection_line_end + tolerance_lines
    return float(lower_bound <= predicted_line <= upper_bound)


def detect_diagnosis_hallucination(diagnosis: ModelDiagnosisOutput) -> bool:
    """Detect unsupported extra diagnoses for a single-bug sample."""

    candidate_text = " ".join(
        part for part in (diagnosis.raw_response_text, diagnosis.parsed_bug_explanation) if part
    ).lower()

    mentioned_bug_types = {
        canonical_label
        for pattern, canonical_label in _KNOWN_BUG_PATTERNS.items()
        if re.search(pattern, candidate_text)
    }

    if len(mentioned_bug_types) >= 2:
        return True

    return any(phrase in candidate_text for phrase in _HALLUCINATION_PHRASES)


def compute_repair_success(
    diagnosis: ModelDiagnosisOutput,
    test_cases: list[TestCase],
    config: EvaluationConfig,
) -> tuple[float, RepairExecutionResult]:
    """Execute patched code and return a binary repair-success score."""

    execution_result = execute_patched_code_safely(diagnosis.parsed_repaired_code, test_cases, config)
    if execution_result.total_test_count == 0:
        return 0.0, execution_result
    success = (
        execution_result.syntax_valid
        and not execution_result.timed_out
        and execution_result.passed_test_count == execution_result.total_test_count
    )
    return float(success), execution_result


def compute_repair_without_true_diagnosis(
    bug_type_accuracy: float,
    localization_accuracy: float,
    repair_success: float,
) -> bool:
    """Return True when repair succeeds without a faithful diagnosis."""

    return bool(repair_success == 1.0 and (bug_type_accuracy < 1.0 or localization_accuracy < 1.0))


def compute_consistency_score(results: list[EvaluationResult]) -> float:
    """Compute agreement across transformed variants using bug type and line only."""

    if not results:
        return 0.0
    if len(results) == 1:
        return 1.0

    signatures = [
        (
            result.predicted_bug_type or "<missing>",
            result.predicted_bug_line_start if result.predicted_bug_line_start is not None else "<missing>",
        )
        for result in results
    ]

    modal_count = max(signatures.count(signature) for signature in signatures)
    return modal_count / len(signatures)


class DiagnosisEvaluator:
    """Evaluate diagnosis outputs against injected ground truth."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()

    def evaluate_single(
        self,
        sample: BuggyProgramSample,
        diagnosis: ModelDiagnosisOutput,
    ) -> EvaluationResult:
        """Evaluate one parsed diagnosis against one buggy sample."""

        bug_type_accuracy = compute_bug_type_accuracy(sample.bug_injection_record, diagnosis)
        localization_accuracy = compute_localization_accuracy(
            sample.bug_injection_record,
            diagnosis,
            tolerance_lines=self.config.localization_tolerance_lines,
        )
        localization_accuracy_exact = compute_localization_accuracy(
            sample.bug_injection_record,
            diagnosis,
            tolerance_lines=0,
        )
        repair_success, repair_execution_result = compute_repair_success(
            diagnosis,
            sample.test_cases,
            self.config,
        )
        hallucination_detected = detect_diagnosis_hallucination(diagnosis)
        repair_without_true_diagnosis = compute_repair_without_true_diagnosis(
            bug_type_accuracy,
            localization_accuracy_exact,
            repair_success,
        )

        notes = [
            f"repair_tests_passed={repair_execution_result.passed_test_count}/{repair_execution_result.total_test_count}",
        ]
        if repair_execution_result.execution_error_message:
            notes.append(repair_execution_result.execution_error_message)
        if diagnosis.parsing_error_message:
            notes.append(f"parse_errors={diagnosis.parsing_error_message}")
        if repair_without_true_diagnosis:
            notes.append("repair_succeeded_without_true_diagnosis")

        return EvaluationResult(
            evaluation_result_id=f"eval::{diagnosis.diagnosis_output_id}",
            sample_id=diagnosis.sample_id,
            transformed_sample_id=diagnosis.transformed_sample_id,
            diagnosis_output_id=diagnosis.diagnosis_output_id,
            ground_truth_bug_type=sample.bug_injection_record.bug_type,
            predicted_bug_type=diagnosis.parsed_bug_type,
            ground_truth_injection_line_start=sample.bug_injection_record.injection_line_start,
            ground_truth_injection_line_end=sample.bug_injection_record.injection_line_end,
            predicted_bug_line_start=diagnosis.parsed_bug_line_start,
            predicted_bug_line_end=diagnosis.parsed_bug_line_end,
            bug_type_accuracy=bug_type_accuracy,
            localization_accuracy=localization_accuracy,
            localization_accuracy_exact=localization_accuracy_exact,
            repair_success=repair_success,
            hallucination_rate=float(hallucination_detected),
            consistency_score=0.0,
            hallucination_detected=hallucination_detected,
            repair_without_true_diagnosis=repair_without_true_diagnosis,
            evaluation_notes=" | ".join(notes),
        )

    def evaluate_consistency_group(self, results: list[EvaluationResult]) -> float:
        """Compute and annotate consistency across transformed variants."""

        consistency_score = compute_consistency_score(results)
        for result in results:
            result.consistency_score = consistency_score
        return consistency_score
