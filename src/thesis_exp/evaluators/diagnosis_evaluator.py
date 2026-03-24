"""Automatic evaluation utilities for diagnosis experiments."""

from __future__ import annotations

import io
import multiprocessing
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any

# Suppress SyntaxWarning from MBPP reference/test code with invalid escape sequences
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

from thesis_exp.execution.safe_exec import build_safe_exec_builtins
from thesis_exp.schemas.sample import (
    BugInjectionRecord,
    BuggyProgramSample,
    EvaluationResult,
    ModelDiagnosisOutput,
    TestCase,
)

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
    r"\boff[\s_-]?by[\s_-]?one\b": "loop_boundary_error",
    r"\bwrong[\s_-]?loop[\s_-]?bound\b": "loop_boundary_error",
    r"\bloop[\s_-]?boundary[\s_-]?error\b": "loop_boundary_error",
    r"\baccumulator[\s_-]?(init|initialization)[\s_-]?error\b": "accumulator_init_error",
    r"\bcondition[\s_-]?inversion\b": "conditional_logic_error",
    r"\bconditional[\s_-]?logic[\s_-]?error\b": "conditional_logic_error",
    r"\bpremature[\s_-]?return\b": "premature_return",
    r"\bwrong[\s_-]?comparison[\s_-]?operator\b": "conditional_logic_error",
}


@dataclass(slots=True)
class EvaluationConfig:
    """Configuration for automatic diagnosis evaluation."""

    localization_tolerance_lines: int = 1
    repair_timeout_seconds: float = 2.0
    use_hidden_tests_for_repair: bool = True
    use_public_tests_for_repair: bool = True
    use_mutation_adequacy: bool = True
    min_mutants_killed: int = 1
    max_mutants_for_adequacy: int = 8


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

    namespace: dict[str, Any] = {"__builtins__": build_safe_exec_builtins(), "__name__": "__main__"}
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


def select_tests_for_evaluation(test_cases: list[TestCase], config: EvaluationConfig) -> list[TestCase]:
    """Tests used for reference validation, buggy-sample quality, and repair scoring.

    Public wrapper so filters and scripts share the same selection rules as ``execute_patched_code_safely``.
    """

    return _select_repair_tests(test_cases, config)


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
    """Return 1.0 when the predicted bug type matches the injected bug type (fine-grained)."""

    from thesis_exp.common.types import canonical_fine_bug_type

    gt = canonical_fine_bug_type(str(ground_truth.bug_type))
    pr = canonical_fine_bug_type(diagnosis.parsed_bug_type)
    if gt is None or pr is None:
        return 0.0
    return float(pr == gt)


def compute_bug_type_accuracy_coarse(
    ground_truth: BugInjectionRecord,
    diagnosis: ModelDiagnosisOutput,
) -> float:
    """Return 1.0 when predicted and ground-truth bug types map to the same coarse category.

    Coarse mapping groups structurally similar bugs (e.g. loop_boundary_error → loop_bound_error).
    For conditional logic, fine and coarse both use ``conditional_logic_error`` (legacy labels are
    normalized via ``canonical_fine_bug_type``).
    """
    from thesis_exp.common.types import to_coarse_bug_type

    gt_coarse = to_coarse_bug_type(ground_truth.bug_type)
    pred_coarse = to_coarse_bug_type(diagnosis.parsed_bug_type)
    if gt_coarse is None or pred_coarse is None:
        return 0.0
    return float(gt_coarse == pred_coarse)


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


def detect_narrative_hallucination(diagnosis: ModelDiagnosisOutput) -> bool:
    """Detect narrative hallucination: text claims multiple bugs or unsupported error causes.

    B. Narrative hallucination: extra claims of multiple bugs, or introducing unsupported
    error explanations. Uses text patterns (e.g. 'multiple bugs', 'another bug').
    """
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


# Backward-compat alias (old name referred to narrative-style hallucination)
detect_diagnosis_hallucination = detect_narrative_hallucination


def compute_diagnosis_hallucination(
    ground_truth: BugInjectionRecord,
    diagnosis: ModelDiagnosisOutput,
) -> float:
    """Return 1.0 when predicted bug type does not match ground truth (diagnosis hallucination).

    A. Diagnosis hallucination: the predicted single bug type differs from ground truth.
    """
    from thesis_exp.common.types import canonical_fine_bug_type

    if diagnosis.parsed_bug_type is None:
        return 0.0
    gt = canonical_fine_bug_type(str(ground_truth.bug_type))
    pr = canonical_fine_bug_type(diagnosis.parsed_bug_type)
    if gt is None or pr is None:
        return 1.0
    return float(pr != gt)


# Values for ``EvaluationResult.mutation_adequacy_status`` (also used in evaluation_notes).
REPAIR_PATCH_NO_TESTS_SELECTED = "repair_patch_no_tests_selected"
REPAIR_PATCH_TESTS_FAILED = "repair_patch_tests_failed"
MUTATION_ADEQUACY_DISABLED = "mutation_adequacy_disabled"
MUTATION_ADEQUACY_SKIPPED_NO_REFERENCE = "mutation_adequacy_skipped_no_reference"
MUTATION_ADEQUACY_NOT_ASSESSABLE_NO_MUTANTS = "mutation_adequacy_not_assessable_no_mutants"
MUTATION_ADEQUACY_FAILED_NO_MUTANTS_KILLED = "mutation_adequacy_failed_no_mutants_killed"
MUTATION_ADEQUACY_PASSED = "mutation_adequacy_passed"
MUTATION_ADEQUACY_ERROR_REFERENCE_FAILED_TESTS = "mutation_adequacy_error_reference_failed_tests"
MUTATION_ADEQUACY_ERROR_EMPTY_REFERENCE = "mutation_adequacy_error_empty_reference"


def compute_repair_success(
    diagnosis: ModelDiagnosisOutput,
    test_cases: list[TestCase],
    config: EvaluationConfig,
    reference_code: str | None = None,
) -> tuple[float, float, RepairExecutionResult, str]:
    """Execute patched code; return patch-test pass, strict repair success, execution, status.

    Returns ``(patch_passes_selected_tests, repair_success_strict, execution_result, mutation_adequacy_status)``.

    ``patch_passes_selected_tests`` is 1.0 iff the patch passes every selected test (no mutation gate).
    ``repair_success_strict`` matches the historical ``repair_success`` field (adds mutation adequacy when on).

    When ``use_mutation_adequacy`` is True and ``reference_code`` is set, the strict score requires
    killing at least ``min_mutants_killed`` mutants **unless** no mutants could be generated
    (``mutation_adequacy_not_assessable_no_mutants``): in that case the patch is not
    penalized—mutation coverage is absent, not weak tests.
    """
    execution_result = execute_patched_code_safely(diagnosis.parsed_repaired_code, test_cases, config)
    if execution_result.total_test_count == 0:
        return 0.0, 0.0, execution_result, REPAIR_PATCH_NO_TESTS_SELECTED
    tests_pass = (
        execution_result.syntax_valid
        and not execution_result.timed_out
        and execution_result.passed_test_count == execution_result.total_test_count
    )
    patch_passes = 1.0 if tests_pass else 0.0
    if not tests_pass:
        return patch_passes, 0.0, execution_result, REPAIR_PATCH_TESTS_FAILED

    if not config.use_mutation_adequacy:
        return patch_passes, 1.0, execution_result, MUTATION_ADEQUACY_DISABLED

    if not reference_code or not str(reference_code).strip():
        return patch_passes, 1.0, execution_result, MUTATION_ADEQUACY_SKIPPED_NO_REFERENCE

    from thesis_exp.evaluators.mutation import check_mutation_adequacy

    adequacy = check_mutation_adequacy(
        reference_code=reference_code,
        test_cases=test_cases,
        config=config,
        max_mutants=config.max_mutants_for_adequacy,
        min_required_killed=config.min_mutants_killed,
    )

    if adequacy.empty_reason == "empty_reference_code":
        return patch_passes, 0.0, execution_result, MUTATION_ADEQUACY_ERROR_EMPTY_REFERENCE

    if adequacy.empty_reason == "reference_failed_tests":
        return patch_passes, 0.0, execution_result, MUTATION_ADEQUACY_ERROR_REFERENCE_FAILED_TESTS

    if adequacy.total_mutants == 0 and adequacy.empty_reason == "no_mutants_generated":
        return patch_passes, 1.0, execution_result, MUTATION_ADEQUACY_NOT_ASSESSABLE_NO_MUTANTS

    if not adequacy.adequate:
        return patch_passes, 0.0, execution_result, MUTATION_ADEQUACY_FAILED_NO_MUTANTS_KILLED

    return patch_passes, 1.0, execution_result, MUTATION_ADEQUACY_PASSED


def _format_mutation_adequacy_note(status: str) -> str | None:
    """Human-readable fragment for ``evaluation_notes``."""

    if status == MUTATION_ADEQUACY_NOT_ASSESSABLE_NO_MUTANTS:
        return "mutation_adequacy=not_assessable(no_mutants_generated)"
    if status == MUTATION_ADEQUACY_FAILED_NO_MUTANTS_KILLED:
        return "tests_adequate=false(mutants_generated_but_none_killed)"
    if status == MUTATION_ADEQUACY_ERROR_REFERENCE_FAILED_TESTS:
        return "mutation_adequacy=error(reference_did_not_pass_tests)"
    if status == MUTATION_ADEQUACY_ERROR_EMPTY_REFERENCE:
        return "mutation_adequacy=error(empty_reference_code)"
    if status == MUTATION_ADEQUACY_PASSED:
        return "mutation_adequacy=passed"
    if status == MUTATION_ADEQUACY_DISABLED:
        return None
    if status == MUTATION_ADEQUACY_SKIPPED_NO_REFERENCE:
        return "mutation_adequacy=skipped(no_reference_code)"
    return None


def compute_repair_without_true_diagnosis(
    bug_type_accuracy: float,
    localization_accuracy: float,
    repair_success: float,
) -> bool:
    """Return True when repair succeeds without a faithful diagnosis."""

    return bool(repair_success == 1.0 and (bug_type_accuracy < 1.0 or localization_accuracy < 1.0))


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
        bug_type_accuracy_coarse = compute_bug_type_accuracy_coarse(sample.bug_injection_record, diagnosis)
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
        patch_passes_selected_tests, repair_success, repair_execution_result, mutation_adequacy_status = (
            compute_repair_success(
                diagnosis,
                sample.test_cases,
                self.config,
                reference_code=sample.reference_code,
            )
        )
        diagnosis_hallucination = compute_diagnosis_hallucination(sample.bug_injection_record, diagnosis)
        narrative_hallucination_detected = detect_narrative_hallucination(diagnosis)
        repair_without_true_diagnosis = compute_repair_without_true_diagnosis(
            bug_type_accuracy,
            localization_accuracy_exact,
            repair_success,
        )

        notes = [
            f"patch_passes_selected_tests={int(patch_passes_selected_tests)}",
            f"repair_tests_passed={repair_execution_result.passed_test_count}/{repair_execution_result.total_test_count}",
        ]
        mut_note = _format_mutation_adequacy_note(mutation_adequacy_status)
        if mut_note is not None:
            notes.append(mut_note)
        if repair_execution_result.execution_error_message:
            notes.append(repair_execution_result.execution_error_message)
        if diagnosis.parsing_error_message:
            notes.append(f"parse_errors={diagnosis.parsing_error_message}")
        if repair_without_true_diagnosis:
            notes.append("repair_succeeded_without_true_diagnosis")

        return EvaluationResult(
            evaluation_result_id=f"eval::{diagnosis.diagnosis_output_id}",
            sample_id=diagnosis.sample_id,
            diagnosis_output_id=diagnosis.diagnosis_output_id,
            ground_truth_bug_type=sample.bug_injection_record.bug_type,
            predicted_bug_type=diagnosis.parsed_bug_type,
            ground_truth_injection_line_start=sample.bug_injection_record.injection_line_start,
            ground_truth_injection_line_end=sample.bug_injection_record.injection_line_end,
            predicted_bug_line_start=diagnosis.parsed_bug_line_start,
            predicted_bug_line_end=diagnosis.parsed_bug_line_end,
            bug_type_accuracy=bug_type_accuracy,
            bug_type_accuracy_coarse=bug_type_accuracy_coarse,
            localization_accuracy=localization_accuracy,
            localization_accuracy_exact=localization_accuracy_exact,
            patch_passes_selected_tests=patch_passes_selected_tests,
            repair_success=repair_success,
            diagnosis_hallucination_rate=diagnosis_hallucination,
            narrative_hallucination_rate=float(narrative_hallucination_detected),
            hallucination_rate=float(narrative_hallucination_detected),
            narrative_hallucination_detected=narrative_hallucination_detected,
            hallucination_detected=narrative_hallucination_detected,
            repair_without_true_diagnosis=repair_without_true_diagnosis,
            evaluation_notes=" | ".join(notes),
            mutation_adequacy_status=mutation_adequacy_status,
        )
