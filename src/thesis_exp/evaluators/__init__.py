"""Exports for evaluation utilities."""

from thesis_exp.evaluators.diagnosis_evaluator import (
    DiagnosisEvaluator,
    EvaluationConfig,
    RepairExecutionResult,
    RepairTestOutcome,
    compute_bug_type_accuracy,
    compute_bug_type_accuracy_coarse,
    compute_diagnosis_hallucination,
    compute_localization_accuracy,
    compute_repair_without_true_diagnosis,
    compute_repair_success,
    detect_diagnosis_hallucination,
    detect_narrative_hallucination,
    execute_patched_code_safely,
    select_tests_for_evaluation,
)
from thesis_exp.evaluators.reference_validation import (
    reference_execution_filter_fields,
    reference_passes_all_selected_tests,
    validate_reference_solution,
)
from thesis_exp.evaluators.metrics import (
    compute_diagnosis_hallucination_flag,
    compute_hallucination_rate,
    compute_patch_passes_selected_tests,
    compute_repair_without_true_diagnosis_flag,
)

__all__ = [
    "compute_bug_type_accuracy",
    "compute_bug_type_accuracy_coarse",
    "compute_diagnosis_hallucination_flag",
    "compute_hallucination_rate",
    "compute_localization_accuracy",
    "compute_patch_passes_selected_tests",
    "compute_repair_without_true_diagnosis_flag",
    "compute_repair_without_true_diagnosis",
    "compute_repair_success",
    "compute_diagnosis_hallucination",
    "detect_diagnosis_hallucination",
    "detect_narrative_hallucination",
    "DiagnosisEvaluator",
    "EvaluationConfig",
    "execute_patched_code_safely",
    "reference_execution_filter_fields",
    "reference_passes_all_selected_tests",
    "RepairExecutionResult",
    "RepairTestOutcome",
    "select_tests_for_evaluation",
    "validate_reference_solution",
]
