"""Exports for evaluation utilities."""

from thesis_exp.evaluators.diagnosis_evaluator import (
    DiagnosisEvaluator,
    EvaluationConfig,
    RepairExecutionResult,
    RepairTestOutcome,
    compute_bug_type_accuracy,
    compute_consistency_score,
    compute_localization_accuracy,
    compute_repair_without_true_diagnosis,
    compute_repair_success,
    detect_diagnosis_hallucination,
    execute_patched_code_safely,
)
from thesis_exp.evaluators.metrics import (
    compute_diagnosis_hallucination_flag,
    compute_hallucination_rate,
    compute_repair_without_true_diagnosis_flag,
)

__all__ = [
    "compute_bug_type_accuracy",
    "compute_consistency_score",
    "compute_diagnosis_hallucination_flag",
    "compute_hallucination_rate",
    "compute_localization_accuracy",
    "compute_repair_without_true_diagnosis_flag",
    "compute_repair_without_true_diagnosis",
    "compute_repair_success",
    "detect_diagnosis_hallucination",
    "DiagnosisEvaluator",
    "EvaluationConfig",
    "execute_patched_code_safely",
    "RepairExecutionResult",
    "RepairTestOutcome",
]
