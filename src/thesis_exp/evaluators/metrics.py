"""Helpers for accessing evaluation metrics."""

from thesis_exp.schemas.sample import EvaluationResult


def compute_bug_type_accuracy(result: EvaluationResult) -> float:
    return result.bug_type_accuracy


def compute_localization_accuracy(result: EvaluationResult) -> float:
    return result.localization_accuracy


def compute_repair_success(result: EvaluationResult) -> float:
    return result.repair_success


def compute_hallucination_rate(result: EvaluationResult) -> float:
    return result.hallucination_rate


def compute_diagnosis_hallucination_flag(result: EvaluationResult) -> bool:
    return result.hallucination_detected


def compute_repair_without_true_diagnosis_flag(result: EvaluationResult) -> bool:
    return result.repair_without_true_diagnosis


def compute_consistency_score(result: EvaluationResult) -> float:
    return result.consistency_score
