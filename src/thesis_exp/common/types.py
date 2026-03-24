"""Common type definitions."""

from typing import Literal

BugType = Literal[
    "loop_boundary_error",
    "accumulator_init_error",
    "conditional_logic_error",
    "premature_return",
]

# Legacy labels (pre-merge) still appear in old result JSONL; normalize for metrics.
_LEGACY_FINE_BUG_TYPE: dict[str, str] = {
    "off_by_one": "loop_boundary_error",
    "wrong_loop_bound": "loop_boundary_error",
    "condition_inversion": "conditional_logic_error",
    "wrong_comparison_operator": "conditional_logic_error",
}


def canonical_fine_bug_type(fine_type: str | None) -> str | None:
    """Map fine-grained bug type to the current taxonomy (merges legacy names)."""

    if fine_type is None or not str(fine_type).strip():
        return None
    key = str(fine_type).strip()
    return _LEGACY_FINE_BUG_TYPE.get(key, key)


# Coarse-grained bug type mapping for less strict evaluation.
FINE_TO_COARSE_BUG_TYPE: dict[str, str] = {
    "conditional_logic_error": "conditional_logic_error",
    "loop_boundary_error": "loop_bound_error",
    "accumulator_init_error": "accumulator_error",
    "premature_return": "control_flow_error",
}


def to_coarse_bug_type(fine_type: str | None) -> str | None:
    """Map fine-grained bug type to coarse category. Returns None if input is None or unknown."""
    if fine_type is None or not fine_type.strip():
        return None
    fine = canonical_fine_bug_type(fine_type.strip()) or fine_type.strip()
    return FINE_TO_COARSE_BUG_TYPE.get(fine, fine)


MetricName = Literal[
    "bug_type_accuracy",
    "bug_type_accuracy_coarse",
    "localization_accuracy",
    "patch_passes_selected_tests",
    "repair_success",
    "diagnosis_hallucination_rate",
    "narrative_hallucination_rate",
    "hallucination_rate",
]

TestCaseType = Literal[
    "stdin_stdout",
    "unit_assertion",
    "property_check",
    "custom",
]
