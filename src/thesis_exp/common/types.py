"""Common type definitions."""

from typing import Literal


BugType = Literal[
    "off_by_one",
    "wrong_loop_bound",
    "accumulator_init_error",
    "condition_inversion",
    "premature_return",
    "wrong_comparison_operator",
]

MetricName = Literal[
    "bug_type_accuracy",
    "localization_accuracy",
    "repair_success",
    "hallucination_rate",
    "consistency_score",
]

TestCaseType = Literal[
    "stdin_stdout",
    "unit_assertion",
    "property_check",
    "custom",
]
