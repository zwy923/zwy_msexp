"""Exports for transformation components."""

from thesis_exp.transforms.base import (
    BaseProgramTransformer,
    BaseTransformer,
    TestExecutionOutcome,
    TransformationResult,
    TransformationValidationReport,
    execute_test_case,
    validate_behavior_equivalence,
)
from thesis_exp.transforms.python import (
    CommentDocstringPlaceholderTransformer,
    EquivalentConstructRewriteTransformer,
    FormattingNormalizationTransformer,
    VariableRenamingTransformer,
)
from thesis_exp.transforms.registry import create_transformer, list_transformer_types

__all__ = [
    "BaseProgramTransformer",
    "BaseTransformer",
    "CommentDocstringPlaceholderTransformer",
    "create_transformer",
    "EquivalentConstructRewriteTransformer",
    "execute_test_case",
    "FormattingNormalizationTransformer",
    "list_transformer_types",
    "TestExecutionOutcome",
    "TransformationResult",
    "TransformationValidationReport",
    "validate_behavior_equivalence",
    "VariableRenamingTransformer",
]
