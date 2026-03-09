"""Registry and factory helpers for program transformers."""

from thesis_exp.transforms.base import BaseProgramTransformer
from thesis_exp.transforms.python import (
    CommentDocstringPlaceholderTransformer,
    EquivalentConstructRewriteTransformer,
    FormattingNormalizationTransformer,
    VariableRenamingTransformer,
)

TRANSFORMER_REGISTRY: dict[str, type[BaseProgramTransformer]] = {
    "variable_renaming": VariableRenamingTransformer,
    "comment_docstring_placeholder": CommentDocstringPlaceholderTransformer,
    "formatting_normalization": FormattingNormalizationTransformer,
    "equivalent_construct_rewrite": EquivalentConstructRewriteTransformer,
}


def create_transformer(transformation_name: str) -> BaseProgramTransformer:
    """Create a transformer instance by name."""

    transformer_class = TRANSFORMER_REGISTRY.get(transformation_name)
    if transformer_class is None:
        raise KeyError(f"Unknown transformer name: {transformation_name}")
    return transformer_class()


def list_transformer_types() -> list[str]:
    """Return registered transformer names."""

    return sorted(TRANSFORMER_REGISTRY)
