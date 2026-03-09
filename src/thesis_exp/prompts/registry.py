"""Registry helpers for prompt variants."""

from thesis_exp.prompts.base import PromptContext, PromptVariant
from thesis_exp.prompts.diagnosis import BugDiagnosisPromptBuilder, PromptBuilderOptions


def build_diagnosis_prompt(
    context: PromptContext,
    variant: PromptVariant,
    *,
    options: PromptBuilderOptions | None = None,
):
    """Build one diagnosis prompt for the requested variant."""

    return BugDiagnosisPromptBuilder(options=options).build_prompt(context, variant)


def list_prompt_variants() -> list[str]:
    """Return supported diagnosis prompt variants."""

    return [
        "direct_diagnosis",
        "diagnosis_with_execution_feedback",
        "diagnosis_with_execution_feedback_no_leakage",
        "diagnosis_with_self_check",
    ]
