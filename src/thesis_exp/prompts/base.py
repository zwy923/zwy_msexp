"""Core abstractions for diagnosis prompt building."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from thesis_exp.llm.base import PromptTemplate

PromptVariant = Literal[
    "direct_diagnosis",
    "diagnosis_with_execution_feedback",  # EF-leaky: includes full assertions (expected outputs) + full traceback
    "diagnosis_with_execution_feedback_no_leakage",  # EF-no-answer: no expected output, only failure count + exception types
    "diagnosis_with_self_check",
]


@dataclass(slots=True)
class ExecutionFeedback:
    """Structured execution feedback for prompt context."""

    feedback_summary: str = ""
    traceback_text: str = ""
    failing_stdout: str = ""
    failing_stderr: str = ""


@dataclass(slots=True)
class InputOutputSpecification:
    """Optional input and output specification."""

    input_specification: str = ""
    output_specification: str = ""


@dataclass(slots=True)
class PromptContext:
    """Structured context used to build a diagnosis prompt."""

    sample_id: str
    problem_statement: str
    buggy_student_code: str
    programming_language: str = "python"
    input_output_specification: InputOutputSpecification | None = None
    failing_test_cases: list[str] = field(default_factory=list)
    execution_feedback: ExecutionFeedback | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class BasePromptBuilder(ABC):
    """Base interface for diagnosis prompt builders."""

    @abstractmethod
    def build_prompt(
        self,
        context: PromptContext,
        variant: PromptVariant,
    ) -> PromptTemplate:
        """Build a provider-neutral prompt template."""
