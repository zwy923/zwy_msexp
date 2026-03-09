"""Prompt builder for bug diagnosis experiments."""

from __future__ import annotations

from dataclasses import dataclass

from thesis_exp.llm.base import PromptTemplate
from thesis_exp.prompts.base import (
    BasePromptBuilder,
    PromptContext,
    PromptVariant,
)


DEFAULT_RESPONSE_SCHEMA_NAME = "bug_diagnosis_json_v1"


@dataclass(slots=True)
class PromptBuilderOptions:
    """Options that control prompt rendering."""

    include_language_tag: bool = True
    include_output_schema_example: bool = True
    require_json_only_response: bool = True


class BugDiagnosisPromptBuilder(BasePromptBuilder):
    """Build structured prompts for bug diagnosis."""

    def __init__(self, options: PromptBuilderOptions | None = None) -> None:
        self.options = options or PromptBuilderOptions()

    def build_prompt(
        self,
        context: PromptContext,
        variant: PromptVariant,
    ) -> PromptTemplate:
        """Build one prompt template for a diagnosis variant."""

        system_prompt = self._build_system_prompt(variant)
        user_prompt = self._build_user_prompt(context, variant)
        return PromptTemplate(
            template_name=variant,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt,
            response_schema_name=DEFAULT_RESPONSE_SCHEMA_NAME,
            is_fully_rendered=True,
        )

    def _build_system_prompt(self, variant: PromptVariant) -> str:
        shared_lines = [
            "You are an expert programming tutor diagnosing a student's buggy program.",
            "Your job is to identify the most likely bug, explain it clearly, and propose a repair strategy.",
            "Do not output markdown fences or any text outside the final JSON object.",
        ]

        if variant == "diagnosis_with_self_check":
            shared_lines.append(
                "Before producing the final answer, internally verify that your diagnosis is consistent with the code and any provided feedback."
            )

        return "\n".join(shared_lines)

    def _build_user_prompt(self, context: PromptContext, variant: PromptVariant) -> str:
        sections = [
            self._build_task_section(),
            self._build_problem_section(context),
            self._build_io_section(context),
            self._build_buggy_code_section(context),
            self._build_failing_tests_section(context, variant),
            self._build_execution_feedback_section(context, variant),
            self._build_variant_instruction_section(variant),
            self._build_output_schema_section(),
        ]
        return "\n\n".join(section for section in sections if section)

    def _build_task_section(self) -> str:
        return (
            "## Task\n"
            "Diagnose the student's buggy program in a programming education setting. "
            "Identify the likely bug category, localize the bug, explain the misconception, "
            "suggest a repair strategy, and provide patched code."
        )

    def _build_problem_section(self, context: PromptContext) -> str:
        lines = ["## Problem Statement", context.problem_statement.strip()]
        return "\n".join(lines)

    def _build_io_section(self, context: PromptContext) -> str:
        if context.input_output_specification is None:
            return ""

        input_specification = context.input_output_specification.input_specification.strip()
        output_specification = context.input_output_specification.output_specification.strip()
        if not input_specification and not output_specification:
            return ""

        lines = ["## Input/Output Specification"]
        if input_specification:
            lines.extend(["Input:", input_specification])
        if output_specification:
            lines.extend(["Output:", output_specification])
        return "\n".join(lines)

    def _build_buggy_code_section(self, context: PromptContext) -> str:
        header = "## Buggy Student Code"
        if self.options.include_language_tag:
            code_block = f"```{context.programming_language}\n{context.buggy_student_code.rstrip()}\n```"
        else:
            code_block = f"```\n{context.buggy_student_code.rstrip()}\n```"
        return "\n".join([header, code_block])

    def _build_failing_tests_section(self, context: PromptContext, variant: PromptVariant) -> str:
        if variant == "direct_diagnosis" or not context.failing_test_cases:
            return ""
        # EF-no-answer: omit full assertions (they expose expected output); EF-leaky includes them
        if variant == "diagnosis_with_execution_feedback_no_leakage":
            return ""

        formatted_tests = "\n".join(f"- {test_case}" for test_case in context.failing_test_cases)
        return "\n".join(["## Failing Test Cases", formatted_tests])

    def _build_execution_feedback_section(self, context: PromptContext, variant: PromptVariant) -> str:
        if variant == "direct_diagnosis" or context.execution_feedback is None:
            return ""

        feedback = context.execution_feedback
        lines = ["## Execution Feedback"]
        if feedback.feedback_summary.strip():
            lines.extend(["Summary:", feedback.feedback_summary.strip()])
        if feedback.traceback_text.strip():
            if variant == "diagnosis_with_execution_feedback_no_leakage":
                # EF-no-answer: strip exception_message to avoid leaking expected output (e.g. "4 != 5")
                traceback = self._sanitize_traceback_no_leakage(feedback.traceback_text)
                if traceback:
                    lines.extend(["Traceback:", traceback])
            else:
                lines.extend(["Traceback:", feedback.traceback_text.strip()])
        if variant != "diagnosis_with_execution_feedback_no_leakage":
            if feedback.failing_stdout.strip():
                lines.extend(["Captured stdout:", feedback.failing_stdout.strip()])
            if feedback.failing_stderr.strip():
                lines.extend(["Captured stderr:", feedback.failing_stderr.strip()])
        return "\n".join(lines)

    def _sanitize_traceback_no_leakage(self, traceback_text: str) -> str:
        """Keep only test_id and exception_type; strip exception_message (e.g. '4 != 5' reveals expected output)."""
        sanitized = []
        for line in traceback_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: "mbpp_103::test::1: AssertionError: 4 != 5" -> "Test 1: AssertionError"
            parts = line.split(": ", 2)
            if len(parts) >= 2:
                test_id, exc_type = parts[0].strip(), parts[1].strip()
                if "::test::" in test_id:
                    idx = test_id.split("::test::")[-1]
                    sanitized.append(f"Test {idx}: {exc_type}")
                else:
                    sanitized.append(f"{test_id}: {exc_type}")
            else:
                sanitized.append(line)
        return "\n".join(sanitized)

    def _build_variant_instruction_section(self, variant: PromptVariant) -> str:
        if variant == "direct_diagnosis":
            return (
                "## Additional Instruction\n"
                "Base your diagnosis on the problem statement and buggy code only."
            )

        if variant == "diagnosis_with_execution_feedback":
            return (
                "## Additional Instruction\n"
                "Use the failing test cases and execution feedback as evidence when localizing the bug."
            )

        if variant == "diagnosis_with_execution_feedback_no_leakage":
            return (
                "## Additional Instruction\n"
                "Use the execution feedback (failure count and exception types) as evidence when localizing the bug. "
                "Expected outputs are not provided; infer the bug from the failure pattern."
            )

        if variant == "diagnosis_with_self_check":
            return (
                "## Additional Instruction\n"
                "First reason internally about whether the diagnosis matches the provided evidence. "
                "Then return only the final JSON object."
            )

        raise ValueError(f"Unsupported prompt variant: {variant}")

    def _build_output_schema_section(self) -> str:
        lines = [
            "## Output Format",
            "Return exactly one JSON object with the following fields:",
            '- "bug_type": one of off_by_one, wrong_loop_bound, accumulator_init_error, condition_inversion, premature_return, wrong_comparison_operator',
            '- "bug_line": integer or null',
            '- "explanation": string',
            '- "fix_strategy": string',
            '- "patched_code": string',
            '- "confidence": number between 0 and 1',
        ]

        if self.options.require_json_only_response:
            lines.append("Do not include any extra text before or after the JSON object.")

        if self.options.include_output_schema_example:
            lines.extend(
                [
                    "Example shape:",
                    "{",
                    '  "bug_type": "off_by_one",',
                    '  "bug_line": 7,',
                    '  "explanation": "The loop iterates one extra time.",',
                    '  "fix_strategy": "Adjust the loop bound to stop before the invalid index.",',
                    '  "patched_code": "def solve(...):\\n    ...",',
                    '  "confidence": 0.84',
                    "}",
                ]
            )

        return "\n".join(lines)
