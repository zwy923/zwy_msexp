"""Semantic-preserving transformers for buggy Python programs."""

from __future__ import annotations

import ast
import io
import json
import keyword
import tokenize
import warnings
from dataclasses import dataclass

from thesis_exp.schemas.sample import BuggyProgramSample, TransformedSample
from thesis_exp.transforms.base import (
    BaseProgramTransformer,
    TransformationResult,
    TransformationValidationReport,
    validate_behavior_equivalence,
)


@dataclass(slots=True)
class _CodeTransformResult:
    """Result of transforming one code string."""

    transformed_code: str
    changed_lines: list[int]
    transformation_description: str


def _line_starts(source_code: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(source_code):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _position_to_index(source_code: str, line_number: int, column_number: int) -> int:
    starts = _line_starts(source_code)
    return starts[line_number - 1] + column_number


def _span_to_indices(source_code: str, node: ast.AST) -> tuple[int, int]:
    starts = _line_starts(source_code)
    start_index = starts[node.lineno - 1] + node.col_offset
    end_index = starts[node.end_lineno - 1] + node.end_col_offset
    return start_index, end_index


def _changed_lines_from_spans(line_numbers: list[int]) -> list[int]:
    return sorted(set(line_numbers))


def _find_function(
    source_code: str,
    function_name: str | None = None,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    module = ast.parse(source_code)
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if function_name is None or node.name == function_name:
                return node
    raise ValueError("No matching function found.")


def _infer_function_name(source_code: str) -> str | None:
    try:
        return _find_function(source_code).name
    except (SyntaxError, ValueError):
        return None


def _compile_ok(source_code: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")
            compile(source_code, "<transformed_code>", "exec")
    except SyntaxError:
        return False
    return True


def _apply_replacements(
    source_code: str,
    replacements: list[tuple[int, int, str, int]],
) -> tuple[str, list[int]]:
    transformed_code = source_code
    changed_lines: list[int] = []

    for start_index, end_index, replacement, line_number in sorted(replacements, reverse=True):
        transformed_code = transformed_code[:start_index] + replacement + transformed_code[end_index:]
        changed_lines.append(line_number)

    return transformed_code, _changed_lines_from_spans(changed_lines)


def _bug_line_range(sample: BuggyProgramSample) -> set[int]:
    start = sample.bug_injection_record.injection_line_start
    end = sample.bug_injection_record.injection_line_end
    return set(range(start, end + 1))


def _overlaps_bug_lines(start_line: int, end_line: int, bug_lines: set[int]) -> bool:
    return any(line_number in bug_lines for line_number in range(start_line, end_line + 1))


def _build_transformed_sample(
    sample: BuggyProgramSample,
    transformer: BaseProgramTransformer,
    transformed_buggy_code: str,
    transformed_reference_code: str,
    transformed_starter_code: str,
    description: str,
    validation_report: TransformationValidationReport,
) -> TransformedSample:
    notes = {
        "transformation_description": description,
        "validation_summary": validation_report.validation_summary,
        "syntax_valid": validation_report.syntax_valid,
        "behavior_preserved": validation_report.behavior_preserved,
    }
    return TransformedSample(
        transformed_sample_id=f"{sample.sample_id}__{transformer.transformation_name}__{transformer.transformation_version}",
        source_sample_id=sample.sample_id,
        problem_id=sample.problem_id,
        transformation_name=transformer.transformation_name,
        transformation_version=transformer.transformation_version,
        statement_transformed=False,
        code_transformed=True,
        transformed_problem_statement=sample.problem_statement,
        transformed_starter_code=transformed_starter_code,
        transformed_reference_code=transformed_reference_code,
        transformed_buggy_code=transformed_buggy_code,
        transformation_notes=json.dumps(notes, ensure_ascii=False, sort_keys=True),
    )


class _PythonProgramTransformer(BaseProgramTransformer):
    """Shared flow for Python code transformers."""

    allow_bug_line_touch: bool = True

    def transform(
        self,
        sample: BuggyProgramSample,
        validate_behavior: bool = True,
    ) -> TransformationResult | None:
        function_name = _infer_function_name(sample.buggy_code)
        buggy_result = self._transform_code(
            sample.buggy_code,
            function_name,
            _bug_line_range(sample),
            allow_bug_line_touch=self.allow_bug_line_touch,
        )
        if buggy_result is None:
            return None

        reference_result = self._transform_code(sample.reference_code, function_name, set(), allow_bug_line_touch=True)
        starter_result = self._transform_code(sample.starter_code, function_name, set(), allow_bug_line_touch=True)

        transformed_reference_code = reference_result.transformed_code if reference_result else sample.reference_code
        transformed_starter_code = starter_result.transformed_code if starter_result else sample.starter_code

        validation_report = (
            validate_behavior_equivalence(sample.buggy_code, buggy_result.transformed_code, sample.test_cases)
            if validate_behavior
            else TransformationValidationReport(
                syntax_valid=_compile_ok(buggy_result.transformed_code),
                supported_test_case_count=0,
                behavior_preserved=None,
                validation_summary="Behavior validation skipped.",
            )
        )

        transformed_sample = _build_transformed_sample(
            sample=sample,
            transformer=self,
            transformed_buggy_code=buggy_result.transformed_code,
            transformed_reference_code=transformed_reference_code,
            transformed_starter_code=transformed_starter_code,
            description=buggy_result.transformation_description,
            validation_report=validation_report,
        )

        return TransformationResult(
            transformed_sample=transformed_sample,
            transformation_name=self.transformation_name,
            changed_lines=buggy_result.changed_lines,
            transformation_description=buggy_result.transformation_description,
            bug_type_preserved=True,
            behavior_preserved=validation_report.behavior_preserved,
            validation_report=validation_report,
        )

    def _transform_code(
        self,
        source_code: str,
        function_name: str | None,
        bug_lines: set[int],
        allow_bug_line_touch: bool,
    ) -> _CodeTransformResult | None:
        raise NotImplementedError


class VariableRenamingTransformer(_PythonProgramTransformer):
    """Rename one local variable consistently."""

    transformation_name = "variable_renaming"

    def _transform_code(
        self,
        source_code: str,
        function_name: str | None,
        bug_lines: set[int],
        allow_bug_line_touch: bool,
    ) -> _CodeTransformResult | None:
        del bug_lines, allow_bug_line_touch
        try:
            function_node = _find_function(source_code, function_name)
        except (SyntaxError, ValueError):
            return None

        candidate_names: list[str] = []
        existing_names: set[str] = set()

        for node in ast.walk(function_node):
            if isinstance(node, ast.arg):
                existing_names.add(node.arg)
                if node.arg not in {"self", "cls"} and not keyword.iskeyword(node.arg):
                    candidate_names.append(node.arg)
            elif isinstance(node, ast.Name):
                existing_names.add(node.id)

        old_name = next((name for name in candidate_names if not name.startswith("_")), None)
        if old_name is None:
            return None

        suffix_candidates = ["_value", "_item", "_var", "_renamed"]
        new_name = ""
        for suffix in suffix_candidates:
            candidate = f"{old_name}{suffix}"
            if candidate not in existing_names and not keyword.iskeyword(candidate):
                new_name = candidate
                break
        if not new_name:
            return None

        replacements: list[tuple[int, int, str, int]] = []
        for node in ast.walk(function_node):
            if isinstance(node, ast.arg) and node.arg == old_name:
                start_index, end_index = _span_to_indices(source_code, node)
                replacements.append((start_index, end_index, new_name, node.lineno))
            elif isinstance(node, ast.Name) and node.id == old_name:
                start_index, end_index = _span_to_indices(source_code, node)
                replacements.append((start_index, end_index, new_name, node.lineno))

        if not replacements:
            return None

        transformed_code, changed_lines = _apply_replacements(source_code, replacements)
        if not _compile_ok(transformed_code):
            return None

        return _CodeTransformResult(
            transformed_code=transformed_code,
            changed_lines=changed_lines,
            transformation_description=f"Renamed local variable '{old_name}' to '{new_name}'.",
        )


class CommentDocstringPlaceholderTransformer(_PythonProgramTransformer):
    """Rewrite comments and docstrings with placeholders."""

    transformation_name = "comment_docstring_placeholder"

    def _transform_code(
        self,
        source_code: str,
        function_name: str | None,
        bug_lines: set[int],
        allow_bug_line_touch: bool,
    ) -> _CodeTransformResult | None:
        del function_name, bug_lines, allow_bug_line_touch
        replacements: list[tuple[int, int, str, int]] = []

        try:
            module = ast.parse(source_code)
        except SyntaxError:
            return None

        def collect_docstrings(node: ast.AST) -> None:
            body = getattr(node, "body", None)
            if not body:
                return
            first_statement = body[0]
            if (
                isinstance(first_statement, ast.Expr)
                and isinstance(first_statement.value, ast.Constant)
                and isinstance(first_statement.value.value, str)
            ):
                start_index, end_index = _span_to_indices(source_code, first_statement.value)
                replacements.append(
                    (
                        start_index,
                        end_index,
                        '"""Rewritten docstring placeholder."""',
                        first_statement.lineno,
                    )
                )

        collect_docstrings(module)
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                collect_docstrings(node)

        token_stream = tokenize.generate_tokens(io.StringIO(source_code).readline)
        for token in token_stream:
            if token.type != tokenize.COMMENT:
                continue
            start_index = _position_to_index(source_code, token.start[0], token.start[1])
            end_index = _position_to_index(source_code, token.end[0], token.end[1])
            replacements.append((start_index, end_index, "# rewritten comment", token.start[0]))

        if not replacements:
            return None

        transformed_code, changed_lines = _apply_replacements(source_code, replacements)
        if not _compile_ok(transformed_code):
            return None

        return _CodeTransformResult(
            transformed_code=transformed_code,
            changed_lines=changed_lines,
            transformation_description="Rewrote comments and docstrings with placeholders.",
        )


class FormattingNormalizationTransformer(_PythonProgramTransformer):
    """Normalize formatting with a canonical AST unparse."""

    transformation_name = "formatting_normalization"

    def _transform_code(
        self,
        source_code: str,
        function_name: str | None,
        bug_lines: set[int],
        allow_bug_line_touch: bool,
    ) -> _CodeTransformResult | None:
        del function_name, bug_lines, allow_bug_line_touch

        token_stream = tokenize.generate_tokens(io.StringIO(source_code).readline)
        if any(token.type == tokenize.COMMENT for token in token_stream):
            return None

        try:
            transformed_code = ast.unparse(ast.parse(source_code))
        except SyntaxError:
            return None

        if transformed_code == source_code or not _compile_ok(transformed_code):
            return None

        changed_lines = list(range(1, max(len(source_code.splitlines()), len(transformed_code.splitlines())) + 1))
        return _CodeTransformResult(
            transformed_code=transformed_code,
            changed_lines=changed_lines,
            transformation_description="Normalized formatting with a canonical Python representation.",
        )


class EquivalentConstructRewriteTransformer(_PythonProgramTransformer):
    """Apply a conservative equivalent construct rewrite."""

    transformation_name = "equivalent_construct_rewrite"
    allow_bug_line_touch = False

    def _transform_code(
        self,
        source_code: str,
        function_name: str | None,
        bug_lines: set[int],
        allow_bug_line_touch: bool,
    ) -> _CodeTransformResult | None:
        try:
            function_node = _find_function(source_code, function_name)
        except (SyntaxError, ValueError):
            return None

        for node in ast.walk(function_node):
            if not isinstance(node, ast.For):
                continue
            if _overlaps_bug_lines(node.iter.lineno, node.iter.end_lineno, bug_lines) and not allow_bug_line_touch:
                continue
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "iter":
                continue

            replacement_node = ast.Call(func=ast.Name(id="iter", ctx=ast.Load()), args=[node.iter], keywords=[])
            start_index, end_index = _span_to_indices(source_code, node.iter)
            transformed_code = source_code[:start_index] + ast.unparse(replacement_node) + source_code[end_index:]
            if not _compile_ok(transformed_code):
                continue
            return _CodeTransformResult(
                transformed_code=transformed_code,
                changed_lines=_changed_lines_from_spans(list(range(node.iter.lineno, node.iter.end_lineno + 1))),
                transformation_description="Wrapped a loop iterable with iter(...) without changing semantics.",
            )

        body = function_node.body
        for index in range(len(body) - 1):
            current_statement = body[index]
            next_statement = body[index + 1]
            if not (
                isinstance(current_statement, ast.If)
                and len(current_statement.body) == 1
                and not current_statement.orelse
                and isinstance(current_statement.body[0], ast.Return)
                and isinstance(next_statement, ast.Return)
                and isinstance(current_statement.body[0].value, ast.Constant)
                and isinstance(next_statement.value, ast.Constant)
                and current_statement.body[0].value.value in {True, False}
                and next_statement.value.value in {True, False}
            ):
                continue

            if _overlaps_bug_lines(current_statement.lineno, next_statement.end_lineno, bug_lines) and not allow_bug_line_touch:
                continue

            if current_statement.body[0].value.value is True and next_statement.value.value is False:
                replacement = f"{' ' * current_statement.col_offset}return bool({ast.unparse(current_statement.test)})"
                description = "Rewrote a boolean-return branch into return bool(...)."
            elif current_statement.body[0].value.value is False and next_statement.value.value is True:
                replacement = f"{' ' * current_statement.col_offset}return not bool({ast.unparse(current_statement.test)})"
                description = "Rewrote a boolean-return branch into return not bool(...)."
            else:
                continue

            start_index, _ = _span_to_indices(source_code, current_statement)
            _, end_index = _span_to_indices(source_code, next_statement)
            transformed_code = source_code[:start_index] + replacement + source_code[end_index:]
            if not _compile_ok(transformed_code):
                continue
            return _CodeTransformResult(
                transformed_code=transformed_code,
                changed_lines=_changed_lines_from_spans(list(range(current_statement.lineno, next_statement.end_lineno + 1))),
                transformation_description=description,
            )

        return None
