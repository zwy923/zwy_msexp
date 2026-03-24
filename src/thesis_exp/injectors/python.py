"""Python bug injectors based on localized AST-guided rewrites."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from thesis_exp.common.types import BugType
from thesis_exp.injectors.base import BaseBugInjector, InjectionResult


@dataclass(slots=True)
class _Edit:
    """Represents one localized source edit."""

    start_index: int
    end_index: int
    replacement: str
    changed_lines: list[int]
    transformation_description: str


def _line_starts(source_code: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(source_code):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _span_to_indices(source_code: str, node: ast.AST) -> tuple[int, int]:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        raise ValueError("AST node is missing location information.")

    starts = _line_starts(source_code)
    start_index = starts[node.lineno - 1] + node.col_offset
    end_index = starts[node.end_lineno - 1] + node.end_col_offset
    return start_index, end_index


def _changed_lines(node: ast.AST) -> list[int]:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return []
    return list(range(node.lineno, node.end_lineno + 1))

def _parse_function(
    source_code: str,
    function_name: str | None,
) -> tuple[ast.Module, ast.FunctionDef | ast.AsyncFunctionDef]:
    module = ast.parse(source_code)
    target_function: ast.FunctionDef | ast.AsyncFunctionDef | None = None

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if function_name is None or node.name == function_name:
                target_function = node
                break

    if target_function is None:
        raise ValueError("No target function found in source code.")

    return module, target_function

def _replace_with_edit(source_code: str, edit: _Edit) -> str:
    return source_code[:edit.start_index] + edit.replacement + source_code[edit.end_index:]


def _accumulator_name_from_assign(node: ast.Assign | ast.AnnAssign) -> str | None:
    target: ast.expr | None
    if isinstance(node, ast.Assign):
        target = node.targets[0] if len(node.targets) == 1 else None
    else:
        target = node.target

    if isinstance(target, ast.Name):
        return target.id
    return None


def _looks_like_accumulator(name: str) -> bool:
    return any(token in name.lower() for token in ("sum", "total", "count", "acc", "result"))


def find_top_level_function(
    module: ast.Module,
    function_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Return the first top-level function/async def with the given name (same resolution as injectors)."""

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    return None


def function_has_premature_return_site(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """True iff PrematureReturnInjector can find a non-return loop statement with a resolvable return_name.

    Mirrors PrematureReturnInjector._build_edit so dataset filtering matches injector applicability.
    """

    accumulator_name: str | None = None
    for statement in function_node.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            candidate = _accumulator_name_from_assign(statement)
            if candidate and _looks_like_accumulator(candidate):
                accumulator_name = candidate
                break

    for node in ast.walk(function_node):
        if not isinstance(node, (ast.For, ast.While)) or not node.body:
            continue

        for statement in node.body:
            if isinstance(statement, ast.Return):
                continue

            if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                return_name = statement.target.id
            elif isinstance(statement, ast.Assign):
                return_name = _accumulator_name_from_assign(statement) or accumulator_name
            else:
                return_name = accumulator_name

            if return_name is None:
                continue

            return True

    return False


# Operator types WrongComparisonOperatorInjector can rewrite (must match _operator_map keys).
_WRONG_COMPARISON_OPERATOR_TYPES: frozenset[type] = frozenset(
    (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq),
)


def function_has_off_by_one_site(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True iff OffByOneInjector can rewrite a range() stop argument."""

    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "range":
            continue
        if len(node.args) not in (1, 2, 3):
            continue
        return True
    return False


def function_has_wrong_loop_bound_site(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True iff WrongLoopBoundInjector can edit a range stop (2/3-arg) or a while Compare bound."""

    for node in ast.walk(function_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            if len(node.args) in (2, 3):
                return True
        if isinstance(node, ast.While) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
            return True
    return False


def function_has_accumulator_init_site(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True iff AccumulatorInitErrorInjector can flip a 0/1 init (top-level body only, same as injector)."""

    for node in function_node.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            name = _accumulator_name_from_assign(node)
            if name is None or not _looks_like_accumulator(name):
                continue
            if node.value.value in (0, 1):
                return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            name = _accumulator_name_from_assign(node)
            if name is None or not _looks_like_accumulator(name):
                continue
            if node.value.value in (0, 1):
                return True
    return False


def function_has_condition_inversion_site(function_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True iff ConditionInversionInjector finds an If or While (first target in walk order)."""

    for node in ast.walk(function_node):
        if isinstance(node, (ast.If, ast.While)):
            return True
    return False


def function_has_wrong_comparison_operator_site(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """True iff WrongComparisonOperatorInjector finds a single-op Compare with a swappable operator."""

    for node in ast.walk(function_node):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if type(node.ops[0]) in _WRONG_COMPARISON_OPERATOR_TYPES:
            return True
    return False


class _PythonAstInjector(BaseBugInjector):
    """Shared AST-based injector logic."""

    def inject(
        self,
        source_code: str,
        function_name: str | None = None,
    ) -> InjectionResult | None:
        try:
            _, function_node = _parse_function(source_code, function_name)
        except (SyntaxError, ValueError):
            return None

        edit = self._build_edit(source_code, function_node)
        if edit is None:
            return None

        modified_code = _replace_with_edit(source_code, edit)
        try:
            ast.parse(modified_code)
        except SyntaxError:
            return None

        return InjectionResult(
            bug_type=self.bug_type,
            modified_code=modified_code,
            changed_lines=edit.changed_lines,
            transformation_description=edit.transformation_description,
            injector_name=self.__class__.__name__,
        )

    def _build_edit(
        self,
        source_code: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> _Edit | None:
        raise NotImplementedError


class OffByOneInjector(_PythonAstInjector):
    """Inject a loop-boundary bug in a ``range`` stop (expand stop by one)."""

    bug_type: BugType = "loop_boundary_error"

    def _build_edit(
        self,
        source_code: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> _Edit | None:
        for node in ast.walk(function_node):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "range":
                continue
            if len(node.args) not in (1, 2, 3):
                continue

            stop_arg_index = 0 if len(node.args) == 1 else 1
            stop_arg = node.args[stop_arg_index]
            new_stop = ast.copy_location(
                ast.BinOp(left=stop_arg, op=ast.Add(), right=ast.Constant(value=1)),
                stop_arg,
            )
            start_index, end_index = _span_to_indices(source_code, stop_arg)
            return _Edit(
                start_index=start_index,
                end_index=end_index,
                replacement=ast.unparse(new_stop),
                changed_lines=_changed_lines(stop_arg),
                transformation_description="Expanded the loop stop bound by one.",
            )
        return None


class WrongLoopBoundInjector(_PythonAstInjector):
    """Inject a loop-boundary bug (shrink ``range`` stop or ``while`` bound by one)."""

    bug_type: BugType = "loop_boundary_error"

    def _build_edit(
        self,
        source_code: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> _Edit | None:
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
                if len(node.args) in (2, 3):
                    stop_arg = node.args[1]
                    new_stop = ast.copy_location(
                        ast.BinOp(left=stop_arg, op=ast.Sub(), right=ast.Constant(value=1)),
                        stop_arg,
                    )
                    start_index, end_index = _span_to_indices(source_code, stop_arg)
                    return _Edit(
                        start_index=start_index,
                        end_index=end_index,
                        replacement=ast.unparse(new_stop),
                        changed_lines=_changed_lines(stop_arg),
                        transformation_description="Shifted the loop upper bound one step earlier.",
                    )

            if isinstance(node, ast.While) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
                comparator = node.test.comparators[0]
                new_comparator = ast.copy_location(
                    ast.BinOp(left=comparator, op=ast.Sub(), right=ast.Constant(value=1)),
                    comparator,
                )
                start_index, end_index = _span_to_indices(source_code, comparator)
                return _Edit(
                    start_index=start_index,
                    end_index=end_index,
                    replacement=ast.unparse(new_comparator),
                    changed_lines=_changed_lines(comparator),
                    transformation_description="Reduced the while-loop bound by one.",
                )

        return None


class AccumulatorInitErrorInjector(_PythonAstInjector):
    """Inject a wrong accumulator initialization."""

    bug_type: BugType = "accumulator_init_error"

    def _build_edit(
        self,
        source_code: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> _Edit | None:
        for node in function_node.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                accumulator_name = _accumulator_name_from_assign(node)
                if accumulator_name is None or not _looks_like_accumulator(accumulator_name):
                    continue
                if node.value.value not in (0, 1):
                    continue

                new_value = ast.Constant(value=1 if node.value.value == 0 else 0)
                start_index, end_index = _span_to_indices(source_code, node.value)
                return _Edit(
                    start_index=start_index,
                    end_index=end_index,
                    replacement=ast.unparse(new_value),
                    changed_lines=_changed_lines(node.value),
                    transformation_description="Changed the accumulator initialization constant.",
                )

            if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
                accumulator_name = _accumulator_name_from_assign(node)
                if accumulator_name is None or not _looks_like_accumulator(accumulator_name):
                    continue
                if node.value.value not in (0, 1):
                    continue

                new_value = ast.Constant(value=1 if node.value.value == 0 else 0)
                start_index, end_index = _span_to_indices(source_code, node.value)
                return _Edit(
                    start_index=start_index,
                    end_index=end_index,
                    replacement=ast.unparse(new_value),
                    changed_lines=_changed_lines(node.value),
                    transformation_description="Changed the accumulator initialization constant.",
                )

        return None


class ConditionInversionInjector(_PythonAstInjector):
    """Inject a condition inversion bug."""

    bug_type: BugType = "conditional_logic_error"

    def _build_edit(
        self,
        source_code: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> _Edit | None:
        for node in ast.walk(function_node):
            if not isinstance(node, (ast.If, ast.While)):
                continue

            test = node.test
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                replacement_node = test.operand
            else:
                replacement_node = ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=test), test)

            start_index, end_index = _span_to_indices(source_code, test)
            return _Edit(
                start_index=start_index,
                end_index=end_index,
                replacement=ast.unparse(replacement_node),
                changed_lines=_changed_lines(test),
                transformation_description="Inverted a branch condition.",
            )

        return None


class PrematureReturnInjector(_PythonAstInjector):
    """Inject an early return inside a loop."""

    bug_type: BugType = "premature_return"

    def _build_edit(
        self,
        source_code: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> _Edit | None:
        accumulator_name: str | None = None
        for statement in function_node.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                candidate = _accumulator_name_from_assign(statement)
                if candidate and _looks_like_accumulator(candidate):
                    accumulator_name = candidate
                    break

        for node in ast.walk(function_node):
            if not isinstance(node, (ast.For, ast.While)) or not node.body:
                continue

            for statement in node.body:
                if isinstance(statement, ast.Return):
                    continue

                if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
                    return_name = statement.target.id
                elif isinstance(statement, ast.Assign):
                    return_name = _accumulator_name_from_assign(statement) or accumulator_name
                else:
                    return_name = accumulator_name

                if return_name is None:
                    continue

                return_source = f"return {return_name}"
                start_index, end_index = _span_to_indices(source_code, statement)
                return _Edit(
                    start_index=start_index,
                    end_index=end_index,
                    replacement=(" " * statement.col_offset) + return_source,
                    changed_lines=_changed_lines(statement),
                    transformation_description="Returned from inside the loop before completion.",
                )

        return None


class WrongComparisonOperatorInjector(_PythonAstInjector):
    """Inject a wrong comparison operator bug."""

    bug_type: BugType = "conditional_logic_error"

    _operator_map: dict[type[ast.cmpop], ast.cmpop] = {
        ast.Lt: ast.LtE(),
        ast.LtE: ast.Lt(),
        ast.Gt: ast.GtE(),
        ast.GtE: ast.Gt(),
        ast.Eq: ast.NotEq(),
        ast.NotEq: ast.Eq(),
    }

    def _build_edit(
        self,
        source_code: str,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> _Edit | None:
        for node in ast.walk(function_node):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue

            operator_type = type(node.ops[0])
            if operator_type not in self._operator_map:
                continue

            replacement_node = ast.Compare(
                left=node.left,
                ops=[self._operator_map[operator_type]],
                comparators=node.comparators,
            )
            start_index, end_index = _span_to_indices(source_code, node)
            return _Edit(
                start_index=start_index,
                end_index=end_index,
                replacement=ast.unparse(replacement_node),
                changed_lines=_changed_lines(node),
                transformation_description="Replaced a comparison operator with a nearby incorrect one.",
            )

        return None
