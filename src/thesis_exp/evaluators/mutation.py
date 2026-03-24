"""Mutation-based test adequacy for repair evaluation.

If tests pass on both correct code and simple wrong variants (mutants),
the test suite is weak and should not count toward strict ``repair_success`` (mutation gate).

Operators (first-order AST edits): Compare (Eq/ineq/order, In/NotIn), BinOp,
BoolOp (and/or), UnaryOp (strip ``not``), Call (``any``/``all`` swap).
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from thesis_exp.schemas.sample import TestCase

from .diagnosis_evaluator import EvaluationConfig, execute_patched_code_safely


@dataclass(slots=True)
class MutationAdequacyResult:
    """Result of mutation-based adequacy check."""

    adequate: bool
    total_mutants: int
    killed_count: int
    min_required_killed: int
    mutant_sources: list[str] = field(default_factory=list)
    #: Why ``total_mutants == 0`` when no mutants were evaluated, or None if mutants were generated.
    empty_reason: str | None = None


# Compare ops: (original, [alternatives])
_COMPARE_ALTERNATIVES: dict[type, list[type]] = {
    ast.Eq: [ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
    ast.NotEq: [ast.Eq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
    ast.Lt: [ast.LtE, ast.Eq, ast.NotEq, ast.Gt, ast.GtE],
    ast.LtE: [ast.Lt, ast.Eq, ast.NotEq, ast.Gt, ast.GtE],
    ast.Gt: [ast.GtE, ast.Eq, ast.NotEq, ast.Lt, ast.LtE],
    ast.GtE: [ast.Gt, ast.Eq, ast.NotEq, ast.Lt, ast.LtE],
}

# BinOp: (original, [alternatives]) - arithmetic only
_BINOP_ALTERNATIVES: dict[type, list[type]] = {
    ast.Add: [ast.Sub, ast.Mult],
    ast.Sub: [ast.Add, ast.Mult],
    ast.Mult: [ast.Add, ast.Sub, ast.FloorDiv],
    ast.FloorDiv: [ast.Mult, ast.Div],
    ast.Div: [ast.Mult, ast.FloorDiv],
}

# BoolOp
_BOOLOP_ALTERNATIVES: dict[type, type] = {
    ast.And: ast.Or,
    ast.Or: ast.And,
}


def _mutate_compare(node: ast.Compare, op_index: int) -> list[ast.AST]:
    """Yield mutated Compare nodes (one alternative per yield)."""
    ops = node.ops
    if op_index >= len(ops):
        return []
    orig_op = type(ops[op_index])
    # Membership: In <-> NotIn
    if orig_op is ast.In or orig_op is ast.NotIn:
        swap_cls = ast.NotIn if orig_op is ast.In else ast.In
        new_ops = list(ops)
        new_ops[op_index] = swap_cls()
        return [
            ast.Compare(
                left=node.left,
                ops=new_ops,
                comparators=node.comparators,
            )
        ]
    alts = _COMPARE_ALTERNATIVES.get(orig_op)
    if not alts:
        return []
    result: list[ast.AST] = []
    for alt_cls in alts[:2]:  # Limit to 2 alternatives per op to control mutant count
        new_ops = list(ops)
        new_ops[op_index] = alt_cls()
        new_node = ast.Compare(
            left=node.left,
            ops=new_ops,
            comparators=node.comparators,
        )
        result.append(new_node)
    return result


def _mutate_binop(node: ast.BinOp) -> list[ast.AST]:
    """Yield mutated BinOp nodes."""
    orig_op = type(node.op)
    alts = _BINOP_ALTERNATIVES.get(orig_op)
    if not alts:
        return []
    return [
        ast.BinOp(left=node.left, op=alt_cls(), right=node.right)
        for alt_cls in alts[:2]
    ]


def _mutate_boolop(node: ast.BoolOp) -> list[ast.AST]:
    """Yield mutated BoolOp (and <-> or)."""
    alt_cls = _BOOLOP_ALTERNATIVES.get(type(node.op))
    if not alt_cls:
        return []
    return [ast.BoolOp(op=alt_cls(), values=node.values)]


def _mutate_unaryop_not(node: ast.UnaryOp) -> list[ast.AST]:
    """``not x`` -> ``x`` (logical negation removal)."""
    if not isinstance(node.op, ast.Not):
        return []
    return [copy.deepcopy(node.operand)]


def _mutate_any_all_call(node: ast.Call) -> list[ast.AST]:
    """``any(iter)`` <-> ``all(iter)`` (built-in quantifiers)."""
    func = node.func
    if not isinstance(func, ast.Name) or func.id not in ("any", "all"):
        return []
    if not node.args:
        return []
    new_id = "all" if func.id == "any" else "any"
    new_func = ast.Name(id=new_id, ctx=ast.Load())
    return [
        ast.Call(
            func=new_func,
            args=[copy.deepcopy(a) for a in node.args],
            keywords=[copy.deepcopy(kw) for kw in node.keywords],
        )
    ]


class _SingleReplacementTransformer(ast.NodeTransformer):
    """Replace exactly one node matching target_dump with replacement."""

    def __init__(self, target_dump: str, replacement: ast.AST) -> None:
        self.target_dump = target_dump
        self.replacement = replacement
        self.replaced = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if not self.replaced and ast.dump(node) == self.target_dump:
            self.replaced = True
            return copy.deepcopy(self.replacement)
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if not self.replaced and ast.dump(node) == self.target_dump:
            self.replaced = True
            return copy.deepcopy(self.replacement)
        return self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        if not self.replaced and ast.dump(node) == self.target_dump:
            self.replaced = True
            return copy.deepcopy(self.replacement)
        return self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        if not self.replaced and ast.dump(node) == self.target_dump:
            self.replaced = True
            return copy.deepcopy(self.replacement)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not self.replaced and ast.dump(node) == self.target_dump:
            self.replaced = True
            return copy.deepcopy(self.replacement)
        return self.generic_visit(node)


def _collect_mutants_simple(source_code: str, max_mutants: int) -> list[str]:
    """Generate up to max_mutants mutated source strings using AST."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    mutants: list[str] = []
    mutation_points: list[tuple[str, ast.AST]] = []  # (target_dump, replacement_node)

    for node in ast.walk(tree):
        if len(mutation_points) >= max_mutants * 3:
            break
        if isinstance(node, ast.Compare) and node.ops:
            for i in range(len(node.ops)):
                for mut_node in _mutate_compare(node, i):
                    mutation_points.append((ast.dump(node), mut_node))
                    break
        elif isinstance(node, ast.BinOp):
            for mut_node in _mutate_binop(node):
                mutation_points.append((ast.dump(node), mut_node))
                break
        elif isinstance(node, ast.BoolOp):
            for mut_node in _mutate_boolop(node):
                mutation_points.append((ast.dump(node), mut_node))
                break
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            for mut_node in _mutate_unaryop_not(node):
                mutation_points.append((ast.dump(node), mut_node))
                break
        elif isinstance(node, ast.Call):
            for mut_node in _mutate_any_all_call(node):
                mutation_points.append((ast.dump(node), mut_node))
                break

    seen: set[str] = set()
    for target_dump, mut_node in mutation_points:
        if len(mutants) >= max_mutants:
            break
        try:
            transformer = _SingleReplacementTransformer(target_dump, mut_node)
            new_tree = transformer.visit(copy.deepcopy(tree))
            if not transformer.replaced:
                continue
            ast.fix_missing_locations(new_tree)
            mutated_source = ast.unparse(new_tree)
            if mutated_source != source_code and mutated_source not in seen:
                seen.add(mutated_source)
                mutants.append(mutated_source)
        except Exception:
            pass

    return mutants[:max_mutants]


def _collect_mutants(source_code: str, max_mutants: int) -> list[str]:
    """Generate up to max_mutants mutated source strings."""
    return _collect_mutants_simple(source_code, max_mutants)


def check_mutation_adequacy(
    reference_code: str,
    test_cases: list[TestCase],
    config: EvaluationConfig,
    max_mutants: int = 8,
    min_required_killed: int = 1,
) -> MutationAdequacyResult:
    """Check if the test suite can kill at least one mutant of the reference code.

    Args:
        reference_code: Correct implementation (gold).
        test_cases: Test cases to run.
        config: Evaluation config for test selection and timeout.
        max_mutants: Maximum number of mutants to generate and run.
        min_required_killed: Minimum mutants that must be killed for adequacy.

    Returns:
        MutationAdequacyResult with adequate=True iff at least min_required_killed mutants are killed.
    """
    if not reference_code or not reference_code.strip():
        return MutationAdequacyResult(
            adequate=False,
            total_mutants=0,
            killed_count=0,
            min_required_killed=min_required_killed,
            empty_reason="empty_reference_code",
        )

    # First verify reference passes all tests
    ref_result = execute_patched_code_safely(reference_code, test_cases, config)
    if (
        not ref_result.syntax_valid
        or ref_result.timed_out
        or ref_result.passed_test_count != ref_result.total_test_count
    ):
        return MutationAdequacyResult(
            adequate=False,
            total_mutants=0,
            killed_count=0,
            min_required_killed=min_required_killed,
            mutant_sources=["reference_failed_tests"],
            empty_reason="reference_failed_tests",
        )

    mutant_sources = _collect_mutants(reference_code, max_mutants)
    if not mutant_sources:
        return MutationAdequacyResult(
            adequate=False,
            total_mutants=0,
            killed_count=0,
            min_required_killed=min_required_killed,
            empty_reason="no_mutants_generated",
        )

    killed_count = 0
    for mut_src in mutant_sources:
        mut_result = execute_patched_code_safely(mut_src, test_cases, config)
        killed = (
            mut_result.syntax_valid
            and not mut_result.timed_out
            and mut_result.passed_test_count < mut_result.total_test_count
        )
        if killed:
            killed_count += 1
            if killed_count >= min_required_killed:
                break

    return MutationAdequacyResult(
        adequate=killed_count >= min_required_killed,
        total_mutants=len(mutant_sources),
        killed_count=killed_count,
        min_required_killed=min_required_killed,
        mutant_sources=mutant_sources[:3],
        empty_reason=None,
    )
