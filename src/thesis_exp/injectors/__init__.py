"""Exports for bug injector components."""

from thesis_exp.injectors.base import BaseBugInjector, BaseInjector, InjectionResult
from thesis_exp.injectors.python import (
    AccumulatorInitErrorInjector,
    ConditionInversionInjector,
    OffByOneInjector,
    PrematureReturnInjector,
    WrongComparisonOperatorInjector,
    WrongLoopBoundInjector,
    find_top_level_function,
    function_has_accumulator_init_site,
    function_has_condition_inversion_site,
    function_has_off_by_one_site,
    function_has_premature_return_site,
    function_has_wrong_comparison_operator_site,
    function_has_wrong_loop_bound_site,
)
from thesis_exp.injectors.registry import create_injector, list_injector_types

__all__ = [
    "AccumulatorInitErrorInjector",
    "find_top_level_function",
    "function_has_accumulator_init_site",
    "function_has_condition_inversion_site",
    "function_has_off_by_one_site",
    "function_has_premature_return_site",
    "function_has_wrong_comparison_operator_site",
    "function_has_wrong_loop_bound_site",
    "BaseBugInjector",
    "BaseInjector",
    "ConditionInversionInjector",
    "create_injector",
    "InjectionResult",
    "list_injector_types",
    "OffByOneInjector",
    "PrematureReturnInjector",
    "WrongComparisonOperatorInjector",
    "WrongLoopBoundInjector",
]
