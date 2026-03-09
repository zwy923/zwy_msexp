"""Registry and factory helpers for bug injectors."""

from thesis_exp.injectors.base import BaseBugInjector
from thesis_exp.injectors.python import (
    AccumulatorInitErrorInjector,
    ConditionInversionInjector,
    OffByOneInjector,
    PrematureReturnInjector,
    WrongComparisonOperatorInjector,
    WrongLoopBoundInjector,
)

INJECTOR_REGISTRY: dict[str, type[BaseBugInjector]] = {
    "off_by_one": OffByOneInjector,
    "wrong_loop_bound": WrongLoopBoundInjector,
    "accumulator_init_error": AccumulatorInitErrorInjector,
    "condition_inversion": ConditionInversionInjector,
    "premature_return": PrematureReturnInjector,
    "wrong_comparison_operator": WrongComparisonOperatorInjector,
}


def create_injector(bug_type: str) -> BaseBugInjector:
    """Create an injector instance for a bug type."""

    injector_class = INJECTOR_REGISTRY.get(bug_type)
    if injector_class is None:
        raise KeyError(f"Unknown injector bug type: {bug_type}")
    return injector_class()


def list_injector_types() -> list[str]:
    """Return registered injector names."""

    return sorted(INJECTOR_REGISTRY)
