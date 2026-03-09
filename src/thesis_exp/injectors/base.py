"""Base interfaces for bug injectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from thesis_exp.common.types import BugType


@dataclass(slots=True)
class InjectionResult:
    """Result returned by a successful bug injection."""

    bug_type: BugType | str
    modified_code: str
    changed_lines: list[int]
    transformation_description: str
    injector_name: str


class BaseBugInjector(ABC):
    """Inject one localized bug into correct source code."""

    bug_type: BugType | str

    @abstractmethod
    def inject(
        self,
        source_code: str,
        function_name: str | None = None,
    ) -> InjectionResult | None:
        """Return one injected variant or None if not applicable."""


# Backward-compatible alias for the initial scaffold.
BaseInjector = BaseBugInjector
