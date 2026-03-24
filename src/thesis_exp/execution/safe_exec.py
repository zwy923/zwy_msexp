"""Restricted builtins + import used for subprocess execution of student/reference code.

Kept in one module so filtering, quality gates, and repair evaluation share the same sandbox.
"""

from __future__ import annotations

from typing import Any

# Align with conservative allowlist in datasets/mbpp_filter.py (static import checks).
ALLOWED_IMPORT_ROOT_MODULES: frozenset[str] = frozenset(
    {
        "bisect",
        "collections",
        "functools",
        "heapq",
        "itertools",
        "math",
        "operator",
        "re",
        "statistics",
        "string",
        "typing",
    }
)


def restricted_import(
    name: str,
    globals_: Any = None,
    locals_: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    """``__import__`` hook that only allows a small stdlib subset."""

    del globals_, locals_, fromlist, level
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORT_ROOT_MODULES:
        raise ImportError(f"Import of module '{root}' is not allowed in the execution sandbox.")
    return __import__(name)


def build_safe_exec_builtins() -> dict[str, Any]:
    """Return the ``__builtins__`` mapping for safe exec of MBPP-style code."""

    safe: dict[str, Any] = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "frozenset": frozenset,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "round": round,
        "reversed": reversed,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "AssertionError": AssertionError,
        "Exception": Exception,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "__import__": restricted_import,
    }
    return safe
