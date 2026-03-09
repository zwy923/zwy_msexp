"""Loader for sanitized MBPP-style datasets."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from thesis_exp.datasets.loader import (
    DatasetProblemRecord,
    DatasetProtocolError,
    dataset_record_to_problem,
)
from thesis_exp.schemas.sample import ProgrammingProblem


def load_sanitized_mbpp_raw_payloads(dataset_path: str) -> list[dict[str, Any]]:
    """Load raw sanitized MBPP payloads from a JSONL-style file."""

    path = Path(dataset_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payloads: list[dict[str, Any]] = []

    for line_index, line in enumerate(lines, start=1):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetProtocolError(
                f"Could not parse sanitized MBPP record on line {line_index}: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})."
            ) from exc

        if isinstance(parsed, dict):
            payloads.append(parsed)
            continue

        if isinstance(parsed, list) and len(lines) == 1:
            if not all(isinstance(item, dict) for item in parsed):
                raise DatasetProtocolError(
                    "Single-line sanitized MBPP array must contain only JSON objects."
                )
            payloads.extend(parsed)
            continue

        raise DatasetProtocolError(
            f"Sanitized MBPP line {line_index} must decode to a JSON object."
        )

    if not payloads:
        raise DatasetProtocolError("Sanitized MBPP dataset is empty.")
    return payloads


def _infer_entry_point(reference_code: str, task_id: int | str) -> str:
    """Infer the entry-point function from the reference code."""

    try:
        module = ast.parse(reference_code)
    except SyntaxError as exc:
        raise DatasetProtocolError(
            f"Could not parse reference code for task_id={task_id}: {exc.msg}."
        ) from exc

    function_names = [
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not function_names:
        raise DatasetProtocolError(
            f"Could not infer entry_point for task_id={task_id}: no top-level function was found."
        )
    return function_names[0]


def _normalize_test_list(
    task_id: int | str,
    test_imports: Any,
    test_list: Any,
) -> list[str]:
    """Normalize MBPP test imports and assertion list into executable tests."""

    if not isinstance(test_list, list) or not test_list:
        raise DatasetProtocolError(
            f"Field 'test_list' is required and must be a non-empty list for task_id={task_id}."
        )

    import_lines: list[str] = []
    if test_imports is None:
        import_lines = []
    elif isinstance(test_imports, list):
        if not all(isinstance(item, str) for item in test_imports):
            raise DatasetProtocolError(
                f"Field 'test_imports' must contain only strings for task_id={task_id}."
            )
        import_lines = [item for item in test_imports if item.strip()]
    else:
        raise DatasetProtocolError(
            f"Field 'test_imports' must be a list when provided for task_id={task_id}."
        )

    normalized_tests: list[str] = []
    prefix = "\n".join(import_lines)
    for index, test_code in enumerate(test_list, start=1):
        if not isinstance(test_code, str) or not test_code.strip():
            raise DatasetProtocolError(
                f"Test case {index} in task_id={task_id} must be a non-empty string."
            )
        if prefix:
            normalized_tests.append(f"{prefix}\n{test_code}")
        else:
            normalized_tests.append(test_code)
    return normalized_tests


def validate_sanitized_mbpp_record(record: dict[str, Any]) -> DatasetProblemRecord:
    """Validate and normalize one sanitized MBPP record."""

    if not isinstance(record, dict):
        raise DatasetProtocolError("Each sanitized MBPP record must be a JSON object.")

    task_id = record.get("task_id")
    if not isinstance(task_id, (int, str)) or str(task_id).strip() == "":
        raise DatasetProtocolError("Field 'task_id' is required and must be an integer or non-empty string.")

    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise DatasetProtocolError(f"Field 'prompt' is required and must be a non-empty string for task_id={task_id}.")

    reference_code = record.get("code")
    if not isinstance(reference_code, str) or not reference_code.strip():
        raise DatasetProtocolError(f"Field 'code' is required and must be a non-empty string for task_id={task_id}.")

    entry_point = record.get("entry_point")
    if isinstance(entry_point, str) and entry_point.strip():
        normalized_entry_point = entry_point.strip()
    else:
        normalized_entry_point = _infer_entry_point(reference_code, task_id)

    normalized_tests = _normalize_test_list(
        task_id,
        record.get("test_imports", []),
        record.get("test_list"),
    )

    return DatasetProblemRecord(
        problem_id=f"mbpp_{task_id}",
        prompt=prompt.strip(),
        entry_point=normalized_entry_point,
        reference_code=reference_code,
        tests=normalized_tests,
        programming_language="python",
        starter_code="",
        problem_title=f"MBPP {task_id}",
        difficulty=str(record.get("difficulty", "")) if record.get("difficulty") is not None else "",
        topic=str(record.get("topic", "")) if record.get("topic") is not None else "",
        source_dataset_name="sanitized_mbpp",
        source_split_name=str(record.get("split", "")) if record.get("split") is not None else "",
    )


def load_sanitized_mbpp_records(dataset_path: str) -> list[DatasetProblemRecord]:
    """Load sanitized MBPP records from a JSONL-style file."""

    payloads = load_sanitized_mbpp_raw_payloads(dataset_path)
    return [validate_sanitized_mbpp_record(payload) for payload in payloads]


def load_sanitized_mbpp_dataset(dataset_path: str) -> list[ProgrammingProblem]:
    """Load sanitized MBPP records into internal problem schemas."""

    return [
        dataset_record_to_problem(record)
        for record in load_sanitized_mbpp_records(dataset_path)
    ]
