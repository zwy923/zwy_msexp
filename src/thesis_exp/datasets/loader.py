"""Dataset loader and validation for programming-problem corpora."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from thesis_exp.schemas.sample import ProgrammingProblem, ReferenceSolution, TestCase


class DatasetProtocolError(ValueError):
    """Raised when a dataset record violates the required protocol."""


@dataclass(slots=True)
class DatasetProblemRecord:
    """Canonical dataset record for one programming problem."""

    problem_id: str
    prompt: str
    entry_point: str
    reference_code: str
    tests: list[str | dict[str, Any]]
    programming_language: str = "python"
    starter_code: str = ""
    problem_title: str = ""
    difficulty: str = ""
    topic: str = ""
    source_dataset_name: str = ""
    source_split_name: str = ""


def _require_non_empty_string(record: dict[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise DatasetProtocolError(f"Field '{field_name}' is required and must be a non-empty string.")
    return value.strip()


def _optional_string(record: dict[str, Any], field_name: str) -> str:
    value = record.get(field_name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DatasetProtocolError(f"Field '{field_name}' must be a string when provided.")
    return value


def _validate_entry_point(entry_point: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry_point):
        raise DatasetProtocolError(
            "Field 'entry_point' must be a valid Python identifier naming the target function."
        )
    return entry_point


def _normalize_tests(problem_id: str, tests: Any) -> list[TestCase]:
    if not isinstance(tests, list) or not tests:
        raise DatasetProtocolError("Field 'tests' is required and must be a non-empty list.")

    normalized_tests: list[TestCase] = []
    for index, item in enumerate(tests, start=1):
        test_case_id = f"{problem_id}::test::{index}"

        if isinstance(item, str):
            if not item.strip():
                raise DatasetProtocolError(f"Test case {index} for problem '{problem_id}' must not be empty.")
            normalized_tests.append(
                TestCase(
                    test_case_id=test_case_id,
                    problem_id=problem_id,
                    test_case_type="unit_assertion",
                    test_code=item,
                )
            )
            continue

        if isinstance(item, dict):
            payload = dict(item)
            payload.setdefault("test_case_id", test_case_id)
            payload.setdefault("problem_id", problem_id)
            payload.setdefault("test_case_type", "unit_assertion")

            test_code = payload.get("test_code")
            if not isinstance(test_code, str) or not test_code.strip():
                raise DatasetProtocolError(
                    f"Structured test case {index} for problem '{problem_id}' must include a non-empty 'test_code'."
                )

            normalized_tests.append(TestCase.from_dict(payload))
            continue

        raise DatasetProtocolError(
            f"Test case {index} for problem '{problem_id}' must be either a string or an object."
        )

    return normalized_tests


def validate_dataset_problem_record(record: dict[str, Any]) -> DatasetProblemRecord:
    """Validate and normalize one dataset record."""

    if not isinstance(record, dict):
        raise DatasetProtocolError("Each dataset record must be a JSON object.")

    problem_id = _require_non_empty_string(record, "problem_id")
    prompt = _require_non_empty_string(record, "prompt")
    entry_point = _validate_entry_point(_require_non_empty_string(record, "entry_point"))
    reference_code = _require_non_empty_string(record, "reference_code")
    tests = record.get("tests")
    normalized_tests = _normalize_tests(problem_id, tests)

    dataset_record = DatasetProblemRecord(
        problem_id=problem_id,
        prompt=prompt,
        entry_point=entry_point,
        reference_code=reference_code,
        tests=[
            test_case.to_dict()
            for test_case in normalized_tests
        ],
        programming_language=_optional_string(record, "programming_language") or "python",
        starter_code=_optional_string(record, "starter_code"),
        problem_title=_optional_string(record, "problem_title") or problem_id,
        difficulty=_optional_string(record, "difficulty"),
        topic=_optional_string(record, "topic"),
        source_dataset_name=_optional_string(record, "source_dataset_name"),
        source_split_name=_optional_string(record, "source_split_name"),
    )
    return dataset_record


def dataset_record_to_problem(record: DatasetProblemRecord) -> ProgrammingProblem:
    """Convert a validated dataset record into the internal schema."""

    normalized_test_cases: list[TestCase] = []
    for index, item in enumerate(record.tests, start=1):
        if isinstance(item, dict):
            normalized_test_cases.append(TestCase.from_dict(item))
            continue
        if isinstance(item, str):
            normalized_test_cases.append(
                TestCase(
                    test_case_id=f"{record.problem_id}::test::{index}",
                    problem_id=record.problem_id,
                    test_case_type="unit_assertion",
                    test_code=item,
                )
            )
            continue
        raise DatasetProtocolError(
            f"DatasetProblemRecord.tests must contain only strings or dict objects for problem '{record.problem_id}'."
        )

    return ProgrammingProblem(
        problem_id=record.problem_id,
        problem_title=record.problem_title or record.problem_id,
        problem_statement=record.prompt,
        entry_point=record.entry_point,
        programming_language=record.programming_language or "python",
        starter_code=record.starter_code,
        reference_solution=ReferenceSolution(
            solution_id=f"{record.problem_id}::reference",
            problem_id=record.problem_id,
            programming_language=record.programming_language or "python",
            reference_code=record.reference_code,
        ),
        test_cases=normalized_test_cases,
        difficulty=record.difficulty,
        topic=record.topic,
        source_dataset_name=record.source_dataset_name,
        source_split_name=record.source_split_name,
    )


def _load_payloads(dataset_path: str, dataset_format: str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if dataset_format == "jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if dataset_format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("problems"), list):
            return payload["problems"]
        raise DatasetProtocolError("JSON dataset must be a list or an object with a 'problems' list.")

    raise DatasetProtocolError(f"Unsupported dataset format: {dataset_format}")


def load_dataset_records(dataset_path: str, dataset_format: str = "jsonl") -> list[DatasetProblemRecord]:
    """Load and validate dataset records from disk."""

    payloads = _load_payloads(dataset_path, dataset_format)
    return [validate_dataset_problem_record(payload) for payload in payloads]


def load_programming_problem_dataset(
    dataset_path: str,
    dataset_format: str = "jsonl",
) -> list[ProgrammingProblem]:
    """Load validated dataset records into internal problem schemas."""

    return [
        dataset_record_to_problem(record)
        for record in load_dataset_records(dataset_path, dataset_format)
    ]
