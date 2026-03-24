"""Structured schemas for experiment data."""

import json
import types
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from thesis_exp.common.types import BugType, TestCaseType

SchemaT = TypeVar("SchemaT", bound="JsonSerializableDataclass")


def _is_dataclass_type(value_type: Any) -> bool:
    return isinstance(value_type, type) and is_dataclass(value_type)


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            dataclass_field.name: _serialize_value(getattr(value, dataclass_field.name))
            for dataclass_field in fields(value)
        }
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _deserialize_value(value_type: Any, value: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(value_type)

    if origin in (list,):
        (item_type,) = get_args(value_type) or (Any,)
        return [_deserialize_value(item_type, item) for item in value]

    if origin in (dict,):
        key_type, item_type = get_args(value_type) or (Any, Any)
        return {
            _deserialize_value(key_type, key): _deserialize_value(item_type, item)
            for key, item in value.items()
        }

    if origin in (types.UnionType, getattr(types, "UnionType", object)):
        non_none_types = [item for item in get_args(value_type) if item is not type(None)]
        if len(non_none_types) == 1:
            return _deserialize_value(non_none_types[0], value)
        return value

    if str(origin) == "typing.Union":
        non_none_types = [item for item in get_args(value_type) if item is not type(None)]
        if len(non_none_types) == 1:
            return _deserialize_value(non_none_types[0], value)
        return value

    if _is_dataclass_type(value_type):
        return _deserialize_dataclass(value_type, value)

    return value


def _deserialize_dataclass(cls: type[SchemaT], payload: dict[str, Any]) -> SchemaT:
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}

    for dataclass_field in fields(cls):
        field_name = dataclass_field.name
        if field_name not in payload:
            continue
        field_type = type_hints.get(field_name, dataclass_field.type)
        kwargs[field_name] = _deserialize_value(field_type, payload[field_name])

    return cls(**kwargs)


@dataclass(slots=True)
class JsonSerializableDataclass:
    """Adds JSON serialization helpers to schema classes."""

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls: type[SchemaT], payload: dict[str, Any]) -> SchemaT:
        return _deserialize_dataclass(cls, payload)

    @classmethod
    def from_json(cls: type[SchemaT], payload: str) -> SchemaT:
        return cls.from_dict(json.loads(payload))


@dataclass(slots=True)
class ReferenceSolution(JsonSerializableDataclass):
    """Reference solution for a problem."""

    solution_id: str
    problem_id: str
    programming_language: str
    reference_code: str
    solution_explanation: str = ""


@dataclass(slots=True)
class TestCase(JsonSerializableDataclass):
    """Single test case definition."""

    test_case_id: str
    problem_id: str
    test_case_type: TestCaseType = "unit_assertion"
    test_input: str = ""
    expected_output: str = ""
    test_code: str = ""
    is_hidden: bool = False


@dataclass(slots=True)
class ProgrammingProblem(JsonSerializableDataclass):
    """Programming problem definition."""

    problem_id: str
    problem_title: str
    problem_statement: str
    entry_point: str
    programming_language: str
    starter_code: str
    reference_solution: ReferenceSolution
    test_cases: list[TestCase] = field(default_factory=list)
    difficulty: str = ""
    topic: str = ""
    source_dataset_name: str = ""
    source_split_name: str = ""


@dataclass(slots=True)
class BugInjectionRecord(JsonSerializableDataclass):
    """Record for one bug injection event."""

    injection_id: str
    problem_id: str
    source_solution_id: str
    bug_type: BugType
    bug_description: str
    injection_operator_name: str
    injection_line_start: int
    injection_line_end: int
    changed_line_count: int = 1
    injection_column_start: int = 0
    injection_column_end: int = 0
    original_code_snippet: str = ""
    buggy_code_snippet: str = ""
    random_seed: int | None = None


@dataclass(slots=True)
class BuggyProgramSample(JsonSerializableDataclass):
    """Student program sample with one injected bug."""

    sample_id: str
    problem_id: str
    programming_language: str
    problem_statement: str
    starter_code: str
    reference_solution_id: str
    reference_code: str
    test_cases: list[TestCase]
    bug_injection_record: BugInjectionRecord
    buggy_code: str
    sample_version: str = "original"
    entry_point: str = ""


@dataclass(slots=True)
class ModelDiagnosisOutput(JsonSerializableDataclass):
    """Structured diagnosis output from a model."""

    diagnosis_output_id: str
    sample_id: str
    model_provider_name: str
    model_name: str
    prompt_template_name: str
    response_schema_name: str
    raw_response_text: str
    parsed_bug_type: str | None = None
    parsed_bug_line_start: int | None = None
    parsed_bug_line_end: int | None = None
    parsed_bug_location_explanation: str = ""
    parsed_bug_explanation: str = ""
    parsed_fix_strategy: str = ""
    parsed_repair_rationale: str = ""
    parsed_repaired_code: str | None = None
    parsed_confidence: float | None = None
    response_format_valid: bool = False
    parsing_error_message: str = ""


@dataclass(slots=True)
class EvaluationResult(JsonSerializableDataclass):
    """Evaluation result for one diagnosis output."""

    evaluation_result_id: str
    sample_id: str
    diagnosis_output_id: str
    ground_truth_bug_type: BugType
    predicted_bug_type: str | None
    ground_truth_injection_line_start: int
    ground_truth_injection_line_end: int
    predicted_bug_line_start: int | None = None
    predicted_bug_line_end: int | None = None
    bug_type_accuracy: float = 0.0
    bug_type_accuracy_coarse: float = 0.0
    localization_accuracy: float = 0.0
    localization_accuracy_exact: float = 0.0
    #: 1.0 iff patched code passes all selected (public/hidden per config) tests; independent of mutation adequacy.
    patch_passes_selected_tests: float = 0.0
    #: Strict repair: ``patch_passes_selected_tests`` plus mutation-adequacy gate when enabled (same as historical ``repair_success``).
    repair_success: float = 0.0
    diagnosis_hallucination_rate: float = 0.0
    narrative_hallucination_rate: float = 0.0
    hallucination_rate: float = 0.0
    narrative_hallucination_detected: bool = False
    hallucination_detected: bool = False
    repair_without_true_diagnosis: bool = False
    evaluation_notes: str = ""
    #: Machine-readable mutation-adequacy outcome; see ``compute_repair_success`` docstring.
    mutation_adequacy_status: str = ""


@dataclass(slots=True)
class ExperimentConfig(JsonSerializableDataclass):
    """Configuration for an experiment run."""

    experiment_id: str
    experiment_name: str
    dataset_name: str
    programming_language: str
    model_provider_names: list[str]
    model_names: list[str]
    prompt_template_name: str
    response_schema_name: str
    bug_types_to_inject: list[BugType]
    max_samples_per_problem: int
    random_seed: int
    temperature: float = 0.0
    max_output_tokens: int = 2048
    checkpoint_file_path: str = ""
    results_directory: str = ""


# Backward-compatible aliases for the initial scaffold.
Problem = ProgrammingProblem
BuggySample = BuggyProgramSample
ModelOutput = ModelDiagnosisOutput
