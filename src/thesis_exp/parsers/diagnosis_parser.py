"""Robust parsers for model diagnosis outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from thesis_exp.llm.base import RawLLMResponse
from thesis_exp.schemas.sample import ModelDiagnosisOutput

REQUIRED_DIAGNOSIS_FIELDS = (
    "bug_type",
    "bug_line",
    "explanation",
    "fix_strategy",
    "patched_code",
    "confidence",
)

DIAGNOSIS_ONLY_FIELDS = ("bug_type", "bug_line", "explanation")

_BUG_TYPE_ALIASES = {
    "off by one": "loop_boundary_error",
    "off-by-one": "loop_boundary_error",
    "off_by_one": "loop_boundary_error",
    "offbyone": "loop_boundary_error",
    "off_by_one_error": "loop_boundary_error",
    "loop_bound_error": "loop_boundary_error",
    "loopboundaryerror": "loop_boundary_error",
    "loop_boundary_error": "loop_boundary_error",
    "wrong loop bound": "loop_boundary_error",
    "wrong_loop_bound": "loop_boundary_error",
    "accumulator init error": "accumulator_init_error",
    "accumulator initialization error": "accumulator_init_error",
    "accumulator_init_error": "accumulator_init_error",
    "condition inversion": "conditional_logic_error",
    "condition_inversion": "conditional_logic_error",
    "premature return": "premature_return",
    "premature_return": "premature_return",
    "wrong comparison operator": "conditional_logic_error",
    "wrong_comparison_operator": "conditional_logic_error",
    "logic_error_inverted_condition": "conditional_logic_error",
    "inverted_condition": "conditional_logic_error",
    "conditional_logic_error": "conditional_logic_error",
    "conditional logic error": "conditional_logic_error",
}


@dataclass(slots=True)
class ParsedJsonPayload:
    """Intermediate result for parsed JSON content."""

    payload: dict[str, Any] | None
    extraction_mode: str
    error_messages: list[str]


def normalize_bug_type_label(label: Any) -> tuple[str | None, str | None]:
    """Normalize a raw bug type label into a stable internal label."""

    if label is None:
        return None, "Missing bug_type value."
    if not isinstance(label, str):
        return None, f"bug_type must be a string, got {type(label).__name__}."

    normalized_key = re.sub(r"[\s\-]+", "_", label.strip().lower())
    normalized_key = re.sub(r"[^a-z0-9_]+", "", normalized_key)
    normalized_label = _BUG_TYPE_ALIASES.get(normalized_key, _BUG_TYPE_ALIASES.get(label.strip().lower()))

    if normalized_label is None:
        return None, f"Unsupported bug_type label: {label!r}."
    return normalized_label, None


def normalize_bug_line(value: Any) -> tuple[int | None, str | None]:
    """Normalize a raw bug_line value into an integer line number."""

    if value is None:
        return None, None

    if isinstance(value, bool):
        return None, "bug_line must be an integer or null, not a boolean."

    if isinstance(value, int):
        if value <= 0:
            return None, f"bug_line must be positive, got {value}."
        return value, None

    if isinstance(value, float):
        if value.is_integer() and value > 0:
            return int(value), None
        return None, f"bug_line must be an integer-like value, got {value}."

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "unknown", "n/a"}:
            return None, None

        if text.isdigit():
            line_number = int(text)
            if line_number <= 0:
                return None, f"bug_line must be positive, got {line_number}."
            return line_number, None

        match = re.search(r"line\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), None

        return None, f"Could not normalize bug_line value: {value!r}."

    return None, f"bug_line has unsupported type {type(value).__name__}."


def normalize_confidence_value(value: Any) -> tuple[float | None, str | None]:
    """Normalize confidence into a float between 0 and 1."""

    if value is None:
        return None, "Missing confidence value."

    if isinstance(value, bool):
        return None, "confidence must be numeric, not a boolean."

    numeric_value: float | None = None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
    elif isinstance(value, str):
        text = value.strip().rstrip("%")
        try:
            numeric_value = float(text)
            if value.strip().endswith("%"):
                numeric_value /= 100.0
        except ValueError:
            return None, f"Could not normalize confidence value: {value!r}."
    else:
        return None, f"confidence has unsupported type {type(value).__name__}."

    if not 0.0 <= numeric_value <= 1.0:
        return None, f"confidence must be between 0 and 1, got {numeric_value}."
    return numeric_value, None


def strict_json_parse(text: str) -> ParsedJsonPayload:
    """Parse a response as strict JSON with no extra text."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ParsedJsonPayload(
            payload=None,
            extraction_mode="strict_json_failed",
            error_messages=[f"Strict JSON parse failed: {exc.msg} at line {exc.lineno}, column {exc.colno}."],
        )

    if not isinstance(payload, dict):
        return ParsedJsonPayload(
            payload=None,
            extraction_mode="strict_json_non_object",
            error_messages=["Strict JSON parse succeeded, but the top-level value is not a JSON object."],
        )

    return ParsedJsonPayload(payload=payload, extraction_mode="strict_json", error_messages=[])


def extract_json_object_from_text(text: str) -> ParsedJsonPayload:
    """Extract the first balanced JSON object from noisy text."""

    start_index = text.find("{")
    if start_index < 0:
        return ParsedJsonPayload(
            payload=None,
            extraction_mode="json_extraction_failed",
            error_messages=["No JSON object start delimiter '{' was found in the raw response."],
        )

    depth = 0
    in_string = False
    escape_next = False
    candidate_ranges: list[tuple[int, int]] = []

    for index, char in enumerate(text[start_index:], start=start_index):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate_ranges.append((start_index, index + 1))
                break

    if not candidate_ranges:
        return ParsedJsonPayload(
            payload=None,
            extraction_mode="json_extraction_failed",
            error_messages=["Found '{' but could not extract a balanced JSON object from the raw response."],
        )

    candidate_text = text[candidate_ranges[0][0] : candidate_ranges[0][1]]
    try:
        payload = json.loads(candidate_text)
    except json.JSONDecodeError as exc:
        return ParsedJsonPayload(
            payload=None,
            extraction_mode="json_extraction_invalid",
            error_messages=[f"Extracted JSON candidate is invalid: {exc.msg} at line {exc.lineno}, column {exc.colno}."],
        )

    if not isinstance(payload, dict):
        return ParsedJsonPayload(
            payload=None,
            extraction_mode="json_extraction_non_object",
            error_messages=["Extracted JSON content is not a top-level object."],
        )

    return ParsedJsonPayload(payload=payload, extraction_mode="json_extracted", error_messages=[])


def parse_response_json_payload(text: str) -> ParsedJsonPayload:
    """Parse JSON strictly, then fall back to extraction from surrounding text."""

    strict_result = strict_json_parse(text)
    if strict_result.payload is not None:
        return strict_result

    extracted_result = extract_json_object_from_text(text)
    if extracted_result.payload is not None:
        return ParsedJsonPayload(
            payload=extracted_result.payload,
            extraction_mode=extracted_result.extraction_mode,
            error_messages=strict_result.error_messages,
        )

    return ParsedJsonPayload(
        payload=None,
        extraction_mode=extracted_result.extraction_mode,
        error_messages=strict_result.error_messages + extracted_result.error_messages,
    )


def validate_required_fields(
    payload: dict[str, Any],
    *,
    diagnosis_only: bool = False,
) -> list[str]:
    """Validate presence of required diagnosis fields."""

    required = DIAGNOSIS_ONLY_FIELDS if diagnosis_only else REQUIRED_DIAGNOSIS_FIELDS
    missing_fields = [f for f in required if f not in payload]
    if not missing_fields:
        return []
    return [f"Missing required fields: {', '.join(missing_fields)}."]


def parse_model_diagnosis_output(
    raw_response: RawLLMResponse,
    *,
    diagnosis_output_id: str,
    response_schema_name: str | None = None,
    diagnosis_only: bool = False,
) -> ModelDiagnosisOutput:
    """Parse one raw model response into a structured diagnosis object."""

    parse_result = parse_response_json_payload(raw_response.response_text)
    payload = parse_result.payload
    error_messages = list(parse_result.error_messages)

    parsed_bug_type: str | None = None
    parsed_bug_line: int | None = None
    parsed_bug_explanation = ""
    parsed_fix_strategy = ""
    parsed_repaired_code: str | None = None
    parsed_confidence: float | None = None
    response_format_valid = False

    if payload is None:
        error_messages.append("Could not parse any valid diagnosis JSON object from the raw response.")
    else:
        schema_name = response_schema_name or ""
        is_diagnosis_only = diagnosis_only or "diagnosis_only" in schema_name
        error_messages.extend(validate_required_fields(payload, diagnosis_only=is_diagnosis_only))

        if "bug_type" in payload:
            parsed_bug_type, bug_type_error = normalize_bug_type_label(payload.get("bug_type"))
            if bug_type_error:
                error_messages.append(bug_type_error)

        if "bug_line" in payload:
            parsed_bug_line, bug_line_error = normalize_bug_line(payload.get("bug_line"))
            if bug_line_error:
                error_messages.append(bug_line_error)

        if "explanation" in payload:
            if isinstance(payload["explanation"], str):
                parsed_bug_explanation = payload["explanation"].strip()
            else:
                error_messages.append(
                    f"Field 'explanation' must be a string, got {type(payload['explanation']).__name__}."
                )

        if "fix_strategy" in payload:
            if isinstance(payload["fix_strategy"], str):
                parsed_fix_strategy = payload["fix_strategy"].strip()
            else:
                error_messages.append(
                    f"Field 'fix_strategy' must be a string, got {type(payload['fix_strategy']).__name__}."
                )

        if "patched_code" in payload:
            if isinstance(payload["patched_code"], str):
                parsed_repaired_code = payload["patched_code"]
            elif payload["patched_code"] is None:
                parsed_repaired_code = None
            else:
                error_messages.append(
                    f"Field 'patched_code' must be a string or null, got {type(payload['patched_code']).__name__}."
                )

        if "confidence" in payload:
            parsed_confidence, confidence_error = normalize_confidence_value(payload.get("confidence"))
            if confidence_error:
                error_messages.append(confidence_error)

        response_format_valid = len(error_messages) == 0

    return ModelDiagnosisOutput(
        diagnosis_output_id=diagnosis_output_id,
        sample_id=raw_response.sample_id,
        model_provider_name=raw_response.provider_name,
        model_name=raw_response.model_name,
        prompt_template_name=raw_response.prompt_template_name,
        response_schema_name=response_schema_name or str(
            raw_response.response_metadata.get("response_schema_name", "")
        ),
        raw_response_text=raw_response.response_text,
        parsed_bug_type=parsed_bug_type,
        parsed_bug_line_start=parsed_bug_line,
        parsed_bug_line_end=parsed_bug_line,
        parsed_bug_location_explanation=f"Normalized from extraction mode: {parse_result.extraction_mode}.",
        parsed_bug_explanation=parsed_bug_explanation,
        parsed_fix_strategy=parsed_fix_strategy,
        parsed_repair_rationale=parsed_fix_strategy,
        parsed_repaired_code=parsed_repaired_code,
        parsed_confidence=parsed_confidence,
        response_format_valid=response_format_valid,
        parsing_error_message=" ".join(error_messages).strip(),
    )


def parse_repair_response(text: str) -> tuple[str | None, str]:
    """Parse repair-only response. Returns (patched_code, error_message)."""

    result = parse_response_json_payload(text)
    if result.payload is None:
        return None, " ".join(result.error_messages).strip() or "Could not parse repair JSON."

    payload = result.payload
    if "patched_code" not in payload:
        return None, "Missing patched_code field in repair response."

    val = payload["patched_code"]
    if isinstance(val, str) and val.strip():
        return val.strip(), ""
    if val is None:
        return None, "patched_code is null."
    return None, f"patched_code must be a non-empty string, got {type(val).__name__}."
