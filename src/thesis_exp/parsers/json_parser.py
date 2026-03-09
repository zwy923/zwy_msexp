"""Helpers for parsing model outputs."""

import json
from typing import Any

from thesis_exp.parsers.diagnosis_parser import parse_response_json_payload

def parse_model_json(text: str) -> dict[str, Any]:
    """Parse JSON text returned by a model."""

    return json.loads(text)


def extract_model_json(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from noisy model text."""

    result = parse_response_json_payload(text)
    if result.payload is None:
        raise ValueError("Could not extract a valid JSON object from the raw model response.")
    return result.payload
