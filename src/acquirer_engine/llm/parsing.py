"""JSON extraction and Pydantic validation from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> str:
    """Extract JSON from a response that may contain markdown fences or prose."""
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find a JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find the matching closing brace/bracket
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text.strip()


def parse_llm_response(raw: str, schema: Type[T]) -> T:
    """Parse raw LLM text into a validated Pydantic model."""
    json_str = extract_json(raw)
    data = json.loads(json_str)
    return schema.model_validate(data)


def format_validation_error(err: ValidationError) -> str:
    """Human-readable summary of validation errors for repair prompts."""
    lines = []
    for e in err.errors():
        loc = " -> ".join(str(x) for x in e["loc"])
        lines.append(f"- {loc}: {e['msg']}")
    return "\n".join(lines)
