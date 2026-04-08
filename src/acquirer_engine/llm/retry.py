"""Retry and repair logic for LLM structured generation."""

from __future__ import annotations

import logging
import time
from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from acquirer_engine.llm.parsing import format_validation_error, parse_llm_response

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


def generate_with_retry(
    generate_fn: Callable[[list[dict[str, str]]], str],
    messages: list[dict[str, str]],
    schema: Type[T],
    max_retries: int = 3,
) -> tuple[T, int]:
    """Call generate_fn, parse, validate. Retry with repair prompt on failure.

    Returns (parsed_result, retry_count).
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            raw = generate_fn(messages)
            result = parse_llm_response(raw, schema)
            return result, attempt
        except (ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(f"Attempt {attempt + 1} failed: {exc}")
            # Build repair prompt
            if isinstance(exc, ValidationError):
                error_detail = format_validation_error(exc)
            else:
                error_detail = str(exc)
            repair_msg = {
                "role": "user",
                "content": (
                    f"Your previous response failed validation:\n{error_detail}\n\n"
                    "Please fix the errors and return ONLY valid JSON matching the required schema."
                ),
            }
            messages = messages + [{"role": "assistant", "content": raw}, repair_msg]
        except Exception as exc:
            # Rate limit or transient — exponential backoff
            logger.warning(f"Attempt {attempt + 1} transient error: {exc}")
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"All {max_retries + 1} attempts failed. Last error: {last_error}")
