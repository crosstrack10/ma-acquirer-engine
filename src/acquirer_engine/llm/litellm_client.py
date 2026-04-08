"""LiteLLM-based unified model gateway for OpenAI and Anthropic."""

from __future__ import annotations

import logging
import os
from typing import Type, TypeVar

import litellm
from pydantic import BaseModel

from acquirer_engine.llm.parsing import parse_llm_response
from acquirer_engine.llm.retry import generate_with_retry
from acquirer_engine.settings import get_settings

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


class LiteLLMClient:
    """Provider-agnostic LLM client using LiteLLM."""

    def __init__(self) -> None:
        settings = get_settings()
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: Type[T],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> tuple[T, dict]:
        """Generate and validate structured output. Returns (result, metadata)."""
        import time

        start = time.time()

        def _call(msgs: list[dict[str, str]]) -> str:
            return self.generate(msgs, model, temperature, max_tokens)

        result, retry_count = generate_with_retry(_call, messages, schema, max_retries)
        elapsed_ms = (time.time() - start) * 1000

        metadata = {
            "model": model,
            "latency_ms": round(elapsed_ms, 1),
            "retry_count": retry_count,
        }
        return result, metadata
