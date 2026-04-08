"""Provider-neutral LLM interface."""

from __future__ import annotations

from typing import Protocol, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str: ...

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        model: str,
        schema: Type[T],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> T: ...
