"""Prompt version registry for tracking and reproducibility."""

from __future__ import annotations

import hashlib

from acquirer_engine.prompts.templates import RERANK_SYSTEM, RATIONALE_SYSTEM


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


PROMPT_VERSIONS = {
    "rerank": {
        "version": "v1",
        "hash": _hash_text(RERANK_SYSTEM),
        "description": "Initial rerank prompt with structured evidence and guardrails",
    },
    "rationale": {
        "version": "v1",
        "hash": _hash_text(RATIONALE_SYSTEM),
        "description": "Initial banker-facing rationale prompt with section structure",
    },
}


def get_prompt_version(prompt_name: str) -> dict:
    return PROMPT_VERSIONS.get(prompt_name, {"version": "unknown", "hash": ""})
