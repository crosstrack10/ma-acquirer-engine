"""LLM-based reranking of deterministic top candidates."""

from __future__ import annotations

import logging

from acquirer_engine.llm.litellm_client import LiteLLMClient
from acquirer_engine.prompts.templates import build_rerank_messages
from acquirer_engine.schemas import (
    CandidateScore,
    EvidencePacket,
    RerankedCandidate,
    RerankedCandidateList,
    TargetProfile,
)

logger = logging.getLogger(__name__)


def rerank_candidates(
    evidence_packets: list[EvidencePacket],
    target: TargetProfile,
    model: str,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> tuple[list[RerankedCandidate], dict]:
    """Rerank candidates using an LLM. Returns (top_10, metadata)."""
    client = LiteLLMClient()
    messages = build_rerank_messages(target, evidence_packets)

    logger.info(f"Reranking {len(evidence_packets)} candidates with {model}")
    result, metadata = client.generate_structured(
        messages=messages,
        model=model,
        schema=RerankedCandidateList,
        temperature=temperature,
        max_tokens=8192,
        max_retries=max_retries,
    )

    # Sort by likelihood descending
    ranked = sorted(result.candidates, key=lambda c: c.likelihood_score, reverse=True)
    metadata["candidates_in"] = len(evidence_packets)
    metadata["candidates_out"] = len(ranked)
    return ranked, metadata
