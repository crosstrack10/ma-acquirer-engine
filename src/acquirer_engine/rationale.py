"""Generate banker-facing one-page rationales for top acquirers."""

from __future__ import annotations

import logging

from acquirer_engine.llm.litellm_client import LiteLLMClient
from acquirer_engine.prompts.templates import build_rationale_messages
from acquirer_engine.schemas import (
    AcquirerRationale,
    EvidencePacket,
    RerankedCandidate,
    TargetProfile,
)

logger = logging.getLogger(__name__)


def generate_rationale(
    candidate: RerankedCandidate,
    evidence: EvidencePacket,
    target: TargetProfile,
    model: str,
    temperature: float = 0.4,
    max_retries: int = 3,
) -> tuple[AcquirerRationale, dict]:
    """Generate a single acquirer rationale. Returns (rationale, metadata)."""
    client = LiteLLMClient()
    messages = build_rationale_messages(target, evidence)

    logger.info(f"Generating rationale for {candidate.acquirer_name} with {model}")
    result, metadata = client.generate_structured(
        messages=messages,
        model=model,
        schema=AcquirerRationale,
        temperature=temperature,
        max_tokens=4096,
        max_retries=max_retries,
    )
    return result, metadata


def generate_all_rationales(
    candidates: list[RerankedCandidate],
    evidence_map: dict[str, EvidencePacket],
    target: TargetProfile,
    model: str,
    temperature: float = 0.4,
    max_retries: int = 3,
) -> tuple[list[AcquirerRationale], list[dict]]:
    """Generate rationales for all top candidates sequentially."""
    rationales: list[AcquirerRationale] = []
    all_metadata: list[dict] = []

    for candidate in candidates:
        evidence = evidence_map.get(candidate.acquirer_name)
        if evidence is None:
            logger.warning(f"No evidence packet for {candidate.acquirer_name}, skipping")
            continue
        rationale, metadata = generate_rationale(
            candidate, evidence, target, model, temperature, max_retries
        )
        rationales.append(rationale)
        all_metadata.append(metadata)

    return rationales, all_metadata
