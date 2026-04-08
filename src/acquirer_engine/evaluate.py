"""Evaluation harness for model and prompt comparison."""

from __future__ import annotations

import logging

from acquirer_engine.schemas import AcquirerRationale, RerankedCandidate

logger = logging.getLogger(__name__)


def evaluate_rerank_quality(candidates: list[RerankedCandidate]) -> dict[str, float]:
    """Compute basic quality metrics on reranked output."""
    names = [c.acquirer_name for c in candidates]
    unique_names = set(names)

    # Score differentiation
    scores = [c.likelihood_score for c in candidates]
    score_spread = max(scores) - min(scores) if scores else 0

    # Risk flag completeness
    risk_counts = [len(c.risk_flags) for c in candidates]
    min_risks = min(risk_counts) if risk_counts else 0

    return {
        "candidate_count": len(candidates),
        "unique_candidates": len(unique_names),
        "duplicate_rate": 1.0 - len(unique_names) / len(names) if names else 0.0,
        "score_spread": score_spread,
        "min_risk_flags": min_risks,
        "avg_supporting_signals": (
            sum(len(c.supporting_signals) for c in candidates) / len(candidates)
            if candidates else 0.0
        ),
    }


def evaluate_rationale_quality(rationales: list[AcquirerRationale]) -> dict[str, float]:
    """Compute basic quality metrics on generated rationales."""
    section_fields = [
        "acquirer_overview", "strategic_fit_thesis", "precedent_activity",
        "valuation_context", "risk_flags",
    ]
    completeness_scores = []
    total_lengths = []
    for r in rationales:
        filled = sum(1 for f in section_fields if getattr(r, f, "").strip())
        completeness_scores.append(filled / len(section_fields))
        total_lengths.append(sum(len(getattr(r, f, "")) for f in section_fields))

    return {
        "rationale_count": len(rationales),
        "avg_section_completeness": (
            sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        ),
        "avg_total_length_chars": (
            sum(total_lengths) / len(total_lengths) if total_lengths else 0.0
        ),
    }
