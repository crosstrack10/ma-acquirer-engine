"""Deterministic scoring engine — weighted feature combination."""

from __future__ import annotations

from acquirer_engine.schemas import AcquirerProfile, CandidateScore, TargetProfile
from acquirer_engine import features as F


FEATURE_FUNCTIONS = {
    "sector_fit": lambda p, t: F.sector_fit(p, t),
    "size_fit": lambda p, t: F.size_fit(p, t),
    "ebitda_fit": lambda p, t: F.ebitda_fit(p, t),
    "geography_fit": lambda p, t: F.geography_fit(p, t),
    "acquirer_type_fit": lambda p, t: F.acquirer_type_fit(p, t),
    "rationale_fit": lambda p, t: F.rationale_tag_fit(p, t),
    "recency_fit": lambda p, t: F.recency_fit(p),
    "execution_fit": lambda p, t: F.execution_fit(p),
}


def score_candidates(
    profiles: dict[str, AcquirerProfile],
    target: TargetProfile,
    weights: dict[str, float],
    top_n: int | None = None,
) -> list[CandidateScore]:
    results: list[CandidateScore] = []
    for name, profile in profiles.items():
        sub_scores: dict[str, float] = {}
        total = 0.0
        for key, fn in FEATURE_FUNCTIONS.items():
            score = fn(profile, target)
            sub_scores[key] = round(score, 4)
            w = weights.get(key, 0.0)
            total += score * w
        results.append(CandidateScore(
            acquirer_name=name,
            sub_scores=sub_scores,
            total_score=round(total, 4),
        ))

    results.sort(key=lambda c: c.total_score, reverse=True)
    for i, c in enumerate(results, 1):
        c.rank = i

    if top_n is not None:
        results = results[:top_n]
    return results
