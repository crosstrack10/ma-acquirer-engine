"""Build evidence packets for LLM consumption."""

from __future__ import annotations

import statistics

from acquirer_engine.schemas import (
    AcquirerProfile,
    CandidateScore,
    EvidencePacket,
    PrecedentDeal,
    TargetProfile,
    Transaction,
    ValuationSummary,
)
from acquirer_engine.preprocess import sector_adjacency_score


def _relevance_key(txn: Transaction, target: TargetProfile) -> float:
    """Score how relevant a single transaction is as a precedent."""
    adj = sector_adjacency_score(txn.sector, target.sector)
    size_closeness = 1.0 / (1.0 + abs(txn.deal_size_mm - target.deal_size_mm) / target.deal_size_mm)
    closed_bonus = 0.2 if txn.outcome == "Closed" else 0.0
    return adj * 0.5 + size_closeness * 0.3 + closed_bonus


def _to_precedent(txn: Transaction) -> PrecedentDeal:
    return PrecedentDeal(
        transaction_id=txn.transaction_id,
        target_company=txn.target_company,
        sector=txn.sector,
        deal_size_mm=txn.deal_size_mm,
        deal_year=txn.deal_year,
        deal_type=txn.deal_type,
        ev_ebitda_multiple=txn.ev_ebitda_multiple,
        outcome=txn.outcome,
        rationale_tags=txn.strategic_rationale_tags,
    )


def build_evidence_packet(
    profile: AcquirerProfile,
    target: TargetProfile,
    score: CandidateScore,
    max_precedents: int = 5,
) -> EvidencePacket:
    # Select most relevant precedent deals
    ranked_txns = sorted(
        profile.transactions,
        key=lambda t: _relevance_key(t, target),
        reverse=True,
    )
    precedents = [_to_precedent(t) for t in ranked_txns[:max_precedents]]

    # Valuation summary from closed deals
    closed = [t for t in profile.transactions if t.outcome == "Closed"]
    val = ValuationSummary()
    if closed:
        ev_ebitda_vals = [t.ev_ebitda_multiple for t in closed]
        ev_rev_vals = [t.ev_revenue_multiple for t in closed]
        sizes = [t.deal_size_mm for t in closed]
        val = ValuationSummary(
            median_ev_ebitda=round(statistics.median(ev_ebitda_vals), 2),
            median_ev_revenue=round(statistics.median(ev_rev_vals), 2),
            deal_size_range_mm=(round(min(sizes), 1), round(max(sizes), 1)),
        )

    return EvidencePacket(
        target_profile=target,
        acquirer_profile=profile,
        precedent_deals=precedents,
        valuation_summary=val,
        score_breakdown=score.sub_scores,
        total_score=score.total_score,
    )
