"""Build acquirer profiles from transaction history."""

from __future__ import annotations

import statistics
from collections import Counter

from acquirer_engine.schemas import AcquirerProfile, DealSizeStats, Transaction


def _compute_stats(values: list[float]) -> DealSizeStats:
    if not values:
        return DealSizeStats()
    return DealSizeStats(
        min=min(values),
        max=max(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        count=len(values),
    )


def build_acquirer_profiles(
    transactions: list[Transaction],
) -> dict[str, AcquirerProfile]:
    grouped: dict[str, list[Transaction]] = {}
    for txn in transactions:
        grouped.setdefault(txn.acquirer, []).append(txn)

    profiles: dict[str, AcquirerProfile] = {}
    for name, txns in grouped.items():
        closed = [t for t in txns if t.outcome == "Closed"]
        closed_deal_sizes = [t.deal_size_mm for t in closed]
        closed_margins = [t.ebitda_margin_pct for t in closed]
        closed_ev_ebitda = [t.ev_ebitda_multiple for t in closed]

        sectors = Counter(t.sector for t in txns)
        geographies = Counter(t.geography for t in txns)
        deal_types = Counter(t.deal_type for t in txns)

        tag_counter: Counter[str] = Counter()
        for t in txns:
            tag_counter.update(t.strategic_rationale_tags)

        most_recent = max(txns, key=lambda t: (t.deal_year, t.deal_quarter))

        close_rate = len(closed) / len(txns) if txns else 0.0

        profiles[name] = AcquirerProfile(
            name=name,
            acquirer_type=txns[0].acquirer_type,
            total_deals=len(txns),
            closed_deals=len(closed),
            sectors=dict(sectors),
            deal_size_stats=_compute_stats(closed_deal_sizes),
            ebitda_margin_stats=_compute_stats(closed_margins),
            ev_ebitda_stats=_compute_stats(closed_ev_ebitda),
            geographies=dict(geographies),
            rationale_tags=dict(tag_counter),
            deal_types=dict(deal_types),
            most_recent_year=most_recent.deal_year,
            most_recent_quarter=most_recent.deal_quarter,
            close_rate=close_rate,
            transactions=txns,
        )
    return profiles
