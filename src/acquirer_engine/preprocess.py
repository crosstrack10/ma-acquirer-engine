"""Normalization helpers and sector adjacency mapping."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations


# ---------------------------------------------------------------------------
# Sector adjacency — data-driven from acquirer cross-sector activity
# ---------------------------------------------------------------------------

# Hardcoded fallback for Healthcare Services (used when no transactions loaded)
_HEALTHCARE_SERVICES_ADJACENCY: dict[str, float] = {
    "Healthcare Services": 1.0,
    "Physician Groups": 0.7,
    "Behavioral Health": 0.7,
    "Home Health/Hospice": 0.7,
    "Health IT": 0.4,
    "Revenue Cycle": 0.4,
    "Dental": 0.2,
    "Medical Devices": 0.1,
    "Health Insurance": 0.1,
    "Pharma/Biotech": 0.1,
}

# Module-level cache for the computed adjacency matrix
_adjacency_matrix: dict[str, dict[str, float]] | None = None


def build_sector_adjacency_matrix(
    transactions: list,  # list[Transaction], avoid circular import
) -> dict[str, dict[str, float]]:
    """Compute sector adjacency from acquirer cross-sector activity.

    For each sector, counts how many acquirers also operate in every other
    sector. Normalizes to 0.0–1.0 where 1.0 = same sector.
    """
    # Group sectors by acquirer
    acquirer_sectors: dict[str, set[str]] = {}
    for txn in transactions:
        acquirer_sectors.setdefault(txn.acquirer, set()).add(txn.sector)

    # Count co-occurrences: how many acquirers are active in both sector A and B
    all_sectors = sorted({txn.sector for txn in transactions})
    cooccur: Counter = Counter()
    sector_acquirer_count: Counter = Counter()
    for _acq, secs in acquirer_sectors.items():
        for s in secs:
            sector_acquirer_count[s] += 1
        for a, b in combinations(sorted(secs), 2):
            cooccur[(a, b)] += 1
            cooccur[(b, a)] += 1

    # Build adjacency: for each target sector, score = co-occurrence / acquirers in target sector
    matrix: dict[str, dict[str, float]] = {}
    for target_sector in all_sectors:
        base_count = sector_acquirer_count[target_sector]
        if base_count == 0:
            matrix[target_sector] = {s: 0.0 for s in all_sectors}
            matrix[target_sector][target_sector] = 1.0
            continue

        scores: dict[str, float] = {}
        for other_sector in all_sectors:
            if other_sector == target_sector:
                scores[other_sector] = 1.0
            else:
                raw = cooccur.get((target_sector, other_sector), 0) / base_count
                scores[other_sector] = round(min(raw, 1.0), 3)
        matrix[target_sector] = scores

    return matrix


def set_adjacency_matrix(transactions: list) -> None:
    """Compute and cache the adjacency matrix from loaded transactions."""
    global _adjacency_matrix
    _adjacency_matrix = build_sector_adjacency_matrix(transactions)


def sector_adjacency_score(sector: str, target_sector: str = "Healthcare Services") -> float:
    """Return adjacency score between two sectors.

    Uses the data-driven matrix if available, falls back to the Healthcare
    Services hardcoded map, returns 0.0 for unknown sectors.
    """
    if sector == target_sector:
        return 1.0
    if _adjacency_matrix is not None and target_sector in _adjacency_matrix:
        return _adjacency_matrix[target_sector].get(sector, 0.0)
    # Fallback for Healthcare Services when no matrix is loaded
    if target_sector == "Healthcare Services":
        return _HEALTHCARE_SERVICES_ADJACENCY.get(sector, 0.0)
    return 0.0


# Quarter ordering for recency calculations
QUARTER_ORDER = {"Q1": 0.00, "Q2": 0.25, "Q3": 0.50, "Q4": 0.75}


def year_quarter_to_float(year: int, quarter: str) -> float:
    return year + QUARTER_ORDER.get(quarter, 0.0)
