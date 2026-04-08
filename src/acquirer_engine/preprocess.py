"""Normalization helpers and sector adjacency mapping."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sector adjacency to Healthcare Services
# Higher = more adjacent.  Used in sector_fit feature.
# ---------------------------------------------------------------------------

SECTOR_ADJACENCY: dict[str, float] = {
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


def sector_adjacency_score(sector: str, target_sector: str = "Healthcare Services") -> float:
    if sector == target_sector:
        return 1.0
    return SECTOR_ADJACENCY.get(sector, 0.0)


# Quarter ordering for recency calculations
QUARTER_ORDER = {"Q1": 0.00, "Q2": 0.25, "Q3": 0.50, "Q4": 0.75}


def year_quarter_to_float(year: int, quarter: str) -> float:
    return year + QUARTER_ORDER.get(quarter, 0.0)
