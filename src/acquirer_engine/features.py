"""Feature functions — each returns a 0.0-to-1.0 score."""

from __future__ import annotations

import math

from acquirer_engine.schemas import AcquirerProfile, TargetProfile
from acquirer_engine.preprocess import sector_adjacency_score, year_quarter_to_float


# ---------------------------------------------------------------------------
# 1. Sector fit
# ---------------------------------------------------------------------------

def sector_fit(profile: AcquirerProfile, target: TargetProfile) -> float:
    """Weighted fraction of acquirer's deals in target or adjacent sectors."""
    if profile.total_deals == 0:
        return 0.0
    weighted_count = 0.0
    for sector, count in profile.sectors.items():
        adj = sector_adjacency_score(sector, target.sector)
        weighted_count += count * adj
    return min(weighted_count / profile.total_deals, 1.0)


# ---------------------------------------------------------------------------
# 2. Size fit
# ---------------------------------------------------------------------------

def size_fit(profile: AcquirerProfile, target: TargetProfile) -> float:
    """Gaussian similarity between acquirer's median deal size and target."""
    if profile.deal_size_stats.count == 0:
        return 0.0
    median_size = profile.deal_size_stats.median
    # Sigma proportional to target — allows reasonable range
    sigma = target.deal_size_mm * 0.8
    distance = abs(median_size - target.deal_size_mm)
    return math.exp(-0.5 * (distance / sigma) ** 2)


# ---------------------------------------------------------------------------
# 3. EBITDA margin fit
# ---------------------------------------------------------------------------

def ebitda_fit(profile: AcquirerProfile, target: TargetProfile) -> float:
    """Similarity of acquirer's target EBITDA margins to target's margin."""
    if profile.ebitda_margin_stats.count == 0:
        return 0.0
    median_margin = profile.ebitda_margin_stats.median
    sigma = 8.0  # percentage points
    distance = abs(median_margin - target.ebitda_margin_pct)
    return math.exp(-0.5 * (distance / sigma) ** 2)


# ---------------------------------------------------------------------------
# 4. Geography fit
# ---------------------------------------------------------------------------

def geography_fit(profile: AcquirerProfile, target: TargetProfile) -> float:
    """Fraction of deals in target geography or broad geographies."""
    if profile.total_deals == 0:
        return 0.0
    broad = {"National", "Multi-Regional"}
    matching = 0
    for geo, count in profile.geographies.items():
        if geo == target.geography or geo in broad:
            matching += count
    return min(matching / profile.total_deals, 1.0)


# ---------------------------------------------------------------------------
# 5. Acquirer type fit
# ---------------------------------------------------------------------------

def acquirer_type_fit(profile: AcquirerProfile, target: TargetProfile) -> float:
    """Score based on type preference, or neutral if no preference."""
    if target.acquirer_type_preference is None:
        # No preference — give a mild boost to strategic for Healthcare Services
        return 0.6 if profile.acquirer_type == "Strategic" else 0.5
    return 1.0 if profile.acquirer_type == target.acquirer_type_preference else 0.3


# ---------------------------------------------------------------------------
# 6. Strategic rationale tag fit
# ---------------------------------------------------------------------------

def rationale_tag_fit(profile: AcquirerProfile, target: TargetProfile) -> float:
    """Jaccard-ish overlap between acquirer's tag history and target tags."""
    if not target.rationale_tags or not profile.rationale_tags:
        return 0.0
    target_set = set(target.rationale_tags)
    acquirer_set = set(profile.rationale_tags.keys())
    intersection = target_set & acquirer_set
    union = target_set | acquirer_set
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# 7. Recency fit
# ---------------------------------------------------------------------------

def recency_fit(profile: AcquirerProfile, reference_year: int = 2024) -> float:
    """Exponential decay — more recent activity scores higher."""
    latest = year_quarter_to_float(profile.most_recent_year, profile.most_recent_quarter)
    ref = float(reference_year) + 0.75  # Q4 of reference year
    years_ago = ref - latest
    decay = 0.3  # lambda
    return math.exp(-decay * max(years_ago, 0.0))


# ---------------------------------------------------------------------------
# 8. Execution / close rate fit
# ---------------------------------------------------------------------------

def execution_fit(profile: AcquirerProfile) -> float:
    """Close rate capped at 1.0."""
    return min(profile.close_rate, 1.0)
