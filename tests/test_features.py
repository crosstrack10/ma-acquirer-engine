"""Tests for feature functions."""

from acquirer_engine.schemas import AcquirerProfile, DealSizeStats, TargetProfile
from acquirer_engine.features import (
    sector_fit, size_fit, ebitda_fit, geography_fit,
    acquirer_type_fit, rationale_tag_fit, recency_fit, execution_fit,
)


def _make_profile(**overrides) -> AcquirerProfile:
    defaults = dict(
        name="TestCo",
        acquirer_type="Strategic",
        total_deals=10,
        closed_deals=8,
        sectors={"Healthcare Services": 5, "Physician Groups": 3, "Pharma/Biotech": 2},
        deal_size_stats=DealSizeStats(min=50, max=500, mean=200, median=180, count=8),
        ebitda_margin_stats=DealSizeStats(min=10, max=25, mean=17, median=18, count=8),
        ev_ebitda_stats=DealSizeStats(min=8, max=20, mean=14, median=13, count=8),
        geographies={"Multi-Regional": 4, "Northeast": 3, "Southeast": 3},
        rationale_tags={"Geographic Expansion": 5, "Platform Build": 3, "Scale": 2},
        deal_types={"Strategic Acquisition": 5, "Bolt-on Acquisition": 5},
        most_recent_year=2024,
        most_recent_quarter="Q2",
        close_rate=0.8,
    )
    defaults.update(overrides)
    return AcquirerProfile(**defaults)


def _make_target(**overrides) -> TargetProfile:
    defaults = dict(
        sector="Healthcare Services",
        deal_size_mm=200.0,
        ebitda_margin_pct=18.0,
        geography="Multi-Regional",
        rationale_tags=["Geographic Expansion", "Platform Build", "Margin Improvement", "Scale"],
    )
    defaults.update(overrides)
    return TargetProfile(**defaults)


def test_sector_fit_range():
    score = sector_fit(_make_profile(), _make_target())
    assert 0.0 <= score <= 1.0


def test_sector_fit_perfect():
    p = _make_profile(sectors={"Healthcare Services": 10}, total_deals=10)
    assert sector_fit(p, _make_target()) == 1.0


def test_size_fit_range():
    score = size_fit(_make_profile(), _make_target())
    assert 0.0 <= score <= 1.0


def test_size_fit_close_to_target():
    p = _make_profile(deal_size_stats=DealSizeStats(min=190, max=210, mean=200, median=200, count=5))
    score = size_fit(p, _make_target())
    assert score > 0.9


def test_ebitda_fit_range():
    assert 0.0 <= ebitda_fit(_make_profile(), _make_target()) <= 1.0


def test_geography_fit_range():
    assert 0.0 <= geography_fit(_make_profile(), _make_target()) <= 1.0


def test_acquirer_type_fit_no_preference():
    score_strat = acquirer_type_fit(_make_profile(acquirer_type="Strategic"), _make_target())
    score_fin = acquirer_type_fit(_make_profile(acquirer_type="Financial Sponsor"), _make_target())
    assert score_strat >= score_fin  # mild strategic bias when no preference


def test_rationale_tag_fit_range():
    assert 0.0 <= rationale_tag_fit(_make_profile(), _make_target()) <= 1.0


def test_recency_fit_recent_is_higher():
    recent = recency_fit(_make_profile(most_recent_year=2024, most_recent_quarter="Q3"))
    old = recency_fit(_make_profile(most_recent_year=2018, most_recent_quarter="Q1"))
    assert recent > old


def test_execution_fit_range():
    assert 0.0 <= execution_fit(_make_profile()) <= 1.0
