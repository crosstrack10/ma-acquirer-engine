"""Tests for scoring engine."""

from pathlib import Path

from acquirer_engine.ingest import load_transactions
from acquirer_engine.profiles import build_acquirer_profiles
from acquirer_engine.scoring import score_candidates
from acquirer_engine.schemas import TargetProfile

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ma_transactions_500.csv"

WEIGHTS = {
    "sector_fit": 0.30,
    "size_fit": 0.20,
    "ebitda_fit": 0.10,
    "geography_fit": 0.10,
    "acquirer_type_fit": 0.10,
    "rationale_fit": 0.10,
    "recency_fit": 0.05,
    "execution_fit": 0.05,
}

TARGET = TargetProfile(
    sector="Healthcare Services",
    deal_size_mm=200.0,
    ebitda_margin_pct=18.0,
    geography="Multi-Regional",
    rationale_tags=["Geographic Expansion", "Platform Build", "Margin Improvement", "Scale"],
)


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_scoring_returns_ranked_list():
    txns = load_transactions(DATA_PATH)
    profiles = build_acquirer_profiles(txns)
    scored = score_candidates(profiles, TARGET, WEIGHTS)
    assert len(scored) == len(profiles)
    assert scored[0].rank == 1
    assert scored[0].total_score >= scored[-1].total_score


def test_scoring_top_n():
    txns = load_transactions(DATA_PATH)
    profiles = build_acquirer_profiles(txns)
    scored = score_candidates(profiles, TARGET, WEIGHTS, top_n=25)
    assert len(scored) == 25


def test_scores_are_bounded():
    txns = load_transactions(DATA_PATH)
    profiles = build_acquirer_profiles(txns)
    scored = score_candidates(profiles, TARGET, WEIGHTS)
    for c in scored:
        assert 0.0 <= c.total_score <= 1.0
        for v in c.sub_scores.values():
            assert 0.0 <= v <= 1.0
