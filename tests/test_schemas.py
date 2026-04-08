"""Tests for Pydantic schemas validation."""

import pytest
from pydantic import ValidationError

from acquirer_engine.schemas import (
    RerankedCandidate,
    RerankedCandidateList,
    AcquirerRationale,
)


def _make_candidate(**overrides):
    defaults = dict(
        acquirer_name="TestCo",
        likelihood_score=75,
        conviction_level="High",
        supporting_signals=["Strong sector fit"],
        risk_flags=["Limited deal history", "Sector concentration"],
        summary="Test summary",
    )
    defaults.update(overrides)
    return RerankedCandidate(**defaults)


def test_reranked_candidate_valid():
    c = _make_candidate()
    assert c.likelihood_score == 75


def test_reranked_candidate_score_bounds():
    with pytest.raises(ValidationError):
        _make_candidate(likelihood_score=101)
    with pytest.raises(ValidationError):
        _make_candidate(likelihood_score=-1)


def test_reranked_candidate_min_risk_flags():
    with pytest.raises(ValidationError):
        _make_candidate(risk_flags=["Only one"])


def test_reranked_candidate_min_signals():
    with pytest.raises(ValidationError):
        _make_candidate(supporting_signals=[])


def test_reranked_list_exactly_10():
    candidates = [_make_candidate(acquirer_name=f"Co{i}") for i in range(10)]
    lst = RerankedCandidateList(candidates=candidates)
    assert len(lst.candidates) == 10


def test_reranked_list_not_10_fails():
    candidates = [_make_candidate(acquirer_name=f"Co{i}") for i in range(9)]
    with pytest.raises(ValidationError):
        RerankedCandidateList(candidates=candidates)


def test_reranked_list_no_duplicates():
    candidates = [_make_candidate(acquirer_name="SameName") for _ in range(10)]
    with pytest.raises(ValidationError):
        RerankedCandidateList(candidates=candidates)


def test_acquirer_rationale_valid():
    r = AcquirerRationale(
        acquirer_name="TestCo",
        acquirer_overview="Overview text",
        strategic_fit_thesis="Thesis text",
        precedent_activity="Precedent text",
        valuation_context="Valuation text",
        risk_flags="Risk flag text",
        conviction_level="High",
    )
    assert r.conviction_level == "High"


def test_acquirer_rationale_invalid_conviction():
    with pytest.raises(ValidationError):
        AcquirerRationale(
            acquirer_name="TestCo",
            acquirer_overview="Overview",
            strategic_fit_thesis="Thesis",
            precedent_activity="Precedent",
            valuation_context="Valuation",
            risk_flags="Risks",
            conviction_level="Very High",
        )
