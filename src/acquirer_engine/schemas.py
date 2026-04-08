"""Pydantic models for the entire pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Target profile
# ---------------------------------------------------------------------------

class TargetProfile(BaseModel):
    sector: str = "Healthcare Services"
    deal_size_mm: float = 200.0
    ebitda_margin_pct: float = 18.0
    geography: str = "Multi-Regional"
    acquirer_type_preference: str | None = None
    target_ownership: str = "Private"
    description: str = ""
    rationale_tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Transaction (one CSV row)
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    transaction_id: str
    target_company: str
    acquirer: str
    sector: str
    sub_sector: str
    deal_year: int
    deal_quarter: str
    deal_type: str
    geography: str
    financing_type: str
    deal_size_mm: float
    target_revenue_mm: float
    target_ebitda_mm: float
    ebitda_margin_pct: float
    revenue_growth_pct: float
    ev_ebitda_multiple: float
    ev_revenue_multiple: float
    synergy_pct_of_deal: float
    outcome: str
    strategic_rationale_tags: list[str] = Field(default_factory=list)
    num_bidders: int
    days_to_close: float | None = None
    acquirer_type: str
    target_ownership_pre: str


# ---------------------------------------------------------------------------
# Acquirer profile (aggregated from transactions)
# ---------------------------------------------------------------------------

class DealSizeStats(BaseModel):
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    count: int = 0


class AcquirerProfile(BaseModel):
    name: str
    acquirer_type: str
    total_deals: int = 0
    closed_deals: int = 0
    sectors: dict[str, int] = Field(default_factory=dict)
    deal_size_stats: DealSizeStats = Field(default_factory=DealSizeStats)
    ebitda_margin_stats: DealSizeStats = Field(default_factory=DealSizeStats)
    ev_ebitda_stats: DealSizeStats = Field(default_factory=DealSizeStats)
    geographies: dict[str, int] = Field(default_factory=dict)
    rationale_tags: dict[str, int] = Field(default_factory=dict)
    deal_types: dict[str, int] = Field(default_factory=dict)
    most_recent_year: int = 2015
    most_recent_quarter: str = "Q1"
    close_rate: float = 0.0
    transactions: list[Transaction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

class CandidateScore(BaseModel):
    acquirer_name: str
    sub_scores: dict[str, float] = Field(default_factory=dict)
    total_score: float = 0.0
    rank: int = 0


# ---------------------------------------------------------------------------
# Evidence packet (sent to LLM)
# ---------------------------------------------------------------------------

class PrecedentDeal(BaseModel):
    transaction_id: str
    target_company: str
    sector: str
    deal_size_mm: float
    deal_year: int
    deal_type: str
    ev_ebitda_multiple: float
    outcome: str
    rationale_tags: list[str] = Field(default_factory=list)


class ValuationSummary(BaseModel):
    median_ev_ebitda: float | None = None
    median_ev_revenue: float | None = None
    deal_size_range_mm: tuple[float, float] | None = None


class EvidencePacket(BaseModel):
    target_profile: TargetProfile
    acquirer_profile: AcquirerProfile
    precedent_deals: list[PrecedentDeal] = Field(default_factory=list)
    valuation_summary: ValuationSummary = Field(default_factory=ValuationSummary)
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    total_score: float = 0.0


# ---------------------------------------------------------------------------
# LLM reranking output
# ---------------------------------------------------------------------------

ConvictionLevel = Literal["High", "Medium", "Low"]


class RerankedCandidate(BaseModel):
    acquirer_name: str
    likelihood_score: int = Field(ge=0, le=100)
    conviction_level: ConvictionLevel
    supporting_signals: list[str] = Field(min_length=1)
    risk_flags: list[str] = Field(min_length=2)
    summary: str


class RerankedCandidateList(BaseModel):
    candidates: list[RerankedCandidate] = Field(min_length=10, max_length=10)

    @field_validator("candidates")
    @classmethod
    def no_duplicate_acquirers(cls, v: list[RerankedCandidate]) -> list[RerankedCandidate]:
        names = [c.acquirer_name for c in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate acquirer names in reranked list")
        return v


# ---------------------------------------------------------------------------
# Rationale output
# ---------------------------------------------------------------------------

class AcquirerRationale(BaseModel):
    acquirer_name: str
    acquirer_overview: str
    strategic_fit_thesis: str
    precedent_activity: str
    valuation_context: str
    risk_flags: str
    conviction_level: ConvictionLevel


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

class ExperimentRecord(BaseModel):
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: str
    prompt_version: str
    prompt_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    retry_count: int = 0
    parse_success: bool = True
    quality_scores: dict[str, float] = Field(default_factory=dict)
    notes: str = ""
