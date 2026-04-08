"""Prompt templates for reranking and rationale generation."""

from __future__ import annotations

import json
from typing import Any

from acquirer_engine.schemas import EvidencePacket, TargetProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_block(target: TargetProfile) -> str:
    return (
        f"Sector: {target.sector}\n"
        f"Estimated Enterprise Value: ~${target.deal_size_mm:.0f}M\n"
        f"EBITDA Margin: ~{target.ebitda_margin_pct:.0f}%\n"
        f"Geography: {target.geography}\n"
        f"Ownership: {target.target_ownership}\n"
        f"Key Strategic Themes: {', '.join(target.rationale_tags)}\n"
        f"Description: {target.description}"
    )


def _evidence_summary(ep: EvidencePacket) -> str:
    p = ep.acquirer_profile
    lines = [
        f"Acquirer: {p.name}",
        f"Type: {p.acquirer_type}",
        f"Total Deals: {p.total_deals} ({p.closed_deals} closed)",
        f"Close Rate: {p.close_rate:.0%}",
        f"Sectors: {json.dumps(p.sectors)}",
        f"Geographies: {json.dumps(p.geographies)}",
        f"Deal Size (closed): min=${p.deal_size_stats.min:.0f}M, "
        f"max=${p.deal_size_stats.max:.0f}M, median=${p.deal_size_stats.median:.0f}M",
        f"EBITDA Margin (closed targets): median={p.ebitda_margin_stats.median:.1f}%",
        f"EV/EBITDA (closed): median={p.ev_ebitda_stats.median:.1f}x",
        f"Rationale Tags: {json.dumps(p.rationale_tags)}",
        f"Most Recent Activity: {p.most_recent_year} {p.most_recent_quarter}",
        f"Deterministic Score: {ep.total_score:.3f}",
        f"Score Breakdown: {json.dumps(ep.score_breakdown)}",
    ]
    if ep.valuation_summary.median_ev_ebitda is not None:
        lines.append(
            f"Valuation Summary: EV/EBITDA median={ep.valuation_summary.median_ev_ebitda}x, "
            f"EV/Revenue median={ep.valuation_summary.median_ev_revenue}x, "
            f"Deal size range=${ep.valuation_summary.deal_size_range_mm[0]:.0f}M–"
            f"${ep.valuation_summary.deal_size_range_mm[1]:.0f}M"
        )
    lines.append("\nPrecedent Deals:")
    for d in ep.precedent_deals:
        lines.append(
            f"  - {d.target_company} ({d.sector}, {d.deal_year}): "
            f"${d.deal_size_mm:.0f}M, {d.deal_type}, EV/EBITDA={d.ev_ebitda_multiple:.1f}x, "
            f"{d.outcome}, Tags: {', '.join(d.rationale_tags)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rerank prompt
# ---------------------------------------------------------------------------

RERANK_SYSTEM = """You are a senior M&A analyst at a leading investment bank. Your task is to evaluate a set of potential acquirers for a target company based on historical transaction evidence.

RULES:
1. Use ONLY the provided evidence to support your analysis. Do not invent facts.
2. Select exactly 10 acquirers from the candidates provided.
3. Each acquirer must have a distinct rationale — avoid generic or repetitive reasoning.
4. Assign a likelihood_score (0-100) reflecting acquisition plausibility.
5. Assign a conviction_level: "High", "Medium", or "Low".
6. List at least 2 specific risk_flags per acquirer.
7. Provide at least 1 supporting_signal per acquirer grounded in the evidence.
8. If evidence is sparse for a candidate, acknowledge it rather than fabricating.

Return ONLY a JSON object matching this exact structure:
{
  "candidates": [
    {
      "acquirer_name": "...",
      "likelihood_score": 0-100,
      "conviction_level": "High" | "Medium" | "Low",
      "supporting_signals": ["..."],
      "risk_flags": ["...", "..."],
      "summary": "2-3 sentence summary"
    }
  ]
}

Return exactly 10 candidates ordered by likelihood_score descending. No markdown, no prose outside JSON."""


def build_rerank_messages(
    target: TargetProfile,
    evidence_packets: list[EvidencePacket],
) -> list[dict[str, str]]:
    candidate_blocks = []
    for i, ep in enumerate(evidence_packets, 1):
        candidate_blocks.append(f"--- Candidate {i} ---\n{_evidence_summary(ep)}")

    user_content = (
        f"TARGET COMPANY PROFILE:\n{_target_block(target)}\n\n"
        f"CANDIDATE ACQUIRERS ({len(evidence_packets)} candidates):\n\n"
        + "\n\n".join(candidate_blocks)
        + "\n\nSelect the 10 most likely acquirers and return the JSON."
    )

    return [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Rationale prompt
# ---------------------------------------------------------------------------

RATIONALE_SYSTEM = """You are a senior investment banker writing a one-page acquirer rationale for a client presentation. Your output will be read by managing directors and must be precise, evidence-grounded, and differentiated.

RULES:
1. Use ONLY the provided evidence. Do not invent deals, companies, or valuation ranges.
2. Write in concise, banker-grade prose — no marketing language or filler.
3. Each section must contain specific, non-generic content.
4. If evidence is limited, state that clearly rather than speculating.
5. Risk flags must be concrete and specific to this acquirer.
6. Conviction level must be consistent with the evidence strength.

Return ONLY a JSON object matching this structure:
{
  "acquirer_name": "...",
  "acquirer_overview": "1-2 paragraph overview of the acquirer's identity, strategy, and M&A track record",
  "strategic_fit_thesis": "2-3 paragraphs on why this acquirer is a strong fit for the target",
  "precedent_activity": "1-2 paragraphs citing specific past deals that demonstrate relevant experience",
  "valuation_context": "1 paragraph on expected valuation range based on precedent multiples",
  "risk_flags": "1 paragraph with at least 2 specific risk factors",
  "conviction_level": "High" | "Medium" | "Low"
}

No markdown fences. No text outside the JSON."""


def build_rationale_messages(
    target: TargetProfile,
    evidence: EvidencePacket,
) -> list[dict[str, str]]:
    user_content = (
        f"TARGET COMPANY PROFILE:\n{_target_block(target)}\n\n"
        f"ACQUIRER EVIDENCE:\n{_evidence_summary(evidence)}\n\n"
        "Generate the one-page acquirer rationale as JSON."
    )

    return [
        {"role": "system", "content": RATIONALE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
