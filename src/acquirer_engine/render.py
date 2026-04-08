"""Rendering rationales and summary tables as markdown / HTML / JSON."""

from __future__ import annotations

import json
from pathlib import Path

from acquirer_engine.schemas import AcquirerRationale, RerankedCandidate


def render_rationale_markdown(rationale: AcquirerRationale) -> str:
    return f"""# {rationale.acquirer_name}

## Acquirer Overview
{rationale.acquirer_overview}

## Strategic Fit Thesis
{rationale.strategic_fit_thesis}

## Precedent Activity
{rationale.precedent_activity}

## Valuation Context
{rationale.valuation_context}

## Risk Flags
{rationale.risk_flags}

## Conviction Level
**{rationale.conviction_level}**
"""


def render_summary_table(candidates: list[RerankedCandidate]) -> str:
    lines = [
        "| Rank | Acquirer | Score | Conviction | Key Signals |",
        "|------|----------|-------|------------|-------------|",
    ]
    for i, c in enumerate(candidates, 1):
        signals = "; ".join(c.supporting_signals[:2])
        lines.append(
            f"| {i} | {c.acquirer_name} | {c.likelihood_score} | "
            f"{c.conviction_level} | {signals} |"
        )
    return "\n".join(lines)


def render_full_report(
    candidates: list[RerankedCandidate],
    rationales: list[AcquirerRationale],
) -> str:
    sections = [
        "# M&A Acquirer Identification Report\n",
        "## Top 10 Acquirer Summary\n",
        render_summary_table(candidates),
        "\n---\n",
    ]
    rationale_map = {r.acquirer_name: r for r in rationales}
    for c in candidates:
        r = rationale_map.get(c.acquirer_name)
        if r:
            sections.append(render_rationale_markdown(r))
            sections.append("\n---\n")
    return "\n".join(sections)


def export_json(
    candidates: list[RerankedCandidate],
    rationales: list[AcquirerRationale],
    path: Path,
) -> None:
    data = {
        "ranked_candidates": [c.model_dump() for c in candidates],
        "rationales": [r.model_dump() for r in rationales],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def export_markdown(
    candidates: list[RerankedCandidate],
    rationales: list[AcquirerRationale],
    path: Path,
) -> None:
    report = render_full_report(candidates, rationales)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(report)
