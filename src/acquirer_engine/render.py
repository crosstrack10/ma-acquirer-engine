"""Rendering rationales and summary tables as markdown / HTML / JSON."""

from __future__ import annotations

import json
import re
from datetime import datetime
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


def _safe_filename(name: str) -> str:
    """Convert acquirer name to a safe filename slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug


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


def export_individual_rationales(
    candidates: list[RerankedCandidate],
    rationales: list[AcquirerRationale],
    output_dir: Path,
) -> Path:
    """Export each acquirer rationale as a separate markdown file.

    Returns the directory where files were written.
    """
    rationale_map = {r.acquirer_name: r for r in rationales}
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for i, c in enumerate(candidates, 1):
        r = rationale_map.get(c.acquirer_name)
        if not r:
            continue
        slug = _safe_filename(c.acquirer_name)
        filename = f"{i:02d}_{slug}.md"
        with open(individual_dir / filename, "w") as f:
            f.write(render_rationale_markdown(r))

    return individual_dir


def export_run(
    candidates: list[RerankedCandidate],
    rationales: list[AcquirerRationale],
    output_dir: Path,
    run_metadata: dict | None = None,
) -> Path:
    """Save a complete timestamped run to outputs/runs/<timestamp>/.

    Creates:
      - summary.md (full report)
      - summary.json (structured data)
      - individual/<NN>_<acquirer>.md (one file per acquirer)
      - metadata.json (run metadata if provided)

    Returns the run directory path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Full report
    export_markdown(candidates, rationales, run_dir / "summary.md")
    export_json(candidates, rationales, run_dir / "summary.json")

    # Individual rationales
    export_individual_rationales(candidates, rationales, run_dir)

    # Run metadata
    if run_metadata:
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(run_metadata, f, indent=2, default=str)

    return run_dir
