"""CLI demo script — runs the full pipeline and prints results."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from acquirer_engine.settings import get_settings, load_target_profile_yaml
from acquirer_engine.schemas import TargetProfile, EvidencePacket
from acquirer_engine.ingest import load_transactions
from acquirer_engine.profiles import build_acquirer_profiles
from acquirer_engine.scoring import score_candidates
from acquirer_engine.retrieve import build_evidence_packet
from acquirer_engine.rerank import rerank_candidates
from acquirer_engine.rationale import generate_all_rationales
from acquirer_engine.render import export_json, export_markdown, render_rationale_markdown

console = Console()


def main():
    settings = get_settings()
    target_cfg = load_target_profile_yaml()
    target = TargetProfile(**target_cfg)

    # Stage 1: Data loading
    console.rule("[bold blue]Stage 1: Data Ingestion")
    transactions = load_transactions(settings.data_path)
    console.print(f"Loaded {len(transactions)} transactions")

    # Stage 2: Profiles & scoring
    console.rule("[bold blue]Stage 2: Profiles & Scoring")
    profiles = build_acquirer_profiles(transactions)
    console.print(f"Built {len(profiles)} acquirer profiles")

    scored = score_candidates(profiles, target, settings.scoring_weights, top_n=settings.top_n_for_rerank)
    console.print(f"Top {len(scored)} scored candidates:")

    table = Table(title="Deterministic Top 25")
    table.add_column("Rank", style="dim")
    table.add_column("Acquirer")
    table.add_column("Type")
    table.add_column("Score", justify="right")
    table.add_column("Sector Fit", justify="right")
    table.add_column("Size Fit", justify="right")
    for cs in scored:
        p = profiles[cs.acquirer_name]
        table.add_row(
            str(cs.rank), cs.acquirer_name, p.acquirer_type,
            f"{cs.total_score:.3f}",
            f"{cs.sub_scores.get('sector_fit', 0):.3f}",
            f"{cs.sub_scores.get('size_fit', 0):.3f}",
        )
    console.print(table)

    # Stage 3: Evidence packets
    evidence_packets: list[EvidencePacket] = []
    evidence_map: dict[str, EvidencePacket] = {}
    for cs in scored:
        ep = build_evidence_packet(profiles[cs.acquirer_name], target, cs)
        evidence_packets.append(ep)
        evidence_map[cs.acquirer_name] = ep

    # Stage 4: LLM reranking
    console.rule("[bold blue]Stage 3: LLM Reranking")
    rerank_model = settings.default_rerank_model
    console.print(f"Reranking with {rerank_model}...")
    top_10, rerank_meta = rerank_candidates(
        evidence_packets, target, rerank_model,
        temperature=settings.rerank_temperature,
        max_retries=settings.max_retries,
    )
    console.print(f"Done in {rerank_meta.get('latency_ms', 0):.0f}ms")

    rerank_table = Table(title="LLM Top 10")
    rerank_table.add_column("Rank", style="dim")
    rerank_table.add_column("Acquirer")
    rerank_table.add_column("Score", justify="right")
    rerank_table.add_column("Conviction")
    rerank_table.add_column("Summary")
    for i, c in enumerate(top_10, 1):
        rerank_table.add_row(
            str(i), c.acquirer_name, str(c.likelihood_score),
            c.conviction_level, c.summary[:80] + "...",
        )
    console.print(rerank_table)

    # Stage 5: Rationale generation
    console.rule("[bold blue]Stage 4: Rationale Generation")
    rationale_model = settings.default_rationale_model
    console.print(f"Generating rationales with {rationale_model}...")
    rationales, rat_metas = generate_all_rationales(
        top_10, evidence_map, target, rationale_model,
        temperature=settings.rationale_temperature,
        max_retries=settings.max_retries,
    )
    total_ms = sum(m.get("latency_ms", 0) for m in rat_metas)
    console.print(f"Generated {len(rationales)} rationales in {total_ms:.0f}ms")

    for r in rationales:
        console.print(Panel(
            Markdown(render_rationale_markdown(r)),
            title=f"[bold]{r.acquirer_name}[/bold] — {r.conviction_level}",
            border_style="green" if r.conviction_level == "High" else "yellow",
        ))

    # Export
    out_dir = settings.output_dir / "rationales"
    export_json(top_10, rationales, out_dir / "latest.json")
    export_markdown(top_10, rationales, out_dir / "latest.md")
    console.print(f"\n[green]Outputs saved to {out_dir}[/green]")


if __name__ == "__main__":
    main()
