"""Streamlit UI for the M&A Acquirer Identification Engine."""

import uuid
import streamlit as st
from pathlib import Path

from acquirer_engine.settings import get_settings, load_target_profile_yaml
from acquirer_engine.schemas import TargetProfile, EvidencePacket, ExperimentRecord
from acquirer_engine.ingest import load_transactions, transactions_from_dataframe, REQUIRED_COLUMNS
from acquirer_engine.profiles import build_acquirer_profiles
from acquirer_engine.scoring import score_candidates
from acquirer_engine.retrieve import build_evidence_packet
from acquirer_engine.rerank import rerank_candidates
from acquirer_engine.rationale import generate_all_rationales
from acquirer_engine.render import (
    render_rationale_markdown,
    render_summary_table,
    render_full_report,
    export_json,
    export_markdown,
    export_individual_rationales,
    export_run,
)
from acquirer_engine.tracking import log_experiment

st.set_page_config(page_title="M&A Acquirer Engine", layout="wide")
st.title("M&A Acquirer Identification Engine")

settings = get_settings()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")

st.sidebar.subheader("Data Source")
uploaded_csv = st.sidebar.file_uploader(
    "Upload custom CSV (optional)",
    type=["csv"],
    help="Upload your own M&A transaction dataset. Must have the same column schema as the default dataset. If not provided, the built-in dataset is used.",
)

model_options = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5",
]
rerank_model = st.sidebar.selectbox("Rerank Model", model_options, index=0)
rationale_model = st.sidebar.selectbox("Rationale Model", model_options, index=0)

st.sidebar.subheader("Target Profile")
target_defaults = load_target_profile_yaml()
sector = st.sidebar.text_input("Sector", value=target_defaults.get("sector", "Healthcare Services"))
deal_size = st.sidebar.number_input("Deal Size ($M)", value=target_defaults.get("deal_size_mm", 200.0))
ebitda_margin = st.sidebar.number_input("EBITDA Margin (%)", value=target_defaults.get("ebitda_margin_pct", 18.0))
geography = st.sidebar.text_input("Geography", value=target_defaults.get("geography", "Multi-Regional"))
tags_str = st.sidebar.text_input(
    "Rationale Tags (comma-separated)",
    value=", ".join(target_defaults.get("rationale_tags", [])),
)

target = TargetProfile(
    sector=sector,
    deal_size_mm=deal_size,
    ebitda_margin_pct=ebitda_margin,
    geography=geography,
    target_ownership=target_defaults.get("target_ownership", "Private"),
    description=target_defaults.get("description", ""),
    rationale_tags=[t.strip() for t in tags_str.split(",") if t.strip()],
)

run_btn = st.sidebar.button("Run Full Pipeline", type="primary")

# ── Main area ──────────────────────────────────────────────────────────────

if run_btn:
    # Stage 1: Load data
    with st.status("Loading and processing data...", expanded=True) as status:
        st.write("Loading transactions...")
        if uploaded_csv is not None:
            import pandas as pd
            df = pd.read_csv(uploaded_csv)
            missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
            if missing_cols:
                st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
                st.stop()
            str_cols = df.select_dtypes(include=["object", "string"]).columns
            df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
            transactions = transactions_from_dataframe(df)
            st.write(f"Loaded {len(transactions)} transactions from uploaded CSV")
        else:
            transactions = load_transactions(settings.data_path)
            st.write(f"Loaded {len(transactions)} transactions")

        st.write("Building acquirer profiles...")
        profiles = build_acquirer_profiles(transactions)
        st.write(f"Built {len(profiles)} acquirer profiles")

        st.write("Scoring candidates...")
        scored = score_candidates(profiles, target, settings.scoring_weights, top_n=settings.top_n_for_rerank)
        st.write(f"Top {len(scored)} candidates scored")

        st.write("Building evidence packets...")
        evidence_packets: list[EvidencePacket] = []
        evidence_map: dict[str, EvidencePacket] = {}
        for cs in scored:
            ep = build_evidence_packet(profiles[cs.acquirer_name], target, cs)
            evidence_packets.append(ep)
            evidence_map[cs.acquirer_name] = ep

        status.update(label="Data processing complete", state="complete")

    # Stage 2: LLM Reranking
    with st.status("Reranking candidates with LLM...", expanded=True) as status:
        st.write(f"Sending {len(evidence_packets)} candidates to {rerank_model}...")
        top_10, rerank_meta = rerank_candidates(
            evidence_packets, target, rerank_model,
            temperature=settings.rerank_temperature,
            max_retries=settings.max_retries,
        )
        st.write(f"Reranking complete — {rerank_meta.get('latency_ms', 0):.0f}ms")
        status.update(label="Reranking complete", state="complete")

    # Stage 3: Rationale generation
    with st.status("Generating rationales...", expanded=True) as status:
        st.write(f"Generating {len(top_10)} rationales with {rationale_model}...")
        rationales, rationale_metas = generate_all_rationales(
            top_10, evidence_map, target, rationale_model,
            temperature=settings.rationale_temperature,
            max_retries=settings.max_retries,
        )
        total_rat_ms = sum(m.get("latency_ms", 0) for m in rationale_metas)
        st.write(f"Rationale generation complete — {total_rat_ms:.0f}ms total")
        status.update(label="Rationale generation complete", state="complete")

    # Display results
    st.header("Top 10 Acquirers")
    summary_md = render_summary_table(top_10)
    st.markdown(summary_md)

    # Score breakdown chart
    st.subheader("Deterministic Score Breakdown")
    import pandas as pd
    score_data = []
    for c in top_10:
        ep = evidence_map.get(c.acquirer_name)
        if ep:
            row = {"Acquirer": c.acquirer_name, **ep.score_breakdown}
            score_data.append(row)
    if score_data:
        df = pd.DataFrame(score_data).set_index("Acquirer")
        st.bar_chart(df)

    # Individual rationales
    st.header("Acquirer Rationales")
    rationale_map = {r.acquirer_name: r for r in rationales}
    for i, c in enumerate(top_10, 1):
        r = rationale_map.get(c.acquirer_name)
        if r:
            with st.expander(f"{i}. {c.acquirer_name} — Score: {c.likelihood_score}, Conviction: {c.conviction_level}"):
                st.markdown(render_rationale_markdown(r))

    # Run metadata
    st.sidebar.subheader("Run Metadata")
    st.sidebar.write(f"Rerank model: {rerank_model}")
    st.sidebar.write(f"Rationale model: {rationale_model}")
    st.sidebar.write(f"Rerank latency: {rerank_meta.get('latency_ms', 0):.0f}ms")
    st.sidebar.write(f"Rationale latency: {total_rat_ms:.0f}ms")
    st.sidebar.write(f"Rerank retries: {rerank_meta.get('retry_count', 0)}")

    # Export buttons
    st.subheader("Export")
    report_md = render_full_report(top_10, rationales)
    st.download_button("Download Markdown Report", report_md, "acquirer_report.md", "text/markdown")

    import json
    report_json = json.dumps({
        "ranked_candidates": [c.model_dump() for c in top_10],
        "rationales": [r.model_dump() for r in rationales],
    }, indent=2, default=str)
    st.download_button("Download JSON", report_json, "acquirer_report.json", "application/json")

    # Auto-save to outputs
    out_dir = settings.output_dir / "rationales"
    export_json(top_10, rationales, out_dir / "latest.json")
    export_markdown(top_10, rationales, out_dir / "latest.md")
    export_individual_rationales(top_10, rationales, out_dir)

    # Save timestamped run
    run_metadata = {
        "rerank_model": rerank_model,
        "rationale_model": rationale_model,
        "rerank_latency_ms": rerank_meta.get("latency_ms", 0),
        "rationale_latency_ms": total_rat_ms,
        "rerank_retries": rerank_meta.get("retry_count", 0),
        "target_sector": target.sector,
        "target_deal_size_mm": target.deal_size_mm,
    }
    run_dir = export_run(top_10, rationales, settings.output_dir, run_metadata)

    # Log experiments
    run_id = str(uuid.uuid4())[:8]
    log_experiment(ExperimentRecord(
        run_id=f"{run_id}_rerank", model=rerank_model, prompt_version="v1",
        latency_ms=rerank_meta.get("latency_ms", 0),
        retry_count=rerank_meta.get("retry_count", 0),
        input_tokens=rerank_meta.get("input_tokens", 0),
        output_tokens=rerank_meta.get("output_tokens", 0),
    ))
    log_experiment(ExperimentRecord(
        run_id=f"{run_id}_rationale", model=rationale_model, prompt_version="v1",
        latency_ms=total_rat_ms,
        input_tokens=sum(m.get("input_tokens", 0) for m in rationale_metas),
        output_tokens=sum(m.get("output_tokens", 0) for m in rationale_metas),
    ))

    st.success(f"Outputs saved to {out_dir} | Run archived to {run_dir}")
else:
    st.info("Configure the target profile in the sidebar and click **Run Full Pipeline** to begin.")
    st.markdown("""
    ### How it works
    
    1. **Data Ingestion** — Loads 500 historical M&A transactions from the provided CSV
    2. **Feature Engineering** — Builds acquirer profiles with sector fit, deal size, geography, and more
    3. **Deterministic Scoring** — Ranks all acquirers using a weighted feature model
    4. **LLM Reranking** — Sends top 25 candidates to an LLM for intelligent reranking to the final top 10
    5. **Rationale Generation** — Generates a one-page banker-ready rationale for each of the top 10 acquirers
    """)
