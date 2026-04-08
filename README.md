# M&A Acquirer Identification Engine

A hybrid acquirer ranking and rationale engine built for William Blair's AI Innovation Team take-home assessment. Given a target company profile and 500 historical M&A transactions, the system identifies the 10 most likely acquirers and generates a grounded, banker-ready one-page rationale for each.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- An OpenAI API key (and optionally an Anthropic key)
- The provided `ma_transactions_500.csv` dataset

### Setup

```bash
git clone https://github.com/crosstrack10/ma-acquirer-engine.git
cd ma-acquirer-engine

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Place the dataset (not included in repo)
mkdir -p data
cp /path/to/ma_transactions_500.csv data/

# Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (required) and ANTHROPIC_API_KEY (optional)
```

> **Note:** The CSV dataset is not included in the repository to keep data separate from code. Place `ma_transactions_500.csv` in the `data/` directory before running.

### Run

**Streamlit UI** (single command):
```bash
uv run streamlit run app.py
```

**CLI** (terminal output with rich formatting):
```bash
uv run python scripts/run_demo.py
```

### Demo

A recorded walkthrough of the Streamlit pipeline running end-to-end is available here:

> **[Loom Demo →](TODO_REPLACE_WITH_LOOM_URL)**

Sample output from a complete run (GPT-4o for both reranking and rationale) is included in [`sample_output/`](sample_output/) — 10 individual acquirer rationales, a summary report, and run metadata.

**Tests**:
```bash
uv run pytest tests/ -v
```

---

## How It Works

The system mirrors the actual banker workflow — define the target, screen the universe, narrow candidates, build evidence, generate rationale — but compresses it from hours to under 60 seconds.

```
CSV ─→ Validate ─→ Build Acquirer Profiles ─→ Deterministic Scoring (top 25)
         │                                            │
         │                                    Build Evidence Packets
         │                                            │
         │                                  LLM Rerank ─→ Final Top 10
         │                                            │
         │                              Generate Rationale (per acquirer)
         │                                            │
         └────────────────────────────── Banker-Ready Output + Exports
```

### Stage 1: Data Ingestion

Loads and validates the 500-row CSV. Parses all 24 fields, handles nulls in `days_to_close` for non-closed deals, and splits pipe-delimited rationale tags into structured lists.

### Stage 2: Acquirer Profiles

Aggregates transaction history into per-acquirer profiles: deal counts, sector distribution, deal size statistics, valuation multiples, geography footprint, rationale tag frequencies, close rates, and recency of activity.

### Stage 3: Deterministic Scoring

Ranks every acquirer using eight weighted features (all 0–1 normalized):

| Feature | Weight | What it measures |
|---------|--------|-----------------|
| Sector fit | 30% | Deals in target sector + adjacent sectors (weighted by adjacency) |
| Size fit | 20% | Gaussian similarity of median deal size to ~$200M target |
| EBITDA fit | 10% | Similarity of historical target EBITDA margins |
| Geography fit | 10% | Fraction of deals in target or broad geographies |
| Acquirer type fit | 10% | Strategic vs. financial sponsor alignment |
| Rationale tag fit | 10% | Jaccard overlap with target's strategic themes |
| Recency fit | 5% | Exponential decay — recent activity scores higher |
| Execution fit | 5% | Historical close rate |

This produces a ranked candidate list. The top 25 advance to LLM reranking.

#### How weights and thresholds were set

**Feature weights** reflect the relative importance of each signal for this specific assignment — a ~$200M Healthcare Services target:

- **Sector fit (30%)** — Highest weight because the assignment is sector-specific. An acquirer with zero relevant healthcare activity is unlikely regardless of size or geography match.
- **Size fit (20%)** — Second highest. Deal size capability is a strong signal of willingness and ability to transact at the target level. A firm that only does $10M tuck-ins is unlikely to pursue a $200M platform deal.
- **EBITDA / Geography / Acquirer type / Rationale tags (10% each)** — Important secondary signals that differentiate within the sector-eligible pool. Each contributes meaningful separation without dominating.
- **Recency / Execution (5% each)** — Useful tiebreakers. An acquirer active 3 years ago with perfect sector and size fit should still rank above a very recent but off-sector buyer.

Weights sum to 1.0 and are configurable in `configs/scoring.yaml`.

**Feature function parameters** were chosen to produce reasonable score distributions over the 500-row dataset:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Size fit σ | 0.8 × target size ($160M) | Acquirers with median deal sizes from ~$80M–$320M score above 0.5, matching M&A reality where buyers regularly do deals 0.5–2× their typical size |
| EBITDA margin σ | 8 percentage points | Margins within ±8pp of the 18% target score above 0.5, wide enough to avoid penalizing natural sub-sector variance |
| Recency decay λ | 0.3 | Halves the score every ~2.3 years. Activity within the last 2 years scores >0.55; 5+ years of inactivity scores <0.22 |
| Strategic type bias | 0.6 vs 0.5 (no preference set) | Mild boost for strategic acquirers in Healthcare Services where platform roll-ups dominate, without excluding PE firms with strong fit |
| Geography broad match | National, Multi-Regional → match any | Large strategic and PE acquirers operate nationally; geographic specificity should not penalize them |
| Sector adjacency | 0.7 / 0.4 / 0.2 / 0.1 tiers | Physician Groups and Behavioral Health are common adjacencies for Healthcare Services roll-ups; Medical Devices and Pharma operate in distinct buyer pools |

These are configurable in `src/acquirer_engine/features.py` (σ, λ, type bias) and `src/acquirer_engine/preprocess.py` (adjacency map).

### Stage 4: LLM Reranking

The top 25 candidates are sent to the LLM as structured **evidence packets** — not raw CSV dumps. Each packet contains the acquirer profile, up to 5 most relevant precedent deals, valuation summary, and the deterministic score breakdown. The LLM selects the final top 10 with likelihood scores, conviction levels, supporting signals, and risk flags.

### Stage 5: Rationale Generation

For each of the top 10 acquirers, a separate LLM call generates a one-page rationale with six sections:

1. **Acquirer Overview** — identity, strategy, M&A track record from the data
2. **Strategic Fit Thesis** — why this target makes sense for this acquirer specifically
3. **Precedent Activity** — specific prior transactions cited from the dataset
4. **Valuation Context** — EV/EBITDA and EV/Revenue comps from comparable closed deals
5. **Risk Flags** — at least 2 concrete risks (integration, financing, competition, etc.)
6. **Conviction Level** — High / Medium / Low with data-grounded reasoning

### Stage 6: Output

Results are displayed in the Streamlit UI or CLI, and automatically exported as JSON and Markdown to `outputs/rationales/`.

---

## Architecture Decisions

### Hybrid pipeline over pure LLM

A pure prompt-in/answer-out approach would dump 500 rows into the context window and hope for the best. Instead, this system does the heavy lifting deterministically (profiling, scoring, evidence selection) and uses the LLM only for what it's good at: reranking nuanced candidates and synthesizing prose from structured evidence.

**Why this matters:** The LLM never sees raw CSV. It sees curated evidence packets with pre-computed scores, relevant precedent deals, and valuation summaries. This produces more grounded, specific output and uses fewer tokens.

### Sector adjacency instead of hard filtering

The Healthcare Services sector has only 46 of 500 transactions, with ~12 in the $100–400M range. Hard-filtering to that sector would miss strong candidates with relevant adjacent-sector activity. The scoring system uses a sector adjacency map:

- **High adjacency (0.7):** Physician Groups, Behavioral Health, Home Health/Hospice
- **Moderate adjacency (0.4):** Health IT, Revenue Cycle
- **Low adjacency (0.2):** Dental
- **Minimal (0.1):** Medical Devices, Health Insurance, Pharma/Biotech

This lets acquirers with cross-sector healthcare M&A experience rank appropriately without artificially narrowing the candidate pool.

### Model-agnostic via LiteLLM

All LLM calls go through LiteLLM, making it trivial to swap between OpenAI and Anthropic models. Model selection is config-driven (`configs/models.yaml`), not hardcoded. Default is `openai/gpt-4o` but can be changed to `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-6`, or `anthropic/claude-haiku-4-5` from the UI sidebar or config.

### Structured output with Pydantic guardrails

Every LLM response is parsed and validated against strict Pydantic schemas:

- Exactly 10 acquirers in the reranked list
- No duplicate acquirer names
- Likelihood scores constrained to 0–100
- At least 2 risk flags per acquirer
- Valid conviction levels only (High / Medium / Low)

If validation fails, a **repair prompt** is issued with the specific errors, and the LLM retries (up to 3 attempts). This catches malformed JSON, missing fields, and business rule violations before they reach the user.

### Prompts designed for specificity

The prompts explicitly instruct the model to:

- Use **only** the provided evidence — no invented facts
- Cite **specific** transactions by name, size, and multiples
- Make each rationale **distinct** — no generic boilerplate
- Acknowledge when evidence is **sparse** rather than fabricating
- Write in **banker-grade prose**, not marketing language

---

## Assumptions

- **"Strong EBITDA margins"** is interpreted as ~18% based on the dataset's sector distribution, not a hard filter value
- **Sector adjacency weights** are judgment calls tuned to healthcare M&A logic — configurable in `src/acquirer_engine/preprocess.py`
- **Closed deals** drive valuation and comp context; all deal outcomes contribute to activity profiling
- **Both strategic and financial sponsor** acquirers are considered — the scoring applies a mild strategic bias when no preference is set, but PE firms with strong fit signals rank competitively
- The system ranks **plausibility based on historical precedent**, not certainty of real-world deal completion

---

## Known Limitations

- **Dataset scope:** 500 transactions is a fraction of the real M&A universe. Current acquirer strategy may differ from historical CSV behavior
- **Geography granularity:** Bucketed into regions (Northeast, Midwest, etc.), not company-footprint precise
- **No external enrichment:** Uses only the provided CSV. Public data (SEC filings, press releases) could improve acquirer overviews but was not required
- **Financial capacity:** Actual financing capacity, balance sheet data, and board appetite are not observed in the dataset
- **LLM quality variance:** Rationale quality depends on model choice. GPT-4o produces more detailed output than GPT-4o-mini; Anthropic models may format differently

---

## Handling Non-Determinism

LLM outputs vary between runs. This system manages that through:

1. **Low temperature:** 0.2 for reranking (precision), 0.4 for rationale (natural prose)
2. **Structured prompts:** Explicit JSON schema instructions reduce format variance
3. **Pydantic validation:** Rejects outputs that don't meet structural requirements
4. **Deterministic first pass:** The initial scoring and evidence selection are fully deterministic — only the LLM stages introduce variance
5. **Experiment logging:** Every run logs model, prompt version, latency, retry count, and outputs to `logs/experiments.jsonl` for comparison
6. **Output caching:** Final results are saved to `outputs/rationales/` so identical runs don't require re-generation

Outputs **will** vary between runs. The deterministic scoring ensures the candidate pool is stable; the LLM reranking may reorder within that pool.

---

## Project Structure

```
├── app.py                          # Streamlit UI entrypoint
├── pyproject.toml                  # uv-managed project config
├── .env.example                    # Required environment variables
├── configs/
│   ├── target_profile.yaml         # Default target: $200M Healthcare Services
│   ├── scoring.yaml                # Feature weights and top-N settings
│   ├── models.yaml                 # Model routing (rerank, rationale, benchmark)
│   └── prompts.yaml                # Prompt version labels
├── data/                           # NOT in repo — place CSV here locally
│   └── ma_transactions_500.csv     # Provided dataset (500 transactions)
├── src/acquirer_engine/
│   ├── settings.py                 # pydantic-settings config loader
│   ├── schemas.py                  # All Pydantic models (8 schemas)
│   ├── ingest.py                   # CSV loading and validation
│   ├── preprocess.py               # Sector adjacency map, normalization
│   ├── features.py                 # 8 feature functions (each 0–1)
│   ├── profiles.py                 # Acquirer profile aggregation
│   ├── scoring.py                  # Weighted deterministic scoring
│   ├── retrieve.py                 # Evidence packet builder
│   ├── rerank.py                   # LLM reranking orchestration
│   ├── rationale.py                # Rationale generation orchestration
│   ├── render.py                   # Markdown/JSON export
│   ├── metrics.py                  # Cost estimation
│   ├── tracking.py                 # Experiment logging (JSONL)
│   ├── evaluate.py                 # Quality metrics
│   ├── prompts/
│   │   ├── templates.py            # Rerank + rationale prompt templates
│   │   └── registry.py             # Prompt versioning
│   └── llm/
│       ├── base.py                 # Provider-neutral interface
│       ├── litellm_client.py       # LiteLLM wrapper (OpenAI + Anthropic)
│       ├── parsing.py              # JSON extraction + Pydantic validation
│       └── retry.py                # Retry with repair prompts
├── scripts/
│   └── run_demo.py                 # CLI demo runner
├── tests/
│   ├── test_ingest.py              # CSV loading tests
│   ├── test_features.py            # Feature function tests
│   ├── test_scoring.py             # Scoring engine tests
│   └── test_schemas.py             # Pydantic validation tests
├── sample_output/                  # Committed example run (GPT-4o)
│   ├── summary.md                  # Full report with all 10 acquirers
│   ├── summary.json                # Structured JSON output
│   ├── metadata.json               # Run metadata (model, latency, etc.)
│   └── individual/                 # One-page rationale per acquirer
│       ├── 01_advocate_health.md
│       ├── 02_upmc.md
│       └── ... (10 total)
└── outputs/                        # Generated reports (gitignored)
```

---

## What I Would Improve Given More Time

1. **Async rationale generation** — The 10 rationale LLM calls run sequentially (~30–60s). Using `asyncio.gather` with LiteLLM's async API would cut this to ~5–10s
2. **Richer evaluation harness** — Automated comparison across models and prompt versions with quality rubrics (distinctiveness, specificity, citation density)
3. **Historical holdout backtest** — Use past transactions as pseudo-targets and check whether the true acquirer appears in the top 10
4. **Arbitrary target profile support** — Full UI for custom sector, size, geography, and description input
5. **Side-by-side comparison mode** — Compare acquirer recommendations for two different target profiles
6. **Public data enrichment** — Augment acquirer overviews with Wikipedia/SEC context (noted as optional in the instructions)

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o / GPT-4o-mini |
| `ANTHROPIC_API_KEY` | No | Anthropic API key for Claude models |
| `LANGSMITH_API_KEY` | No | LangSmith key for optional tracing |
| `LANGSMITH_TRACING` | No | Set to `true` to enable LangSmith traces |
