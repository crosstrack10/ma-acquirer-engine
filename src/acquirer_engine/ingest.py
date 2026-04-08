"""CSV ingestion and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from acquirer_engine.schemas import Transaction

REQUIRED_COLUMNS = [
    "transaction_id", "target_company", "acquirer", "sector", "sub_sector",
    "deal_year", "deal_quarter", "deal_type", "geography", "financing_type",
    "deal_size_mm", "target_revenue_mm", "target_ebitda_mm", "ebitda_margin_pct",
    "revenue_growth_pct", "ev_ebitda_multiple", "ev_revenue_multiple",
    "synergy_pct_of_deal", "outcome", "strategic_rationale_tags", "num_bidders",
    "days_to_close", "acquirer_type", "target_ownership_pre",
]


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
    return df


def transactions_from_dataframe(df: pd.DataFrame) -> list[Transaction]:
    """Convert a validated DataFrame into Transaction objects."""
    transactions: list[Transaction] = []
    for _, row in df.iterrows():
        tags_raw = row.get("strategic_rationale_tags", "")
        tags = [t.strip() for t in str(tags_raw).split("|") if t.strip()] if pd.notna(tags_raw) else []

        txn = Transaction(
            transaction_id=row["transaction_id"],
            target_company=row["target_company"],
            acquirer=row["acquirer"],
            sector=row["sector"],
            sub_sector=row["sub_sector"],
            deal_year=int(row["deal_year"]),
            deal_quarter=row["deal_quarter"],
            deal_type=row["deal_type"],
            geography=row["geography"],
            financing_type=row["financing_type"],
            deal_size_mm=float(row["deal_size_mm"]),
            target_revenue_mm=float(row["target_revenue_mm"]),
            target_ebitda_mm=float(row["target_ebitda_mm"]),
            ebitda_margin_pct=float(row["ebitda_margin_pct"]),
            revenue_growth_pct=float(row["revenue_growth_pct"]),
            ev_ebitda_multiple=float(row["ev_ebitda_multiple"]),
            ev_revenue_multiple=float(row["ev_revenue_multiple"]),
            synergy_pct_of_deal=float(row["synergy_pct_of_deal"]),
            outcome=row["outcome"],
            strategic_rationale_tags=tags,
            num_bidders=int(row["num_bidders"]),
            days_to_close=float(row["days_to_close"]) if pd.notna(row["days_to_close"]) else None,
            acquirer_type=row["acquirer_type"],
            target_ownership_pre=row["target_ownership_pre"],
        )
        transactions.append(txn)
    return transactions


def load_transactions(path: Path) -> list[Transaction]:
    df = load_dataframe(path)
    return transactions_from_dataframe(df)
