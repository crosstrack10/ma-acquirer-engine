"""Tests for CSV ingestion."""

from pathlib import Path

from acquirer_engine.ingest import load_transactions, load_dataframe

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ma_transactions_500.csv"


def test_load_dataframe_has_expected_columns():
    df = load_dataframe(DATA_PATH)
    assert "transaction_id" in df.columns
    assert "acquirer" in df.columns
    assert "deal_size_mm" in df.columns
    assert len(df) == 500


def test_load_transactions_returns_500():
    txns = load_transactions(DATA_PATH)
    assert len(txns) == 500


def test_transactions_have_parsed_tags():
    txns = load_transactions(DATA_PATH)
    tagged = [t for t in txns if t.strategic_rationale_tags]
    assert len(tagged) > 400  # vast majority have tags


def test_days_to_close_nullable():
    txns = load_transactions(DATA_PATH)
    closed = [t for t in txns if t.outcome == "Closed"]
    non_closed = [t for t in txns if t.outcome != "Closed"]
    assert all(t.days_to_close is not None for t in closed)
    assert all(t.days_to_close is None for t in non_closed)
