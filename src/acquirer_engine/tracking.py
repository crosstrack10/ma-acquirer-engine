"""Experiment logging — local JSONL and optional LangSmith."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from acquirer_engine.schemas import ExperimentRecord
from acquirer_engine.settings import get_settings

logger = logging.getLogger(__name__)


def log_experiment(record: ExperimentRecord, log_dir: Path | None = None) -> None:
    if log_dir is None:
        log_dir = get_settings().log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "experiments.jsonl"
    with open(log_path, "a") as f:
        f.write(record.model_dump_json() + "\n")
    logger.info(f"Logged experiment {record.run_id} to {log_path}")
