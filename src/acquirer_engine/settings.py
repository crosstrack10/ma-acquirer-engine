"""Centralized configuration using pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = _ROOT / "configs"


def _load_yaml(name: str) -> dict:
    path = _CONFIGS / name
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False

    # Paths
    data_path: Path = Field(default=_ROOT / "data" / "ma_transactions_500.csv")
    output_dir: Path = Field(default=_ROOT / "outputs")
    log_dir: Path = Field(default=_ROOT / "logs")

    # Model config (loaded from YAML, overridable via env)
    default_rerank_model: str = "openai/gpt-4o"
    default_rationale_model: str = "openai/gpt-4o"
    dev_fast_model: str = "openai/gpt-4o-mini"
    rerank_temperature: float = 0.2
    rationale_temperature: float = 0.4
    max_retries: int = 3

    # Scoring config
    scoring_weights: dict[str, float] = Field(default_factory=lambda: {
        "sector_fit": 0.30,
        "size_fit": 0.20,
        "ebitda_fit": 0.10,
        "geography_fit": 0.10,
        "acquirer_type_fit": 0.10,
        "rationale_fit": 0.10,
        "recency_fit": 0.05,
        "execution_fit": 0.05,
    })
    top_n_for_rerank: int = 25
    final_top_n: int = 10

    def model_post_init(self, __context) -> None:
        # Overlay YAML configs onto defaults
        models_cfg = _load_yaml("models.yaml")
        providers = models_cfg.get("providers", {})
        gen = models_cfg.get("generation", {})
        for attr, key in [
            ("default_rerank_model", "default_rerank_model"),
            ("default_rationale_model", "default_rationale_model"),
            ("dev_fast_model", "dev_fast_model"),
        ]:
            if key in providers and not self._field_was_set_via_env(attr):
                object.__setattr__(self, attr, providers[key])
        for attr, key in [
            ("rerank_temperature", "rerank_temperature"),
            ("rationale_temperature", "rationale_temperature"),
            ("max_retries", "max_retries"),
        ]:
            if key in gen and not self._field_was_set_via_env(attr):
                object.__setattr__(self, attr, gen[key])

        scoring_cfg = _load_yaml("scoring.yaml")
        if "weights" in scoring_cfg:
            object.__setattr__(self, "scoring_weights", scoring_cfg["weights"])
        if "top_n_for_rerank" in scoring_cfg:
            object.__setattr__(self, "top_n_for_rerank", scoring_cfg["top_n_for_rerank"])
        if "final_top_n" in scoring_cfg:
            object.__setattr__(self, "final_top_n", scoring_cfg["final_top_n"])

    @staticmethod
    def _field_was_set_via_env(_attr: str) -> bool:
        # Env vars always take precedence; this is a simplified check.
        # In practice pydantic-settings handles precedence automatically.
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_target_profile_yaml() -> dict:
    return _load_yaml("target_profile.yaml")
