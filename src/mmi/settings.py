"""Typed configuration, loaded from environment variables / `.env`.

Using pydantic-settings keeps config validated, documented and testable — no loose
``os.getenv`` calls scattered around the codebase.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root resolved from this file: src/mmi/settings.py -> parents[2] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

_settings_log = logging.getLogger("mmi.settings")


class Settings(BaseSettings):
    """All runtime configuration for the platform."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Storage.
    # ``duckdb_path`` is ALWAYS a local file (dev / CI / offline demo). MotherDuck — the
    # deployed shared store — is enabled separately via ``motherduck_database`` +
    # ``motherduck_token``; we never overload ``duckdb_path`` with an ``md:`` URL, so its
    # type stays a clean local Path.
    duckdb_path: Path = Field(default=REPO_ROOT / "data" / "mmi.duckdb", alias="MMI_DUCKDB_PATH")
    assets_path: Path = Field(default=REPO_ROOT / "config" / "assets.yml", alias="MMI_ASSETS_PATH")
    events_path: Path = Field(default=REPO_ROOT / "config" / "events.yml", alias="MMI_EVENTS_PATH")
    # The static Parquet snapshot of the marts schema: `mmi snapshot` writes it, and the public
    # demo dashboard reads from it (no DB, no secrets) when MMI_SNAPSHOT_DIR is set.
    snapshot_dir: Path = Field(default=REPO_ROOT / "data" / "public", alias="MMI_SNAPSHOT_DIR")
    # When true, the dashboard reads the Parquet snapshot in ``snapshot_dir`` IN-PROCESS instead of
    # opening DuckDB/MotherDuck — the public, secret-free deploy path (Streamlit Community Cloud
    # sets it). `mmi snapshot` writes the files; the accessors and their SQL are unchanged.
    snapshot_mode: bool = Field(default=False, alias="MMI_SNAPSHOT_MODE")
    motherduck_database: str = Field(default="", alias="MMI_MOTHERDUCK_DATABASE")
    # secret — never log or display
    motherduck_token: str = Field(default="", alias="MOTHERDUCK_TOKEN")

    # Data source keys
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")
    odds_api_key: str = Field(default="", alias="ODDS_API_KEY")

    # GenAI layer
    llm_provider: Literal["gemini", "groq", "claude"] = Field(
        default="gemini", alias="LLM_PROVIDER"
    )
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # GenAI behaviour — Gemini 3.x "thinking" effort. medium is Google's default and best
    # quality/cost trade-off; low is faster/cheaper, high for hard reasoning. gemini-only.
    gemini_thinking_level: Literal["low", "medium", "high"] = Field(
        default="low", alias="GEMINI_THINKING_LEVEL"
    )

    # Behaviour
    log_level: str = Field(default="INFO", alias="MMI_LOG_LEVEL")
    # Bootstrap resamples for the portfolio backtest (GO_LIVE_PLAN D1). Lower for fast local
    # tuning; never commit a data/public snapshot produced with n_boot < 2000.
    portfolio_n_boot: int = Field(default=2000, alias="MMI_PORTFOLIO_N_BOOT")
    # Fail-loud size cap (bytes) for snapshot Parquet exports (GO_LIVE_PLAN D6): prevents
    # accidental commits of oversized data. Remedy is a new downsampled dbt mart, never a
    # trimmed export.
    snapshot_max_bytes: int = Field(default=12_000_000, alias="MMI_SNAPSHOT_MAX_BYTES")

    @field_validator("portfolio_n_boot", "snapshot_max_bytes", mode="before")
    @classmethod
    def _defensive_positive_int(cls, raw: Any, info: ValidationInfo) -> int:
        """Warn and fall back to the field default for non-integer / non-positive env values.

        These knobs were historically parsed defensively at the call site; a fat-fingered
        env value must warn and keep the run going, never crash the overnight cron.
        """
        field = cls.model_fields[info.field_name or ""]
        try:
            parsed = int(raw)
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 0:
            return parsed
        _settings_log.warning(
            "%s=%r is not a valid positive integer; falling back to default %d",
            field.alias,
            raw,
            field.default,
        )
        return field.default

    @property
    def use_motherduck(self) -> bool:
        """True when the deployed/scheduled path should target MotherDuck."""
        return bool(self.motherduck_database and self.motherduck_token)

    def storage_label(self) -> str:
        """Human-safe storage description for logs/UI — never includes the token."""
        if self.snapshot_mode:
            return f"Parquet snapshot · {self.snapshot_dir.name}/"
        if self.use_motherduck:
            return f"MotherDuck · {self.motherduck_database}"
        return f"DuckDB · {self.duckdb_path.name}"

    def ensure_dirs(self) -> None:
        """Create local directories the pipeline writes to (no-op on MotherDuck)."""
        if not self.use_motherduck:
            self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton accessor."""
    return Settings()


def load_assets(path: Path | None = None) -> dict[str, Any]:
    """Load the declarative asset universe from ``config/assets.yml``."""
    path = path or get_settings().assets_path
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


DEFAULT_EVENTS: list[dict[str, str]] = [
    {
        "date": "2008-09-15",
        "label": "Lehman Collapse",
        "description": "Lehman Brothers files for bankruptcy.",
        "category": "crisis",
    },
    {
        "date": "2020-03-23",
        "label": "COVID Market Low",
        "description": "S&P 500 bottoms at the peak of pandemic panic.",
        "category": "market_shock",
    },
    {
        "date": "2022-03-16",
        "label": "Fed First Rate Hike",
        "description": "Federal Reserve initiates tightening cycle.",
        "category": "monetary_policy",
    },
    {
        "date": "2023-03-10",
        "label": "SVB Collapse",
        "description": "Silicon Valley Bank fails.",
        "category": "crisis",
    },
]


def load_events(path: Path | None = None) -> dict[str, Any]:
    """Load the declarative major market and macro events from ``config/events.yml``."""
    target = path or get_settings().events_path
    try:
        if target and Path(target).exists():
            with open(target, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
                if isinstance(loaded, dict) and "events" in loaded:
                    return loaded
    except Exception as exc:  # noqa: BLE001
        _settings_log.warning(
            "Failed to load events from %s (%s); using fallback defaults", target, exc
        )
    return {"events": DEFAULT_EVENTS}


# Convenience module-level singleton.
settings = get_settings()
