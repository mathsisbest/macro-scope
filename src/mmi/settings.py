"""Typed configuration, loaded from environment variables / `.env`.

Using pydantic-settings keeps config validated, documented and testable — no loose
``os.getenv`` calls scattered around the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root resolved from this file: src/mmi/settings.py -> parents[2] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration for the platform."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage.
    # ``duckdb_path`` is ALWAYS a local file (dev / CI / offline demo). MotherDuck — the
    # deployed shared store — is enabled separately via ``motherduck_database`` +
    # ``motherduck_token``; we never overload ``duckdb_path`` with an ``md:`` URL, so its
    # type stays a clean local Path.
    duckdb_path: Path = Field(default=REPO_ROOT / "data" / "mmi.duckdb", alias="MMI_DUCKDB_PATH")
    assets_path: Path = Field(default=REPO_ROOT / "config" / "assets.yml", alias="MMI_ASSETS_PATH")
    motherduck_database: str = Field(default="", alias="MMI_MOTHERDUCK_DATABASE")
    motherduck_token: str = Field(default="", alias="MOTHERDUCK_TOKEN")  # secret — never log/display

    # Data source keys
    fred_api_key: str = Field(default="", alias="FRED_API_KEY")
    coingecko_api_key: str = Field(default="", alias="COINGECKO_API_KEY")
    odds_api_key: str = Field(default="", alias="ODDS_API_KEY")

    # GenAI layer
    llm_provider: Literal["gemini", "groq", "claude"] = Field(
        default="gemini", alias="LLM_PROVIDER"
    )
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Behaviour
    log_level: str = Field(default="INFO", alias="MMI_LOG_LEVEL")

    @property
    def use_motherduck(self) -> bool:
        """True when the deployed/scheduled path should target MotherDuck."""
        return bool(self.motherduck_database and self.motherduck_token)

    def storage_label(self) -> str:
        """Human-safe storage description for logs/UI — never includes the token."""
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


# Convenience module-level singleton.
settings = get_settings()
