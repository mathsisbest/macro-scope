"""Pydantic schema validation models for raw ingestion payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class YahooPriceRow(BaseModel):
    """Schema validation for raw Yahoo Finance price rows."""

    symbol: str = Field(..., description="Ticker symbol")
    date: str = Field(..., description="Trading date (YYYY-MM-DD)")
    open: float | None = Field(None, description="Open price")
    high: float | None = Field(None, description="High price")
    low: float | None = Field(None, description="Low price")
    close: float = Field(..., description="Close price")
    volume: float | None = Field(None, description="Trading volume")
    daily_return: float | None = Field(None, description="Daily simple return")


class FredObservationRow(BaseModel):
    """Schema validation for raw FRED macroeconomic observation rows."""

    series_id: str = Field(..., description="FRED series code")
    date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    value: float = Field(..., description="Macroeconomic indicator value")


class WorldBankIndicatorRow(BaseModel):
    """Schema validation for raw World Bank indicator rows."""

    country: str = Field(..., description="ISO country code")
    indicator_id: str = Field(..., description="World Bank indicator code")
    date: str = Field(..., description="Year (YYYY)")
    value: float = Field(..., description="Indicator value")
    source: str = Field(..., description="Source identifier")


class OddsRow(BaseModel):
    """Schema validation for raw prediction market / sports odds rows."""

    event_id: str = Field(..., description="Unique event identifier")
    market_id: str = Field(..., description="Market / outcome identifier")
    title: str = Field(..., description="Event description or title")
    outcome: str = Field(..., description="Outcome name (e.g. Yes/No, Team A)")
    implied_probability: float = Field(..., description="Implied probability [0.0, 1.0]")
    decimal_odds: float = Field(..., description="Decimal odds (e.g. 2.50)")
    timestamp: str = Field(..., description="Observation timestamp")
