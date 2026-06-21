"""KPI tile helpers."""

from __future__ import annotations

import streamlit as st


def metric_row(items: list[dict]) -> None:
    """Render a row of st.metric tiles. Each item: {label, value, delta?}."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items, strict=False):
        col.metric(item["label"], item["value"], item.get("delta"))
