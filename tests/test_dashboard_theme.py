"""Regression tests for dashboard/theme.py.

Two concerns:
1. Lock the primitive PALETTE — existing keys and hex values must never change.
2. Verify semantic tokens exist and style_fig still returns a figure.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# 1. PALETTE lock — existing keys + hex values are immutable contract.
# ---------------------------------------------------------------------------


def test_palette_has_all_required_keys() -> None:
    from dashboard.theme import PALETTE

    required = {"bg", "panel", "grid", "text", "muted", "accent", "up", "down", "series"}
    assert required <= set(PALETTE), f"Missing PALETTE keys: {required - set(PALETTE)}"


def test_palette_existing_hex_values_unchanged() -> None:
    """Exact hex strings are locked — this test must fail if any hex is mutated."""
    from dashboard.theme import PALETTE

    locked: dict[str, object] = {
        "bg": "#0e1117",
        "panel": "#161a25",
        "grid": "#2a2f3a",
        "text": "#e6e6e6",
        "muted": "#9aa0aa",
        "accent": "#4f9dff",
        "up": "#27c08a",
        "down": "#ff5d6c",
        "series": ["#4f9dff", "#27c08a", "#ffb454", "#c678dd", "#ff5d6c", "#56b6c2"],
    }
    for key, expected in locked.items():
        assert PALETTE[key] == expected, (
            f"PALETTE[{key!r}] changed: expected {expected!r}, got {PALETTE[key]!r}"
        )


def test_palette_series_length_unchanged() -> None:
    from dashboard.theme import PALETTE

    assert len(PALETTE["series"]) == 6, "PALETTE['series'] length must stay 6"


# ---------------------------------------------------------------------------
# 2. Semantic tokens — new keys exist with the expected types / values.
# ---------------------------------------------------------------------------


def test_semantic_status_tokens_exist_and_map_to_palette() -> None:
    from dashboard import theme
    from dashboard.theme import PALETTE

    assert PALETTE["up"] == theme.SUCCESS, "SUCCESS must map to PALETTE['up']"
    assert PALETTE["down"] == theme.WARN, "WARN must map to PALETTE['down']"
    assert PALETTE["accent"] == theme.INFO, "INFO must map to PALETTE['accent']"


def test_semantic_typography_tokens_exist() -> None:
    from dashboard import theme

    assert isinstance(theme.TITLE_SIZE, int) and theme.TITLE_SIZE > 0
    assert isinstance(theme.BODY_SIZE, int) and theme.BODY_SIZE > 0
    assert isinstance(theme.CAPTION_SIZE, int) and theme.CAPTION_SIZE > 0
    # Sizes should be in a sane typographic order
    assert theme.CAPTION_SIZE < theme.BODY_SIZE < theme.TITLE_SIZE


def test_semantic_layout_tokens_exist() -> None:
    from dashboard import theme

    assert isinstance(theme.CARD_RADIUS, str) and "px" in theme.CARD_RADIUS
    assert isinstance(theme.PANEL_BORDER, str) and len(theme.PANEL_BORDER) > 0
    assert isinstance(theme.CARD_PADDING, str) and len(theme.CARD_PADDING) > 0


# ---------------------------------------------------------------------------
# 3. style_fig contract — positional arg + existing defaults still work.
# ---------------------------------------------------------------------------


def test_style_fig_returns_figure_with_default_height() -> None:
    from dashboard.theme import style_fig

    fig = go.Figure()
    result = style_fig(fig)
    assert isinstance(result, go.Figure)
    assert result.layout.height == 360


def test_style_fig_accepts_custom_height() -> None:
    from dashboard.theme import style_fig

    fig = go.Figure()
    result = style_fig(fig, 260)
    assert result.layout.height == 260


def test_style_fig_applies_palette_colours() -> None:
    from dashboard.theme import PALETTE, style_fig

    fig = go.Figure()
    result = style_fig(fig)
    assert result.layout.font.color == PALETTE["text"]
    assert list(result.layout.colorway) == PALETTE["series"]


# ---------------------------------------------------------------------------
# 4. WCAG-AA contrast validation (computed, not just documented).
# ---------------------------------------------------------------------------


def _linearize(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)


def _contrast(c1: str, c2: str) -> float:
    l1, l2 = _luminance(c1), _luminance(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize(
    "fg_key, bg_key",
    [
        ("text", "bg"),
        ("text", "panel"),
        ("muted", "bg"),
        ("muted", "panel"),
        ("up", "bg"),
        ("up", "panel"),
        ("down", "bg"),
        ("down", "panel"),
        ("accent", "bg"),
        ("accent", "panel"),
    ],
)
def test_wcag_aa_normal_text_contrast(fg_key: str, bg_key: str) -> None:
    """All text-coloured tokens must achieve WCAG AA for normal text (≥ 4.5:1)."""
    from dashboard.theme import PALETTE

    fg = PALETTE[fg_key]
    bg = PALETTE[bg_key]
    ratio = _contrast(fg, bg)  # type: ignore[arg-type]
    assert ratio >= 4.5, (
        f"WCAG-AA fail: PALETTE[{fg_key!r}] ({fg}) vs PALETTE[{bg_key!r}] ({bg}) "
        f"= {ratio:.2f}:1 (need ≥ 4.5:1)"
    )


@pytest.mark.parametrize("series_idx", [1, 2, 3, 4, 5])
def test_wcag_aa_series_palette_contrast(series_idx: int) -> None:
    """Every categorical series colour must achieve WCAG AA for normal text (≥ 4.5:1).

    series[0] == accent and is already covered by test_wcag_aa_normal_text_contrast;
    series[1..5] are rendered as lines/markers in crypto_chart, vol_chart,
    yield_curve_chart, ml_gate_chart and regime_sharpe_chart. style_fig() gives every
    figure a transparent plot/paper background, so the effective backdrop is the app
    background (PALETTE['bg']), not the metric-card panel — hence we test against 'bg'.
    Locking these keeps a future palette change from silently dropping a series below
    the AA text threshold. (Current values: 6.33:1–10.72:1 vs bg.)
    """
    from dashboard.theme import PALETTE

    fg = PALETTE["series"][series_idx]
    bg = PALETTE["bg"]
    ratio = _contrast(fg, bg)  # type: ignore[arg-type]
    assert ratio >= 4.5, (
        f"WCAG-AA fail: PALETTE['series'][{series_idx}] ({fg}) vs PALETTE['bg'] ({bg}) "
        f"= {ratio:.2f}:1 (need ≥ 4.5:1)"
    )


def test_grid_colour_is_decorative_and_exempt_from_contrast_minimums() -> None:
    """grid (#2a2f3a) is a *deliberate*, documented contrast exemption — not a silent gap.

    It is used only for Plotly gridlines and the metric-panel border: purely decorative,
    non-text elements. WCAG 2.1 SC 1.4.11 (Non-text Contrast) requires 3:1 for
    *meaningful* graphical objects / UI components, but explicitly exempts elements that
    are decorative or not needed to understand the content — gridlines are the canonical
    example. grid is therefore held to neither the 4.5:1 (text) nor the 3:1 (meaningful
    object) bar.

    We assert it stays *below* 3:1 to encode that decision as a test: grid is meant to be
    a faint, decorative-only token (currently ~1.41:1 vs bg, ~1.30:1 vs panel). If a
    future change pushes it to ≥ 3:1 it has started to read as a meaningful boundary, and
    this test fails to force a conscious re-evaluation of whether it must now meet WCAG
    1.4.11. (The exact hex is separately locked by test_palette_existing_hex_values_unchanged.)
    """
    from dashboard.theme import PALETTE

    non_text_object_min = 3.0  # WCAG 2.1 SC 1.4.11 threshold for meaningful graphics
    for bg_key in ("bg", "panel"):
        ratio = _contrast(PALETTE["grid"], PALETTE[bg_key])  # type: ignore[arg-type]
        assert ratio < non_text_object_min, (
            f"grid ({PALETTE['grid']}) vs PALETTE[{bg_key!r}] ({PALETTE[bg_key]}) "
            f"= {ratio:.2f}:1, which now meets/exceeds the {non_text_object_min}:1 "
            "non-text bar. grid is a decorative-only token — reconsider whether it must "
            "now satisfy WCAG 1.4.11, or update this guard if the exemption still holds."
        )
