"""Educational glossary + tooltip helpers for the dashboard.

Provides one curated source of truth for domain-concept explanations:

- ``GLOSSARY``       — static term → plain-language definition dict.
- ``definition``     — pure, case-insensitive lookup (raises ``KeyError`` on unknown terms).
- ``tooltip_markdown`` — pure helper producing a bold term + "?" marker whose native HTML
  ``title`` attribute carries the definition (hover to read).
- ``glossary_markdown`` — pure helper rendering the full glossary as a markdown list
  (the mobile-friendly "static context" expander).
- ``glossary_tooltip`` — Streamlit wrapper for ``tooltip_markdown``.
- ``concept_expander`` — Streamlit wrapper for a titled expander of static context.

Honest framing rule: definitions explain what a term means AND what it does not mean
(e.g. "small positive values are still economically small"), matching the project's
honesty-over-leaderboard ethos. No definitions promise future returns.
"""

from __future__ import annotations

import html

import streamlit as st

# ---------------------------------------------------------------------------
# The glossary — single source of truth. Keys are machine slugs; definitions are
# short (2–3 sentence) plain-language explanations with honest caveats.
# ---------------------------------------------------------------------------

GLOSSARY: dict[str, str] = {
    "oos_r2": (
        "Out-of-sample R² — how much of the future return variation the model explains on "
        "data it never saw during training, measured by strict walk-forward evaluation. "
        "A value > 0 means the model beats the historical-mean forecast out-of-sample. "
        "Small positive values (like the ones here) are real but economically small."
    ),
    "ic": (
        "Information Coefficient (IC) — the rank correlation between the model's forecast "
        "and the realised return, evaluated out-of-sample. Positive IC means higher forecasts "
        "tended to be followed by higher returns on average; it says nothing about "
        "any individual call."
    ),
    "direction_accuracy": (
        "Directional accuracy ('hit rate') — the share of periods in which the forecast got "
        "the sign of the next-period return right (up vs down). 50% is no better than a coin "
        "flip, which is why the skill gate requires > 50% out-of-sample."
    ),
    "skill_gate": (
        "The project's formal honesty rule for deploying a forecast: a model earns a 'tilt' "
        "only if out-of-sample R² > 0 AND directional accuracy > 50%, measured strictly "
        "out-of-sample — never on data it was trained on."
    ),
    "shrinkage": (
        "Bayesian shrinkage — pulling each forecast toward the asset's long-run average "
        "return so that small-sample noise cannot produce extreme bets. It is a "
        "regularisation technique that keeps forecasts conservative, not a source of edge."
    ),
    "walk_forward": (
        "Walk-forward backtesting — at each step the model is trained only on past data, "
        "scored on the next period, then re-trained, mimicking how it would actually be used. "
        "This prevents look-ahead bias: the model never sees the future it is scored on."
    ),
    "vol_regime": (
        "Volatility regime — the market state defined by 20-day realised volatility cut into "
        "terciles (low / medium / high). Regimes are cut within each backtest window, so "
        "labels are not comparable across different windows."
    ),
    "cape": (
        "Shiller CAPE (cyclically adjusted price-to-earnings) — the price divided by the "
        "10-year average of inflation-adjusted earnings. A high CAPE suggests a historically "
        "expensive market. It is a slow-moving valuation gauge, not a short-term timing signal."
    ),
    "bootstrap_ci": (
        "Bootstrap confidence interval — the range of plausible values for a statistic "
        "(e.g. Sharpe) estimated by re-sampling the observed return sequence many times "
        "(a stationary block-bootstrap here). When a difference's CI excludes 0, the "
        "difference is 'distinguishable' from no effect at the chosen confidence level."
    ),
    "risk_parity": (
        "Risk parity vs equal weight — equal weight splits money evenly across assets; "
        "risk parity splits risk evenly, so lower-volatility assets like bonds get more "
        "weight. Risk parity is usually smoother; equal weight is simpler and more "
        "transparent. Neither is 'correct' — they are different bets, and the portfolio "
        "tab shows both side by side."
    ),
    "benchmark_6040": (
        "60/40 benchmark — a simple portfolio holding 60% equities (SPY) and 40% bonds "
        "(TLT), the classic baseline the strategies are compared against, over the same "
        "dates with the same rebalancing and costs so the comparison is like-for-like."
    ),
    "vol_rich_plus": (
        "vol_rich+ features — the models' expanded feature set: volatility metrics, "
        "cross-asset ratio spreads, and 75+ macro series, including breakeven inflation "
        "(the TLT vs TIP return spread, a market-implied inflation measure). More features "
        "do not mean better forecasts — out-of-sample evaluation decides."
    ),
    "yield_curve_spread": (
        "Yield-curve spread — the gap between long- and short-term Treasury yields "
        "(here 10-year minus 3-month). It often goes negative ('inverts') ahead of "
        "recessions — the Estrella-Mishkin signal — but it is a leading indicator with "
        "false positives, not a certainty."
    ),
    "sharpe": (
        "Sharpe ratio — average return earned per unit of volatility (risk) over the "
        "window. Higher is better, but it is a historical number: past Sharpe does not "
        "guarantee future risk-adjusted return, and short windows make it noisy."
    ),
}

#: Human-readable display titles keyed by slug — used when rendering a term chip so the
#: inline marker reads as "OOS R² ?" rather than "oos_r2 ?". Falls back to the slug.
TITLES: dict[str, str] = {
    "oos_r2": "OOS R²",
    "ic": "IC",
    "direction_accuracy": "Directional accuracy",
    "skill_gate": "Skill gate",
    "shrinkage": "Bayesian shrinkage",
    "walk_forward": "Walk-forward backtest",
    "vol_regime": "Volatility regime",
    "cape": "Shiller CAPE",
    "bootstrap_ci": "Bootstrap CI",
    "risk_parity": "Risk parity vs equal weight",
    "benchmark_6040": "60/40 benchmark",
    "vol_rich_plus": "vol_rich+ features",
    "yield_curve_spread": "Yield-curve spread",
    "sharpe": "Sharpe ratio",
}


def _normalize(term: str) -> str:
    """Normalise a term for case-insensitive lookup: lowercase, separators → underscore."""
    return (
        term.strip()
        .lower()
        .replace("²", "2")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("·", "_")
    )


def definition(term: str) -> str:
    """Return the glossary definition for *term* (case/separator-insensitive).

    Raises ``KeyError`` for unknown terms so wiring typos fail loudly rather than
    silently rendering an empty tooltip.
    """
    key = _normalize(term)
    if key not in GLOSSARY:
        raise KeyError(f"Unknown glossary term {term!r} — add it to GLOSSARY or fix the typo.")
    return GLOSSARY[key]


def known_terms() -> list[str]:
    """Sorted list of glossary slugs (stable order for tests + the glossary expander)."""
    return sorted(GLOSSARY)


def title(term: str) -> str:
    """Human-readable display title for *term* (falls back to the raw slug)."""
    return TITLES.get(_normalize(term), term)


# ---------------------------------------------------------------------------
# Pure render helpers (testable without Streamlit)
# ---------------------------------------------------------------------------


def tooltip_markdown(term: str, *, label: str | None = None) -> str:
    """Markdown for an inline 'Term ?' marker whose hover-tooltip carries the definition.

    ``label`` overrides the displayed text (defaults to the term's human title). The
    definition is HTML-escaped into the ``title`` attribute, so quotes in a definition
    cannot break the tag.
    """
    text = label if label is not None else title(term)
    body = html.escape(definition(term), quote=True)
    return f"**{text}** <span class='glossary-q' title=\"{body}\">?</span>"


def glossary_markdown(terms: list[str] | None = None) -> str:
    """Render the (optionally filtered) glossary as a markdown list for a static expander."""
    if terms is None:
        slugs = known_terms()
    else:
        slugs = []
        for t in terms:
            definition(t)  # validate — raises KeyError on typos, keeps wiring honest
            slugs.append(_normalize(t))
    return "\n".join(f"- **{title(slug)}** — {GLOSSARY[slug]}" for slug in slugs)


# ---------------------------------------------------------------------------
# Streamlit wrappers
# ---------------------------------------------------------------------------


def glossary_tooltip(term: str, *, label: str | None = None) -> None:
    """Render an inline 'Term ?' hover tooltip via ``st.markdown``."""
    st.markdown(tooltip_markdown(term, label=label), unsafe_allow_html=True)


def concept_expander(title: str, body: str, *, expanded: bool = False) -> None:
    """Render a titled expander of static context (a 'concept glossary')."""
    with st.expander(f"❔ {title}", expanded=expanded):
        st.markdown(body)
