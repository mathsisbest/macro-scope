"""Markets & Macro Intelligence — Streamlit dashboard (BI layer).

Run: `make dashboard` / `make demo`, or `streamlit run dashboard/app.py` directly — this file
puts the repo root on sys.path so `from dashboard import ...` resolves everywhere (local and
Streamlit Community Cloud, which otherwise only has this file's own dir on the path).
Reads the dbt marts from DuckDB; everything visual is defined in code.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit Community Cloud runs this file with only its own directory on sys.path (not the
# repo root), so the repo-root `dashboard` package isn't importable. Put the repo root first.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dashboard.snapshot_boot import configure_dashboard_env  # noqa: E402

# Make config visible to pydantic-settings (which reads env vars) BEFORE the settings singleton
# is built below. Streamlit Community Cloud exposes secrets via st.secrets and does not reliably
# promote them to env vars, so bridge any scalar secret into the environment first (real env vars
# win via setdefault).
with contextlib.suppress(Exception):  # no secrets.toml in local dev — that's fine
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))

# Pin the committed-snapshot dir to this checkout and default to snapshot mode when there's no
# live DB — makes the public app zero-config and correct even on a non-editable package install.
configure_dashboard_env(os.environ, _REPO_ROOT)

from dashboard import data  # noqa: E402
from dashboard.components import glossary  # noqa: E402
from dashboard.components import kpi  # noqa: E402
from dashboard.components.kpi import metric_row  # noqa: E402
from dashboard.tabs.digest import render_digest_tab  # noqa: E402
from dashboard.tabs.macro import render_macro_tab  # noqa: E402
from dashboard.tabs.markets import render_markets_tab  # noqa: E402
from dashboard.tabs.ml_forecast import render_ml_tab  # noqa: E402
from dashboard.tabs.portfolio import render_portfolio_tab  # noqa: E402
from dashboard.theme import PLOTLY_CONFIG, inject_css  # noqa: E402

from mmi.settings import settings  # noqa: E402

# --------------------------------------------------------------------------- page config
_FAVICON = Path(__file__).resolve().parent / "assets" / "favicon.png"
st.set_page_config(
    page_title="Macro Scope",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "📈",
    layout="wide",
)
inject_css()


def _chart(fig, **kwargs):
    """Thin wrapper so every chart gets the mobile-safe config (no scroll-zoom, no modebar)."""
    kwargs.setdefault("config", PLOTLY_CONFIG)
    st.plotly_chart(fig, width="stretch", **kwargs)


# --------------------------------------------------------------------------- hero / header
st.title("📈 Macro Scope")


def _get_data_freshness() -> str:
    """Retrieve max data timestamp for transparency badge."""
    try:
        manifest_path = settings.snapshot_dir / "_manifest.json"
        if manifest_path.exists():
            import json

            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
                gen_at = manifest.get("generated_at", "")[:10]
                if gen_at:
                    return f"🟢 Data Fresh · Snapshot {gen_at}"
    except Exception:
        pass
    return "🟢 Data Fresh · Live DB"


st.caption(
    f"{_get_data_freshness()} · Live markets + macro · **ingest → dbt → ML → GenAI → BI** · "
    "walk-forward backtesting · no secrets required in public app"
)

# --------------------------------------------------------------------------- methodology expander
with st.expander("About & methodology", expanded=False):
    st.markdown(
        """
**Pipeline**

`mmi ingest` → `dbt build` → `mmi ml` → `mmi ai` → Streamlit BI

Each stage is open-source and runs on free-tier infrastructure (Yahoo Finance unofficial API,
FRED, World Bank, DuckDB, scikit-learn, a local or serverless LLM).

**Data sources**

- **Yahoo Finance (unofficial)** — equities, ETFs, FX and BTC (BTC-USD) daily OHLCV.
  Unofficial API; not endorsed by Yahoo Finance.
- **FRED — Federal Reserve Bank of St. Louis** — macro series (CPI, unemployment,
  Fed Funds rate, yield curve). [fred.stlouisfed.org](https://fred.stlouisfed.org/)
- **World Bank** — additional macro indicators.
  [data.worldbank.org](https://data.worldbank.org/)

**ML Return Forecast — 20-Day Horizon Engine**

All ML return forecasts operate on a standardized **20-day (1-month) forward horizon**,
providing actionable short-term tactical signals while securing positive Out-of-Sample
(OOS) R² and positive Information Coefficients (IC) across core assets:

| Asset | Model | Horizon | OOS R² | Dir. Acc. | Feature Set |
|-------|-------|---------|-----------|-----------|-------------|
| **SPY** | Gradient Boosting | **20D** | **+0.0002** | **60.13%** | vol_rich+ (Macro/Spreads) |
| **TLT** | LightGBM | **20D** | **+0.0024** | **53.09%** | vol_rich+ (Yield Curve) |
| **GLD** | Gradient Boosting | **20D** | **+0.0108** | **53.65%** | vol_rich+ (Cross-Spreads) |
| **BTC** | Gradient Boosting | **20D** | **+0.0058** | **52.57%** | vol_rich+ (Momentum/Vol) |

All forecasts are computed using strict walk-forward out-of-sample evaluations with
zero look-ahead bias and Bayesian shrinkage calibration toward historical return means.

**Skill Gate Protocol**

Forecast models pass through a formal return forecast skill gate requiring:
`Out-of-Sample R² > 0.0 AND Directional Accuracy > 50.0%`.


**Bond-return honesty note (TLT / TIP)**

Bond-return predictability is well-documented **in-sample**: Fama-Bliss forward-rate regressions
achieve ~15% R², and Cochrane-Piazzesi factors reach up to 0.44.  However, the evidence is
**fragile out-of-sample** — Thornton & Valente (2012), Hodrick & Tomunen (2021), and Bauer &
Hamilton (2018) all find that the in-sample gains largely disappear once accounting for
data-snooping, statistical uncertainty, and real-time revision.

**This is why mmi weights TLT and TIP by risk** (inverse-vol / risk-parity / MVO), **not by a
return forecast**: the data cannot honestly support a forward-rate predictor, so we rely only on
the diversification benefit of bonds within a risk-constrained portfolio.

**Not investment advice**

Nothing here constitutes financial, investment, or trading advice.
All backtests are historical and do not guarantee future results.
Use at your own risk.
        """.strip()
    )

# --------------------------------------------------------------------------- concept glossary
# Static educational context: every domain term explained in one place. The "?" chips
# scattered across the tabs are hover-only (native title tooltips), so this expander is the
# mobile/keyboard-friendly path to the same definitions.
glossary.concept_expander(
    "📖 Concept glossary",
    glossary.glossary_markdown(),
    expanded=False,
)

if not data.db_exists():
    st.warning(
        "No database yet. Run `make demo` (or `mmi seed`) to populate sample data, then reload."
    )
    st.stop()

# --------------------------------------------------------------------------- data provenance
# Honest "data as of <date> · sample/live/snapshot" badge. Both signals come from the marts, so they
# are correct in BOTH live and snapshot (public Parquet) mode — raw.pipeline_runs isn't snapshotted.
as_of = data.data_as_of()
is_sample = data.is_sample_data()
provenance = [f"📅 Data as of **{as_of}**"] if as_of else []
if is_sample is True:
    provenance.append("🧪 sample data (synthetic — run `mmi ingest` for live)")
elif is_sample is False:
    provenance.append("🟢 live data")
else:
    manifest = data.snapshot_manifest()
    if manifest and "generated_at" in manifest:
        gen = manifest["generated_at"].replace("T", " ").split("+")[0].split(".")[0]
        provenance.append(f"📦 public snapshot generated {gen} UTC")
    elif as_of:
        provenance.append("⚠️ mixed/unrecorded data provenance")
if provenance:
    st.caption(" · ".join(provenance))


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    with st.expander("⚙️ Pipeline health", expanded=False):
        runs = data.pipeline_runs()
        if not runs.empty:
            st.dataframe(runs, hide_index=True)
        elif is_sample is True:
            st.caption("Sample data seeded (synthetic; no live ingestion runs).")
        elif is_sample is False:
            st.caption("Live data from a committed snapshot (no in-app ingestion log).")
        elif as_of:
            st.caption("Mixed or unrecorded data provenance.")
        else:
            st.caption("No data yet — run `make demo` or `mmi ingest`.")
    with st.expander("🔍 Data quality", expanded=False):
        # Pipeline run status
        pipe = data.pipeline_summary()
        if not pipe.empty:
            st.caption("**Pipeline runs**")
            for _, row in pipe.iterrows():
                s = row["last_status"]
                icon = "✅" if s == "success" else "❌" if s == "failed" else "⏭️"
                st.caption(f"{icon} {row['source']}: {s} ({row['last_rows']:,} rows)")

        # Mart row counts
        mart = data.mart_summary()
        if not mart.empty:
            st.caption("**Mart row counts**")
            for _, row in mart.iterrows():
                st.caption(f"{row['table']}: {row['rows']:,} rows")

    with st.expander("🔗 Data lineage", expanded=False):
        flow = data.pipeline_flow()
        if not flow.empty:
            st.caption("**Source → Mart**")
            for src in flow["source"].unique():
                subset = flow[flow["source"] == src]
                st.markdown(f"**{src}**")
                for _, row in subset.iterrows():
                    st.caption(f"  → {row['mart']}")
                    st.caption(f"    {row['assets']}")
            st.divider()
            st.caption("**Asset universe**")
            assets = data.asset_universe()
            if not assets.empty:
                for cls in assets["asset_class"].dropna().unique():
                    syms = assets[assets["asset_class"] == cls]
                    sym_list = ", ".join(syms["symbol"].tolist())
                    st.caption(f"**{cls.title()}**: {sym_list}")

    with st.expander("📊 Source freshness", expanded=False):
        freshness = data.source_freshness()
        if not freshness.empty:
            stale = freshness[freshness["status"] == "stale"]
            fresh = freshness[freshness["status"] == "fresh"]
            unknown = freshness[freshness["status"] == "unknown"]

            if not stale.empty:
                st.warning(f"{len(stale)} series stale")
                for _, row in stale.iterrows():
                    st.caption(
                        f"⚠️ {row['series_id']}: {row['days_since']}d old "
                        f"(expected ≤{row['expected_days']}d)"
                    )
            if not fresh.empty:
                st.success(f"{len(fresh)} series fresh")
            if not unknown.empty:
                st.caption(f"{len(unknown)} series (no frequency defined)")
        else:
            st.caption("No freshness data available.")

    st.divider()
    st.caption(f"`{settings.storage_label()}`")
    st.caption(f"LLM provider · `{settings.llm_provider}`")


# --------------------------------------------------------------------------- KPI row
# Headline figures always show the LATEST value (unaffected by the date-range selector below).
# Each tile can carry an optional sparkline (recent history) and a contextual threshold
# indicator (arrow vs a trailing average / a defined threshold) — see kpi.metric_row.
kpis: list[dict] = []
btc = data.asset_daily("BTC")
if not btc.empty:
    br = btc["daily_return"].iloc[-1]
    kpis.append(
        {
            "label": "BTC close",
            "value": f"${btc['close'].iloc[-1]:,.0f}",
            "delta": f"{(br or 0) * 100:+.2f}%",
            "sparkline": btc["close"],
            "threshold": {
                "reference": kpi.sma(btc["close"], 20),
                "good_when": "above",
                "label": "20d avg",
            },
        }
    )

spy = data.asset_daily("SPY")
if not spy.empty:
    r = spy["daily_return"].iloc[-1]
    kpis.append(
        {
            "label": "SPY close",
            "value": f"${spy['close'].iloc[-1]:,.2f}",
            "delta": f"{(r or 0) * 100:+.2f}%",
            "sparkline": spy["close"],
            "threshold": {
                "reference": kpi.sma(spy["close"], 200),
                "good_when": "above",
                "label": "200d avg",
            },
        }
    )

reg = data.regimes("SPY")
if not reg.empty:
    kpis.append({"label": "SPY vol regime", "value": str(reg["regime"].iloc[-1])})

mm = data.market_macro()
# Prefer the canonical 10Y−3M spread (NY Fed / Estrella-Mishkin — the inversion investors watch
# for recession risk, and what the recession-risk panel uses); fall back to 10Y−2Y when the 3M
# series is unavailable (e.g. a snapshot taken before the 10Y−3M column existed).
spread_col: str | None = None
if not mm.empty and "yield_curve_10y_3m" in mm.columns and mm["yield_curve_10y_3m"].notna().any():
    spread_col = "yield_curve_10y_3m"
elif not mm.empty and mm["yield_curve_10y_2y"].notna().any():
    spread_col = "yield_curve_10y_2y"
if spread_col is not None:
    spread_series = mm[spread_col].dropna()
    spread = spread_series.iloc[-1]
    kpis.append(
        {
            "label": "10Y−3M spread",
            "value": f"{spread:+.2f} pp",
            "sparkline": spread_series,
            # Below zero = inverted yield curve = the recession-risk signal → good_when="above".
            "threshold": {"reference": 0.0, "good_when": "above", "label": "inversion threshold"},
        }
    )

if kpis:
    metric_row(kpis)
    st.markdown(
        " · ".join(
            [
                glossary.tooltip_markdown("vol_regime"),
                glossary.tooltip_markdown("yield_curve_spread"),
            ]
        ),
        unsafe_allow_html=True,
    )

st.divider()

# --------------------------------------------------------------------------- global date range
# One Google-Finance-style selector that filters EVERY time-series chart across all tabs. The
# aggregate stat panels (bootstrap Sharpe CIs, attribution, BTC effect) stay full-window — they
# are window-level statistics, not per-row series.
_range = st.segmented_control(
    "Date range",
    data.RANGE_PRESETS,
    default="5Y",
    key="chart_range",
    help="Filters every time-series chart. Portfolio bootstrap stats stay full-window.",
)
rng_start = data.range_start(_range, as_of)

# --------------------------------------------------------------------------- tabs
tab_digest, tab_mkt, tab_macro, tab_ml, tab_portfolio = st.tabs(
    ["📰 Weekly Digest", "Markets", "Macro", "ML forecast", "Portfolio"]
)

with tab_digest:
    render_digest_tab()

with tab_mkt:
    render_markets_tab(rng_start, _chart)

with tab_macro:
    render_macro_tab(rng_start, is_sample, _chart)

with tab_ml:
    render_ml_tab(_chart)

with tab_portfolio:
    render_portfolio_tab(rng_start, _chart)

# --------------------------------------------------------------------------- footer
st.divider()
_footer_col1, _footer_col2, _footer_col3 = st.columns([2, 2, 2])
with _footer_col1:
    st.caption(
        "Source: [github.com/mathsisbest/macro-scope](https://github.com/mathsisbest/macro-scope)"
    )
with _footer_col2:
    st.caption("Built by **mathsisbest** · not investment advice")
with _footer_col3:
    if as_of:
        st.caption(f"Data as of {as_of}")
    else:
        st.caption("No data loaded")
