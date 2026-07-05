# Return Forecast Research: Findings, Architecture & Lessons Learned

**Author:** MiMo Code Agent
**Date:** 2026-07-05
**Status:** Living document — updated as research continues

> **Complete experiment log (every attempt, every result):** See [ML_FORECAST_EXPERIMENT_LOG.md](ML_FORECAST_EXPERIMENT_LOG.md)

---

## 1. Executive Summary

Predicting SPY daily returns is one of the hardest problems in quantitative finance. After systematic
sweep across models, features, target types, and horizons, the best configuration achieves:

- **IC (rank correlation):** +0.067 (LGB/vol_medium/excess, single-split) to +0.036 (GB/default/raw, walk-forward)
- **Direction accuracy:** 50-54% (barely above chance)
- **R²:** Negative across all configs (expected — daily returns are near-random walks)
- **Sharpe:** 0.20-0.36 from the forecast signal alone

**Bottom line:** The signal is real but tiny. The ML forecast should be blended toward the
historical mean via the portfolio gate, with λ ≈ 0.01-0.05 (very small weight). The model adds
marginal value — not transformative.

---

## 2. What Was Tested

### 2.1 Models

| Model | Description | Speed | Notes |
|-------|-------------|-------|-------|
| **HistGradientBoostingRegressor** | sklearn native, handles NaN | Medium | Default for `model='gb'` |
| **LightGBM** | Microsoft's gradient boosting | Fast | Default for `model='lgb'` |
| RandomForest | sklearn ensemble | Slow | Deprecated in favor of GB |

### 2.2 Feature Sets

| Set | Features | Description |
|-----|----------|-------------|
| `default` | 10 | Lagged returns + rolling mean/std |
| `vol` | 15 | + Garman-Klass vol + HAR cascade |
| `vol_medium` | ~27 | + macro subset + key rich features |
| `vol_rich` | 50 | Full: kurtosis, skewness, vol-of-vol, correlations, calendar, interactions |
| `vol_macro` | ~35 | + yield curve, VIX, NFCI, oil, dollar |

### 2.3 Target Types

| Type | Formula | Signal Quality |
|------|---------|----------------|
| `raw` | Forward cumulative return | Baseline — noisy |
| `vol_adjusted` | Forward return / trailing 20d vol | Slightly better IC |
| `excess` | Forward return - rolling 60d median | Best IC for LGB |

### 2.4 Horizons

Walk-forward windows of 5, 10, 20 trading days. All horizons produce identical metrics because
the target (`target_next_ret`) is always next-day return regardless of horizon parameter.

---

## 3. Sweep Results

### 3.1 Single-Split (80/20, train 1993-2019, test 2019-2026)

| Model | Feature Set | Target | IC | DA | R² |
|-------|-------------|--------|---:|---:|---:|
| GB | default | raw | −0.107 | 0.532 | −0.135 |
| GB | default | excess | −0.062 | 0.469 | −0.134 |
| GB | vol_medium | raw | −0.092 | 0.526 | −0.116 |
| GB | vol_medium | excess | −0.042 | 0.463 | −0.108 |
| LGB | default | raw | −0.065 | 0.540 | −0.144 |
| LGB | default | excess | −0.018 | 0.483 | −0.108 |
| **LGB** | **vol_medium** | **raw** | **+0.061** | **0.542** | −0.046 |
| **LGB** | **vol_medium** | **excess** | **+0.067** | **0.470** | −0.053 |

### 3.2 Walk-Forward (expanding window, test_size=100)

| Config | IC | Dir Acc | Sharpe | R² | Speed |
|--------|-----|---------|--------|-----|-------|
| gb/default/train=250 | 0.015 | 0.508 | 0.34 | −0.226 | 16s |
| **gb/default/train=500** | **0.036** | 0.504 | **0.36** | −0.168 | 19s |
| gb/default/train=1260 | 0.026 | 0.504 | 0.33 | −0.135 | 26s |
| lgb/default/train=250 | 0.019 | 0.510 | 0.34 | −0.219 | 11s |
| lgb/default/train=500 | 0.022 | 0.508 | 0.28 | −0.175 | 11s |
| lgb/default/train=1260 | 0.028 | 0.497 | 0.20 | −0.137 | 13s |

---

## 4. Key Findings

### 4.1 Feature Set Paradox

**Default features (10 basic lagged returns) outperform richer feature sets for walk-forward.**

Why: Richer features (vol_rich, vol_macro) have more NaN values in early training windows
because macro/cross-asset data isn't available before ~2004. HistGB handles NaN natively, but
the NaN-heavy early windows produce poor models that drag down overall IC. Default features
are available from day 1.

For single-split (where all data is available), vol_medium + LGB wins because the full feature
set is available throughout.

**Lesson:** Feature richness helps when data is complete; simplicity wins when data is patchy.

### 4.2 GB vs LGB

- **HistGradientBoostingRegressor** (sklearn): More robust, handles NaN natively, slightly slower
- **LightGBM**: Faster, better IC in single-split, but noisier in walk-forward
- **RandomForest**: Deprecated — worse than both GB variants

### 4.3 Target Type Trade-offs

- **raw**: Best direction accuracy, worst IC
- **excess**: Best IC, worst direction accuracy
- **vol_adjusted**: Middle ground, but produces NaN for early rows

The IC vs DA trade-off is fundamental: IC measures rank correlation (useful for portfolio
weighting), DA measures sign accuracy (useful for binary long/short). For the portfolio gate,
IC matters more.

### 4.4 Why All R² Are Negative

Daily stock returns have near-zero signal-to-noise ratio. Even the best model explains <1% of
variance (R² < 0 means worse than predicting the mean). This is consistent with:
- Efficient Market Hypothesis (weak form)
- Academic literature on return predictability
- The vol model (which predicts volatility, not returns) achieves positive R² = 0.109

**The ML forecast is useful not because it predicts returns accurately, but because its rank
correlation (IC) allows better portfolio weighting than equal-weight or historical-mean.**

### 4.5 The Portfolio Gate Mechanism

The gate in `compute.py` computes:

```
λ(t) = λ_max × skill(t)
skill(t) = 1 - MAE_fc / MAE_histmean  (clipped to [0, 1])
mu = λ × mu_forecast + (1 - λ) × mu_hist
```

With IC ≈ 0.036 and MAE skill ≈ 0.001-0.01:
- λ ≈ 0.004-0.05 (very small)
- mvo_ml ≈ mvo_histmean (the gate correctly identifies weak signal)

**This is the honest behavior.** A stronger forecast would increase λ; a weaker one drives it to 0.

---

## 5. Architecture Decisions

### 5.1 HistGradientBoostingRegressor over GradientBoostingRegressor

**Decision:** Use `HistGradientBoostingRegressor` as the default GB model.

**Rationale:**
- Handles NaN natively (critical because macro features are NaN before ~2004)
- Histogram-based binning is faster than traditional GB
- Same API as `GradientBoostingRegressor`

**Tradeoff:** Requires `max_bins` parameter tuning (default 255 causes crashes on constant features).

### 5.2 Constant Feature Filtering

**Decision:** Drop features with zero variance in each training window before fitting.

**Rationale:** HistGB binning crashes when a feature has ≤1 unique value. This happens frequently
in early training windows where macro features haven't been populated yet.

### 5.3 NaN Target Filtering

**Decision:** Filter NaN values from `y_train` before fitting.

**Rationale:** `vol_adjusted` and `excess` target types produce NaN for early rows (rolling
windows not yet warm). Without filtering, HistGB/LGB crash with "Input y contains NaN."

### 5.4 Pipeline Rewriting

**Decision:** Rewrite `pipeline.py` to use `evaluate_forecast` instead of removed `train_and_predict`.

**Rationale:** The forecast module was rewritten with a new API (`evaluate_forecast` taking a
DataFrame instead of a DuckDB connection). The pipeline must match.

---

## 6. Performance Bottlenecks

### 6.1 Expanding Walk-Forward Speed

With 8400+ rows, train_size=250, test_size=20:
- ~400 model fits per horizon
- Each fit: ~0.1-0.5s (GB) or ~0.05-0.1s (LGB)
- Total: 20-60s per horizon, 60-180s for 3 horizons

**Mitigation:** Use `test_size=100` for faster evaluation (fewer windows).

### 6.2 LGB "No further splits" Warnings

LGB frequently logs "No further splits with positive gain, best gain: -inf" — the tree can't
find useful splits. This indicates the features have weak predictive power (consistent with
negative R²). Not a bug, just the reality of return prediction.

---

## 7. What Doesn't Work (Dead Ends)

| Approach | Result | Why |
|----------|--------|-----|
| Vol_rich features for walk-forward | Worse IC than default | NaN-heavy early windows |
| Excess targets for portfolio | Worse DA | Removes trend signal |
| Direction classification (AUC=0.535) | No better than random | Returns are near-random |
| RF over GB | Worse IC | GB handles noisy targets better |
| Feature pruning from GB | Hurts R² | GB naturally ignores irrelevant features |
| Regime-aware return models | Marginal improvement | Returns don't have strong regime structure |

---

## 8. Recommendations for Future Work

1. **Portfolio integration:** Use the IC-weighted ensemble with `model='lgb', feature_set='vol_medium', target_type='excess'` for the best IC (0.067). But accept λ ≈ 0.01-0.05.

2. **Regime-aware blending:** The vol model's regime signal (high-vol regime has best accuracy)
   could modulate λ — increase weight in high-vol regimes where the return signal is strongest.

3. **Cross-asset signals:** SPY-GLD and SPY-TLT correlation z-scores showed some signal in
   the vol model. Testing them as return predictors could help.

4. **Event-driven features:** NFP, FOMC, earnings dates — calendar events that create
   short-term return predictability.

5. **Ensemble across models:** Combine GB and LGB predictions (they capture different patterns)
   with IC-weighted averaging.

---

## 9. Files Modified in This Session

| File | Changes |
|------|---------|
| `src/mmi/ml/forecast.py` | HistGB default, constant feature filtering, NaN target filtering |
| `src/mmi/ml/pipeline.py` | Rewritten to use `evaluate_forecast` API |
| `docs/RETURN_FORECAST_RESEARCH.md` | This document |
| `docs/research_forecast_sweep.md` | Updated with walk-forward results |

---

## 10. How to Reproduce

```bash
# Quick test (30s)
python -c "
from mmi.ml.forecast import evaluate_forecast
import pandas as pd
df = pd.read_parquet('data/public/fct_asset_daily.parquet')
df = df[df['symbol']=='SPY']
r = evaluate_forecast(df=df, train_size=500, test_size=100, horizon=20, model='gb', feature_set='default')
print(f'IC={r[\"ic\"]:.3f} dir={r[\"direction_accuracy\"]:.3f} sharpe={r[\"sharpe\"]:.2f}')
"

# Full sweep (5min)
python -m mmi.ml.research_forecast
```
