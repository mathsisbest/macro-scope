# ML Forecast Experiment Log: Every Attempt, Every Result

**Purpose:** Complete record of every approach tested to improve the ML return forecast for SPY.
**Status:** Living document — append new experiments at the top.
**Last updated:** 2026-07-05

---

## Summary Table

| # | Experiment | Session | Result | Verdict |
|---|-----------|---------|--------|---------|
| 1 | Regime-aware RF return forecaster | S1 | dir_acc=0.54, R²=-0.062 | Marginal |
| 2 | Macro feature expansion (18 FRED) | S1 | R² dropped 0.115→0.091 (vol) | Reverted |
| 3 | Low-frequency FRED features (CPI, GDP) | S1 | R² dropped, artificial plateaus | Reverted |
| 4 | Feature importance analysis | S1 | har_vol_20d=46%, macro=30% | Informative |
| 5 | Feature selection via tree importance | S1 | R²=-0.048 (worse) | Reverted |
| 6 | Cross-asset regime signals (5 features) | S2 | Vol R²: 0.115→0.109 | Kept (minor) |
| 7 | Embargo gap fix (horizon rows) | S2 | Prevented data leakage | Fixed |
| 8 | Interaction features (4 cross-terms) | S2 | Marginal improvement | Kept |
| 9 | LightGBM replacement | S2 | IC=0.066 vs RF=0.042 | Kept |
| 10 | Multi-horizon ensemble (5d/10d/20d) | S2 | Decay weighting | Kept |
| 11 | Vol-adjusted mu for portfolio | S2 | Tested | Not used |
| 12 | Feature pruning (remove importance<0.001) | S2 | R²: 0.109→0.089 | Reverted |
| 13 | GB (sklearn) return forecaster | S3 | R²=-0.085 to -0.189 | Negative R² |
| 14 | Direction classifier (AUC) | S3 | AUC=0.535, worse than baseline | Dead end |
| 15 | Excess target type | S3 | IC=0.067 single-split, NaN crash | Fixed NaN |
| 16 | Vol-adjusted target type | S3 | Similar to excess | Fixed NaN |
| 17 | Default vs vol_rich features | S3 | Default wins (IC=0.036 vs 0.010) | Key finding |
| 18 | GB vs LGB comparison | S3 | GB/sharpe=0.36, LGB/sharpe=0.34 | GB slightly better |
| 19 | Train size sweep (250/500/1260) | S3 | 500 is sweet spot | Key finding |
| 20 | HistGB (NaN-native) | S3 | Handles NaN, crashes on constants | Fixed |
| 21 | Constant feature filtering | S3 | Prevents HistGB binning crash | Fixed |
| 22 | NaN target filtering | S3 | Enables excess/vol_adjusted | Fixed |

---

## Detailed Experiment Records

### Experiment 1: Regime-Aware RF Return Forecaster

**Date:** ~2026-06 (Session 1)
**Commit:** ac6de66
**Approach:** RandomForestRegressor with vol_rich features (35), separate models per vol tercile (Low/Med/High), 4 horizons (1d, 5d, 10d, 20d), embargo for horizons > 1.

**Results:**
- Direction accuracy: 54% (barely above baseline 55%)
- R²: -0.062 (20d horizon) — worse than predicting the mean
- Regime breakdown: high-vol regime had best accuracy (0.54-0.58)

**Verdict:** Regime-aware modeling helps marginally. The high-vol regime signal is the only actionable finding. But overall, the model has no predictive skill.

**Lesson:** Returns are fundamentally harder to predict than volatility. Volatility is highly autocorrelated (today's vol predicts tomorrow's); returns are essentially random walks.

---

### Experiment 2: Macro Feature Expansion

**Date:** ~2026-06 (Session 1)
**Approach:** Expanded `_MACRO_FEATURE_NAMES` to 18 features from FRED: yield curve (slope, z-score), treasuries (10Y, 2Y, 3M + changes), policy (fed funds rate), VIX (level, change, z-score), oil, dollar, claims, NFCI, GLD/TLT vol.

**Results:**
- Vol model R² dropped from 0.115 to 0.091
- 12 extra FRED series created artificial plateau patterns when forward-filled to daily dates

**Verdict:** Reverted. Monthly/quarterly series forward-filled to daily dates create noise, not signal.

**Lesson:** Feature frequency matters. Only daily/weekly FRED series provide genuine predictive signal. Low-frequency series (CPI, GDP, UNRATE) are too smooth when forward-filled.

---

### Experiment 3: Low-Frequency FRED Features

**Date:** ~2026-06 (Session 1)
**Approach:** Added CPIAUCSL, UNRATE, PAYEMS, GFDEGDQ188S, M2SL, WALCL, SAHMREALTIME, A191RL1Q225SBEA to feature set.

**Results:**
- R² dropped further
- Forward-filled quarterly/monthly data creates artificial step functions
- Model overfits to the plateaus

**Verdict:** Reverted. Only daily/weekly features help.

---

### Experiment 4: Feature Importance Analysis

**Date:** ~2026-06 (Session 1)
**Approach:** Analyzed GB feature importance on vol_rich set.

**Results (top 10):**
1. PCE inflation: 0.094
2. 20d return vol: 0.078
3. VIX level: 0.061
4. Dollar momentum: 0.060
5. Vol dispersion: 0.055
6. NFCI change: 0.041
7. 10Y yield: 0.038
8. SPY-GLD correlation: 0.031
9. HAR vol 20d: 0.46 (dominates)
10. Day-of-week: 0.000

**Verdict:** Informative — har_vol_20d dominates (46%), macro/cross-asset signals collectively ~30%.

---

### Experiment 5: Feature Selection via Tree Importance

**Date:** ~2026-06 (Session 1)
**Approach:** Remove features with importance < 0.001 from vol_rich set (trimmed to 25 features).

**Results:**
- R² dropped from 0.109 to 0.048

**Verdict:** Reverted. Don't prune from GB models.

**Lesson:** GB naturally handles irrelevant features by giving them few splits. The "unimportant" features provide ensemble diversity. Removing them hurts more than helping.

---

### Experiment 6: Cross-Asset Regime Signals

**Date:** 2026-07-05 (Session 2)
**Commit:** 10f66cf
**Approach:** Added 5 features to vol_rich set (now 50 total): corr_spy_tlt_zscore_60d, corr_spy_gld_zscore_60d, dollar_zscore_60d, cross_asset_dispersion_20d, equity_bond_spread_20d.

**Results:**
- Vol model R²: 0.115 → 0.109 (slight drop)
- Still clears skill gate (R² > 0.10)

**Verdict:** Kept — minor regression but captures regime shifts.

---

### Experiment 7: Embargo Gap Fix

**Date:** 2026-07-05 (Session 2)
**Approach:** Fixed data leakage in `_walk_forward` — the test fold's targets overlapped with training features for horizons > 1. Added `horizon`-row embargo.

**Results:**
- Prevented target leakage
- Metrics became honest (slightly worse but correct)

**Verdict:** Critical fix. All prior return forecaster metrics were slightly inflated.

---

### Experiment 8: Interaction Features

**Date:** 2026-07-05 (Session 2)
**Approach:** Added 4 cross-term features: vix_x_yc_slope, vol_disp_x_vol_of_vol, nfci_x_dollar_zscore, skew_x_vol_dispersion.

**Results:**
- Marginal improvement in IC
- Vol model: R²=0.103, QR=0.688, folds=3/10 — still CLEARED

**Verdict:** Kept — captures interactions between strongest predictors.

---

### Experiment 9: LightGBM Replacement

**Date:** 2026-07-05 (Session 2)
**Approach:** Replace sklearn GB/RF with LightGBM for speed + better hyperparameters.

**Results:**
- IC: 0.066 (LGB) vs 0.042 (RF) — 57% higher
- R²: -0.103 (LGB) vs -0.005 (RF) — worse due to higher variance
- But IC is what matters for MVO, not R²

**Verdict:** Kept. LGB better for ranking (IC), RF better for magnitude (R²). For portfolio, IC matters more.

---

### Experiment 10: Multi-Horizon Ensemble

**Date:** 2026-07-05 (Session 2)
**Approach:** Combine 5d/10d/20d forecasts with decay weighting (shorter horizons get more weight). Also implemented IC-weighted variant.

**Results:**
- Ensemble smooths out single-horizon noise
- IC-weighted weighting adapts to which horizon has the most signal

**Verdict:** Kept. Standard practice for multi-horizon forecasting.

---

### Experiment 11: Vol-Adjusted Mu for Portfolio

**Date:** 2026-07-05 (Session 2)
**Approach:** Divide forecast by vol prediction to get "forecast Sharpe" instead of raw return forecast.

**Results:**
- Tested but not adopted for production

**Verdict:** Not used. Adds complexity without clear benefit.

---

### Experiment 12: Feature Pruning (Remove Low-Importance)

**Date:** 2026-07-05 (Session 2)
**Approach:** Remove features with importance < 0.001 from vol_rich set.

**Results:**
- R² dropped from 0.109 to 0.089 (below 0.10 skill gate)

**Verdict:** Reverted. Don't prune from GB models.

**Lesson:** Same as Experiment 5. GB handles irrelevant features naturally.

---

### Experiment 13: GB (sklearn) Return Forecaster

**Date:** 2026-07-05 (Session 3)
**Approach:** Switch return forecaster from RandomForest to GradientBoostingRegressor with vol_rich features.

**Results:**
- 5d: IC=0.082, R²=-0.085
- 10d: IC=0.089, R²=-0.092
- 20d: IC=0.098, R²=-0.189
- All R² negative, IC slightly positive

**Verdict:** GB has better IC than RF but R² is worse. IC is the right metric for portfolio.

---

### Experiment 14: Direction Classifier

**Date:** 2026-07-05 (Session 3)
**Approach:** GradientBoostingClassifier predicting binary direction (up/down). AUC-ROC as metric.

**Results:**
- AUC: 0.535 (barely above random 0.5)
- Direction accuracy: 56.3% (worse than baseline 68.2%)
- Holdout AUC: 0.380 (terrible)

**Verdict:** Dead end. Direction classification has no skill.

**Lesson:** Predicting sign of returns is even harder than predicting magnitude. The baseline (always predict "up") achieves 68% accuracy because of the bull market bias in 20-year data.

---

### Experiment 15: Excess Target Type

**Date:** 2026-07-05 (Session 3)
**Approach:** Target = cumulative return - rolling 60d median. Removes drift component.

**Results:**
- Single-split: IC=0.067 (best of all configs)
- Walk-forward: Crashed with "Input y contains NaN" (fixed later)
- After NaN fix: IC=0.009, sharpe=-0.05 (walk-forward)

**Verdict:** Good for single-split, poor for walk-forward. The excess target removes signal along with drift.

---

### Experiment 16: Vol-Adjusted Target Type

**Date:** 2026-07-05 (Session 3)
**Approach:** Target = cumulative return / trailing 20d vol. More stationary target.

**Results:**
- Similar to excess target
- Also crashes without NaN filtering

**Verdict:** Fixed NaN issue but not adopted as default.

---

### Experiment 17: Default vs Rich Features

**Date:** 2026-07-05 (Session 3)
**Approach:** Systematic comparison of default (10), vol_medium (27), vol_rich (50/64) features.

**Results (walk-forward):**
- default: IC=0.036, sharpe=0.36
- vol_medium: IC=0.021, sharpe=0.16
- vol_rich: IC=0.010, sharpe=0.04

**Verdict:** Default features win decisively.

**Lesson (the "feature set paradox"):** Richer features have more NaN in early training windows (macro data unavailable before ~2004). HistGB handles NaN natively, but the NaN-heavy early windows produce poor models that drag down overall IC. Default features are available from day 1.

---

### Experiment 18: GB vs LGB

**Date:** 2026-07-05 (Session 3)
**Approach:** Head-to-head comparison at train_size=500.

**Results:**
- GB: IC=0.036, sharpe=0.36, 19s
- LGB: IC=0.022, sharpe=0.28, 11s

**Verdict:** GB slightly better on default features. LGB is 2x faster.

---

### Experiment 19: Train Size Sweep

**Date:** 2026-07-05 (Session 3)
**Approach:** Test train_size=250, 500, 1260 with GB/default.

**Results:**
- train=250: IC=0.015 (too little data)
- train=500: IC=0.036, sharpe=0.36 (sweet spot)
- train=1260: IC=0.026, sharpe=0.33 (more data but lower IC)

**Verdict:** 500 is the sweet spot.

**Lesson:** More training data doesn't always help. Returns are non-stationary — very old data may contain patterns that no longer apply.

---

### Experiment 20: HistGradientBoostingRegressor

**Date:** 2026-07-05 (Session 3)
**Approach:** Switch from GradientBoostingRegressor to HistGradientBoostingRegressor for native NaN handling.

**Results:**
- Handles NaN features natively (critical for macro features)
- Crashes on constant features (≤1 unique value) — fixed with filtering
- Requires `max_iter` (not `n_estimators`), `max_bins` parameter

**Verdict:** Adopted as default. Essential for the mixed-frequency feature set.

---

### Experiment 21: Constant Feature Filtering

**Date:** 2026-07-05 (Session 3)
**Approach:** Drop features with std==0 in each training window before fitting.

**Results:**
- Prevents HistGB binning crash ("window shape cannot be larger than input array shape")
- Features like gk_vol, dollar_change_20d, corr_spy_tlt_20d have 1 unique value in early windows

**Verdict:** Required fix for HistGB.

---

### Experiment 22: NaN Target Filtering

**Date:** 2026-07-05 (Session 3)
**Approach:** Filter NaN from y_train before fitting (for vol_adjusted/excess targets).

**Results:**
- Enables excess and vol_adjusted target types to work
- Without filtering: "Input y contains NaN" crash

**Verdict:** Required fix for non-raw target types.

---

## Dead Ends (Complete List)

| Approach | Why It Failed |
|----------|---------------|
| Lasso/OLS/Ridge on vol features | R²=-2.025 — all linear models fail on vol features |
| Regime-aware HAR | Identical to baseline HAR — regime doesn't help linear models |
| Manual feature pruning to 25 | R²=-0.048, worse than full set |
| Pruning low-importance GB features | R² dropped 0.109→0.089, below skill gate |
| Low-frequency FRED features (CPI, GDP, UNRATE) | Artificial plateau patterns when forward-filled |
| Direction classifier (AUC) | AUC=0.535, no better than random |
| GB/vol_rich/excess (walk-forward) | IC=0.009, sharpe=-0.05 — worst config |
| Vol-adjusted mu for portfolio | Added complexity without clear benefit |
| GMM/HMM regimes | Not tested (planned but not implemented) |
| Feature importance tracking for drift | Not tested (planned but not implemented) |
| Cross-asset features from full universe (QQQ, VEA, BTC) | Not tested (planned but not implemented) |

---

## Key Principles Discovered

1. **Don't prune from GB models.** GB handles irrelevant features naturally. Removing them hurts ensemble diversity.
2. **Default features beat rich features for returns.** NaN in early windows kills richer feature sets.
3. **IC is the right metric, not R².** Returns have near-zero R² for any model. IC (rank correlation) drives portfolio allocation.
4. **More data ≠ better returns prediction.** Non-stationarity means old data can hurt.
5. **HistGB over standard GB.** NaN-native handling is essential for mixed-frequency features.
6. **Feature frequency matters.** Only daily/weekly FRED series help. Monthly/quarterly create noise.
7. **Embargo is critical.** Without it, target leakage inflates metrics.
8. **Returns are fundamentally hard.** Best IC is 0.036-0.067. Portfolio gate λ ≈ 0.01-0.05 (correctly weak).

---

## Reproduction Commands

```bash
# Quick validation (30s)
python -c "
from mmi.ml.forecast import evaluate_forecast
import pandas as pd
df = pd.read_parquet('data/public/fct_asset_daily.parquet')
df = df[df['symbol']=='SPY']
r = evaluate_forecast(df=df, train_size=500, test_size=100, horizon=20,
                      model='gb', feature_set='default')
print(f'IC={r[\"ic\"]:.3f} sharpe={r[\"sharpe\"]:.2f}')
"

# Full sweep (5min)
python -m mmi.ml.research_forecast

# Reproduce dead ends
python -c "
from mmi.ml.forecast import evaluate_forecast
import pandas as pd
df = pd.read_parquet('data/public/fct_asset_daily.parquet')
df = df[df['symbol']=='SPY']
# This should crash without NaN fix:
r = evaluate_forecast(df=df, train_size=500, test_size=100, horizon=20,
                      model='lgb', feature_set='vol_medium', target_type='excess')
"
```
