# Return Forecast Research Sweep

**Date:** 2026-07-05
**Asset:** SPY (8413 rows, 1993-01-29 to 2026-07-02)
**Source:** Snapshot Parquet (`data/public/`)

> **Full findings, architecture decisions, and lessons learned:** See [RETURN_FORECAST_RESEARCH.md](RETURN_FORECAST_RESEARCH.md)

## Single-Split Results (80/20, train 1993–2019, test 2019–2026)

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

## Walk-Forward Results (expanding window, test_size=100)

| Config | IC | Dir Acc | Sharpe | R² |
|--------|-----|---------|--------|-----|
| gb/default/train=250 | 0.015 | 0.508 | 0.34 | −0.226 |
| **gb/default/train=500** | **0.036** | 0.504 | **0.36** | −0.168 |
| gb/default/train=1260 | 0.026 | 0.504 | 0.33 | −0.135 |
| lgb/default/train=250 | 0.019 | 0.510 | 0.34 | −0.219 |
| lgb/default/train=500 | 0.022 | 0.508 | 0.28 | −0.175 |
| lgb/default/train=1260 | 0.028 | 0.497 | 0.20 | −0.137 |

## Key Takeaways

1. **LGB/vol_medium is the best single-split config** (IC=0.067)
2. **GB/default is the best walk-forward config** (IC=0.036, sharpe=0.36)
3. **All R² negative** — expected for daily return prediction
4. **Default features outperform richer sets in walk-forward** due to NaN in early windows
5. **The ML forecast adds marginal value** — portfolio gate λ ≈ 0.01-0.05
