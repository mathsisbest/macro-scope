"""Backtest windows — the single source of truth for the ``window`` dimension (Phase D).

Including BTC hard-limits a backtest to its ~2015 inception, so the BTC impact is read off three
windows rather than a single confounded comparison:

- ``ex_btc_2002``  — the non-crypto sleeves over their longest common history (the long baseline).
- ``ex_btc_2015``  — the *same* sleeves restricted to BTC's era (the same-period control).
- ``inc_btc_2015`` — those sleeves + BTC over BTC's era (adds only BTC, period held fixed).

So **BTC-effect** = ``inc_btc_2015 − ex_btc_2015`` (identical dates, ± BTC) and **period-effect** =
``ex_btc_2015 − ex_btc_2002`` (identical universe, different era). This module is imported by both
the ``mmi portfolio`` orchestration and the dbt accepted_values, so the enum can never drift.
"""

from __future__ import annotations

# The strategic portfolio universe — ONE representative asset per class, so risk-parity / MVO see
# *independent* risks rather than redundant ones (SPY+QQQ+VEA would triple-count equity beta;
# EURUSD+GBPUSD double-count a USD-FX bet). Equity · bonds · commodity; BTC is added only by the
# inc_btc window. Every other ingested ticker stays available for the Markets-tab charts — it is
# simply excluded from the optimiser's input. Change here and the universe changes everywhere.
PORTFOLIO_UNIVERSE = ("SPY", "TLT", "GLD")

EX_BTC_2002 = "ex_btc_2002"
EX_BTC_2015 = "ex_btc_2015"
INC_BTC_2015 = "inc_btc_2015"

# Ordered for display (long baseline first); D4 sources the dbt accepted_values list from this.
# NOTE: the "_2002" id is historical/stable (referenced by dbt accepted_values + the snapshot). The
# baseline's *effective* start is the longest common history of PORTFOLIO_UNIVERSE — bounded by GLD
# (lists Nov 2004) — so the data begins ~2004, surfaced honestly in the dashboard label.
WINDOWS = (EX_BTC_2002, EX_BTC_2015, INC_BTC_2015)

# `cmd_portfolio` runs only this window until the multi-window machinery lands (D6); it reproduces
# the pre-Phase-D backtest (the non-crypto sleeves over their longest common history).
DEFAULT_WINDOW = EX_BTC_2002
