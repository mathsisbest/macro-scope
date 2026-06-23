"""Backtest windows — the single source of truth for the ``window`` dimension (Phase D).

Including BTC hard-limits a backtest to its ~2015 inception, so the BTC impact is read off three
windows rather than a single confounded comparison:

- ``ex_btc_2002``  — the 5 non-crypto assets over their full history (the long-history baseline).
- ``ex_btc_2015``  — the *same* 5 assets restricted to BTC's era (the same-period control).
- ``inc_btc_2015`` — those 5 assets + BTC over BTC's era (adds only BTC, period held fixed).

So **BTC-effect** = ``inc_btc_2015 − ex_btc_2015`` (identical dates, ± BTC) and **period-effect** =
``ex_btc_2015 − ex_btc_2002`` (identical universe, different era). This module is imported by both
the ``mmi portfolio`` orchestration and the dbt accepted_values, so the enum can never drift.
"""

from __future__ import annotations

EX_BTC_2002 = "ex_btc_2002"
EX_BTC_2015 = "ex_btc_2015"
INC_BTC_2015 = "inc_btc_2015"

# Ordered for display (long baseline first); D4 sources the dbt accepted_values list from this.
WINDOWS = (EX_BTC_2002, EX_BTC_2015, INC_BTC_2015)

# `cmd_portfolio` runs only this window until the multi-window machinery lands (D6); it reproduces
# the pre-Phase-D backtest exactly (the 5 non-crypto assets over their full history).
DEFAULT_WINDOW = EX_BTC_2002
