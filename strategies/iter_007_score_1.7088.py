"""
strategy.py — EMA Cross + Weekly BX-Trender + Daily BX-Trender + ATR Trailing Stop

Entry (all must be true):
  1. Daily fast EMA freshly crosses above slow EMA    ← entry trigger
  2. Weekly BX-Trender > bxt_weekly_min               ← HTF must be positive
  3. Daily BX-Trender > bxt_daily_min AND rising      ← daily momentum confirmed
  4. Price above slow EMA at entry (trend confirmation)
  5. Fast EMA slope positive for 2+ bars before cross ← momentum was building

Exit (first to trigger wins):
  - Hard stop loss (last resort)
  - ATR trailing stop (factor=2.9 — give trades room to breathe)
  - Fast EMA crosses below slow EMA

Key change from v6:
  - Added fast EMA pre-slope check: fast EMA must have been rising for 2 bars
    BEFORE the crossover bar. This ensures the crossover has momentum behind it
    rather than being a weak/slow grind through the slow EMA.
  - bxt_daily_min raised: 15 → 20 (tighter momentum filter — only strong readings)
  - bxt_weekly_min raised: 0 → 5 (require clearly positive weekly trend, not just neutral)
  - stop_loss_pct tightened slightly: 0.10 → 0.08 (cut losers faster — win rate improvement)

Rationale: Win rate 31.6% is the main problem. The pre-slope check targets the
root cause: EMA crossovers where the fast EMA was already rising strongly tend to
succeed more than "crawling" crossovers where momentum is weak. Combining this with
higher BXT thresholds should reject more marginal setups while keeping true momentum
entries. The tighter stop also prevents small losses from becoming large ones.
"""

import numpy as np
import pandas as pd

PARAMS = {
    # EMA cross
    "ema_fast": 8,
    "ema_slow": 21,

    # BX-Trender (weekly)
    "bxt_l1": 5,
    "bxt_l2": 20,
    "bxt_l3": 5,
    "bxt_weekly_min": 5,          # raised from 0: require positive weekly trend

    # BX-Trender (daily)
    "bxt_daily_min": 20,          # raised from 15: require strong daily momentum
    "bxt_daily_rising_bars": 1,   # bars over which daily BXT must be rising

    # Fast EMA pre-slope filter
    "ema_fast_slope_bars": 2,     # fast EMA must have been rising for this many bars before cross

    # ATR trailing stop — loosened to give trades room to breathe
    "atr_length": 9,
    "atr_factor": 2.9,            # gives winners room to run

    # Hard stop loss (last resort only — tightened to cut losers faster)
    "stop_loss_pct": 0.08,        # tightened from 0.10 to reduce loss magnitude
    "max_hold_bars": 120,
}


# ── Indicators ────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _bx_trender(close: pd.Series, l1: int, l2: int, l3: int) -> pd.Series:
    """RSI( EMA(close, l1) - EMA(close, l2), l3 ) - 50"""
    diff = _ema(close, l1) - _ema(close, l2)
    return _rsi(diff, l3) - 50


def _weekly_bx_trender(df: pd.DataFrame, as_of_bar: int, l1, l2, l3) -> float:
    subset = df["Close"].iloc[: as_of_bar + 1]
    weekly = subset.resample("W").last().dropna()
    if len(weekly) < max(l1, l2) + l3 + 2:
        return None
    return _bx_trender(weekly, l1, l2, l3).iloc[-1]


def _daily_bx_trender_series(df: pd.DataFrame, as_of_bar: int, l1, l2, l3) -> pd.Series:
    """Return the full BX-Trender series on daily closes up to as_of_bar."""
    close = df["Close"].iloc[: as_of_bar + 1]
    if len(close) < max(l1, l2) + l3 + 2:
        return None
    return _bx_trender(close, l1, l2, l3)


def _atr_trailing_stop(df_slice: pd.DataFrame, length: int, factor: float) -> pd.Series:
    """
    ATR trailing stop (SuperTrend-style), long-only.
      source = (H + L + C + C) / 4
      basic  = source - factor × ATR(length)
      trail  = only moves up — locks in gains as price rises
    """
    high  = df_slice["High"]
    low   = df_slice["Low"]
    close = df_slice["Close"]

    source = (high + low + close + close) / 4

    # True Range → ATR via Wilder smoothing (RMA = EWM alpha=1/length)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean()

    basic = source - factor * atr

    # Trail: only moves up when price is above it
    trail = basic.copy().values
    for j in range(1, len(trail)):
        if close.iloc[j] > trail[j - 1]:
            trail[j] = max(basic.iloc[j], trail[j - 1])
        else:
            trail[j] = basic.iloc[j]

    return pd.Series(trail, index=df_slice.index)


# ── Entry signal ──────────────────────────────────────────────────────────────

def get_entry_signal(df, i, params):
    """
    Enter long when ALL conditions are met:
      1. Fast EMA freshly crosses above slow EMA (today)
      2. Weekly BX-Trender > bxt_weekly_min (HTF context — positive weekly trend)
      3. Daily BX-Trender > bxt_daily_min (strong momentum required) AND rising
      4. Current price is above the slow EMA (trend confirmation)
      5. Fast EMA was already rising for slope_bars before the crossover bar
         (ensures momentum was building, not a weak/crawling cross)
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]
    rising_bars = params["bxt_daily_rising_bars"]
    slope_bars = params["ema_fast_slope_bars"]

    # Need enough history for indicators + slope check
    if i < slow + slope_bars + 5:
        return False

    close = df["Close"].iloc[: i + 1]

    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)

    # Condition 1: fresh EMA crossover (today crossed, yesterday was below)
    crossed_up = (
        (ema_fast.iloc[-2] <= ema_slow.iloc[-2]) and
        (ema_fast.iloc[-1] > ema_slow.iloc[-1])
    )
    if not crossed_up:
        return False

    # Condition 4: Price must be above slow EMA (avoid false crossovers in downtrend)
    current_price = close.iloc[-1]
    if current_price <= ema_slow.iloc[-1]:
        return False

    # Condition 5: Fast EMA must have been rising for slope_bars BEFORE the crossover
    # Check that ema_fast[-2] > ema_fast[-2-slope_bars] (rising into the cross)
    # This rejects "crawling" crossovers where fast EMA was flat or declining
    ema_fast_before_cross = ema_fast.iloc[-2]           # value at crossover bar (prev)
    ema_fast_slope_start = ema_fast.iloc[-(2 + slope_bars)]  # value slope_bars before that
    if ema_fast_before_cross <= ema_fast_slope_start:
        return False

    # Condition 2: Weekly BX-Trender must be above minimum threshold (positive weekly trend)
    bxt_w = _weekly_bx_trender(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_w is None or bxt_w <= params["bxt_weekly_min"]:
        return False

    # Condition 3: Daily BX-Trender — above threshold AND rising
    bxt_series = _daily_bx_trender_series(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_series is None or len(bxt_series) < rising_bars + 2:
        return False

    bxt_now = bxt_series.iloc[-1]

    # Must be above the minimum threshold (strong momentum filter)
    if bxt_now <= params["bxt_daily_min"]:
        return False

    # Must be rising: current value > value N bars ago
    bxt_prev = bxt_series.iloc[-(rising_bars + 1)]
    if bxt_now <= bxt_prev:
        return False

    return True


# ── Exit signal ───────────────────────────────────────────────────────────────

def get_exit_signal(df, i, entry_price, params):
    """
    Exit when any of:
      1. Hard stop loss (last resort — tightened to 8% to cut losers faster)
      2. ATR trailing stop hit (factor=2.9 — gives trades room to develop)
      3. Fast EMA crosses below slow EMA
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]

    df_slice = df.iloc[: i + 1]
    close = df_slice["Close"]
    current_price = close.iloc[-1]

    # 1. Hard stop loss (tightened to cut losers faster and protect capital)
    if current_price <= entry_price * (1 - params["stop_loss_pct"]):
        return True

    if len(close) < max(slow, params["atr_length"]) + 2:
        return False

    # 2. ATR trailing stop (factor=2.9 — gives winners room to run)
    trail = _atr_trailing_stop(df_slice, params["atr_length"], params["atr_factor"])
    if current_price < trail.iloc[-1]:
        return True

    # 3. EMA cross below
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return True

    return False