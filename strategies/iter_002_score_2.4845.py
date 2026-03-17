"""
strategy.py — EMA Cross + Weekly BX-Trender + Daily BX-Trender + ATR Trailing Stop

Entry (all must be true):
  1. Daily fast EMA freshly crosses above slow EMA    ← entry trigger
  2. Weekly BX-Trender > bxt_weekly_min               ← HTF not in downtrend
  3. Daily BX-Trender > bxt_daily_min                 ← daily momentum confirming

Exit (first to trigger wins):
  - Price closes below ATR trailing stop (ATR length=9, factor=2.9)
  - Fast EMA crosses below slow EMA
  - Hard stop loss (last resort)

ATR Trailing Stop:
  source = (High + Low + Close + Close) / 4   (weighted close / HLC4)
  stop   = source - factor × ATR(length)       (only moves up, never down)
"""

import numpy as np
import pandas as pd

PARAMS = {
    # EMA cross — autoresearch can tune these (base: 8/21, also try 3/8, 5/13)
    "ema_fast": 8,
    "ema_slow": 21,

    # BX-Trender (weekly) — autoresearch can tune threshold
    "bxt_l1": 5,
    "bxt_l2": 20,
    "bxt_l3": 5,
    "bxt_weekly_min": -10,

    # BX-Trender (daily) — extra momentum filter
    "bxt_daily_min": 0,   # daily BX-Trender must be > 0 (positive momentum)

    # ATR trailing stop — autoresearch can tune
    "atr_length": 9,
    "atr_factor": 2.9,

    # Hard stop loss (last resort only)
    "stop_loss_pct": 0.10,
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


def _daily_bx_trender(df: pd.DataFrame, as_of_bar: int, l1, l2, l3) -> float:
    """Compute BX-Trender on daily closes up to as_of_bar."""
    close = df["Close"].iloc[: as_of_bar + 1]
    if len(close) < max(l1, l2) + l3 + 2:
        return None
    bxt = _bx_trender(close, l1, l2, l3)
    return bxt.iloc[-1]


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
    for i in range(1, len(trail)):
        if close.iloc[i] > trail[i - 1]:
            trail[i] = max(basic.iloc[i], trail[i - 1])
        else:
            trail[i] = basic.iloc[i]

    return pd.Series(trail, index=df_slice.index)


# ── Entry signal ──────────────────────────────────────────────────────────────

def get_entry_signal(df, i, params):
    """
    Enter long when:
      1. Fast EMA freshly crosses above slow EMA
      2. Weekly BX-Trender > bxt_weekly_min
      3. Daily BX-Trender > bxt_daily_min  (new filter — momentum confirmation)
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]

    if i < slow + 5:
        return False

    close = df["Close"].iloc[: i + 1]

    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)

    # Condition 1: fresh EMA crossover (today crossed, yesterday was below)
    crossed_up = (ema_fast.iloc[-2] <= ema_slow.iloc[-2]) and (ema_fast.iloc[-1] > ema_slow.iloc[-1])
    if not crossed_up:
        return False

    # Condition 2: Weekly BX-Trender must be above minimum threshold
    bxt_w = _weekly_bx_trender(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_w is None or bxt_w <= params["bxt_weekly_min"]:
        return False

    # Condition 3: Daily BX-Trender must be positive (daily momentum confirming)
    bxt_d = _daily_bx_trender(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_d is None or bxt_d <= params["bxt_daily_min"]:
        return False

    return True


# ── Exit signal ───────────────────────────────────────────────────────────────

def get_exit_signal(df, i, entry_price, params):
    """
    Exit when any of:
      1. Price closes below ATR trailing stop
      2. Fast EMA crosses below slow EMA
      3. Hard stop loss (last resort)
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]

    df_slice = df.iloc[: i + 1]
    close = df_slice["Close"]
    current_price = close.iloc[-1]

    # 1. Hard stop loss
    if current_price <= entry_price * (1 - params["stop_loss_pct"]):
        return True

    if len(close) < max(slow, params["atr_length"]) + 2:
        return False

    # 2. ATR trailing stop (checked before EMA cross — tighter exit)
    trail = _atr_trailing_stop(df_slice, params["atr_length"], params["atr_factor"])
    if current_price < trail.iloc[-1]:
        return True

    # 3. EMA cross below
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return True

    return False