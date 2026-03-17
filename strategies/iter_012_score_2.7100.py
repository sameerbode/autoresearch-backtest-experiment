"""
strategy.py — EMA Cross + Weekly BX-Trender + Daily BX-Trender + ATR Trailing Stop

Entry (all must be true):
  1. Daily fast EMA freshly crosses above slow EMA    ← entry trigger
  2. Weekly BX-Trender > bxt_weekly_min               ← HTF trend filter
  3. Daily BX-Trender > bxt_daily_min                 ← daily momentum confirmed
  4. Price above slow EMA at entry (trend confirmation)
  5. Fast EMA slope positive for slope_bars before cross
  6. Volume on crossover bar > volume_ma_mult × avg volume
  7. NEW: EMA was separated (fast was genuinely below slow) before crossover
     — ensures we're catching real trend reversals, not consolidation wiggles

Key change from v11:
  - Relax bxt_weekly_min back to -5 (was +5) — the over-tightening reduced trades
    from 20+ to 14 without improving win rate meaningfully
  - Try EMA 10/30 instead of 8/21 — the 10/30 pair gives fewer, higher-quality
    crossovers on daily charts with less noise; reduces false breaks in consolidation
  - Add "separation filter": require fast EMA was below slow EMA by at least
    separation_pct of price for N bars before the cross. This ensures we're
    catching genuine trend reversals, not choppy consolidation crosses.
  - Slightly tighter daily BXT: 5 → 8 to compensate for the relaxed weekly filter
  - ATR factor: 3.2 → 3.0 — slight tightening to lock in more profits

Rationale: Win rate 42.9% with only 14 trades means we need BOTH more trades AND
better win rate. The 8/21 EMA generates many small crosses in consolidation.
The 10/30 pair smooths out noise while the separation filter ensures each entry
is a genuine trend change.
"""

import numpy as np
import pandas as pd

PARAMS = {
    # EMA cross — switched to 10/30 for higher quality signals
    "ema_fast": 10,
    "ema_slow": 30,

    # BX-Trender (weekly)
    "bxt_l1": 5,
    "bxt_l2": 20,
    "bxt_l3": 5,
    "bxt_weekly_min": -5,         # relaxed from +5 — was over-filtering trades

    # BX-Trender (daily)
    "bxt_daily_min": 8,           # slightly tighter to compensate for relaxed weekly
    "bxt_daily_rising_bars": 0,   # 0 = disabled

    # EMA separation filter — require fast was genuinely below slow before cross
    "ema_separation_bars": 3,     # fast EMA must have been below slow for N bars
    "ema_separation_pct": 0.001,  # by at least 0.1% of price (filters tiny wiggles)

    # Fast EMA pre-slope filter
    "ema_fast_slope_bars": 2,     # require 2-bar rising slope before cross

    # Volume confirmation
    "volume_ma_period": 20,
    "volume_ma_mult": 0.8,        # loose to avoid rejecting valid setups

    # ATR trailing stop
    "atr_length": 9,
    "atr_factor": 3.0,            # slightly tighter than 3.2 to lock in more profits

    # Hard stop loss (last resort only)
    "stop_loss_pct": 0.12,        # 12% hard stop
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

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False).mean()

    basic = source - factor * atr

    trail = basic.copy().values
    for j in range(1, len(trail)):
        if close.iloc[j] > trail[j - 1]:
            trail[j] = max(basic.iloc[j], trail[j - 1])
        else:
            trail[j] = basic.iloc[j]

    return pd.Series(trail, index=df_slice.index)


def _volume_ma(df: pd.DataFrame, as_of_bar: int, period: int) -> float:
    """Average volume over the last `period` bars (excluding the current bar)."""
    if "Volume" not in df.columns:
        return None
    vol = df["Volume"].iloc[: as_of_bar + 1]
    if len(vol) < period + 1:
        return None
    return vol.iloc[-(period + 1):-1].mean()


# ── Entry signal ──────────────────────────────────────────────────────────────

def get_entry_signal(df, i, params):
    """
    Enter long when ALL conditions are met:
      1. Fast EMA freshly crosses above slow EMA (today)
      2. Weekly BX-Trender > bxt_weekly_min (relaxed to -5 for more trades)
      3. Daily BX-Trender > bxt_daily_min (slightly tighter at 8)
      4. Current price is above the slow EMA (trend confirmation)
      5. Fast EMA was already rising for slope_bars before the crossover bar
      6. Volume on crossover bar > volume_ma_mult × avg volume (conviction check)
      7. Fast EMA was genuinely below slow EMA for separation_bars before cross
         (ensures real trend reversal, not consolidation wiggle)
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]
    slope_bars = params["ema_fast_slope_bars"]
    sep_bars = params["ema_separation_bars"]

    # Need enough history
    lookback = max(slow + slope_bars + sep_bars + 5, 50)
    if i < lookback:
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

    # Condition 4: Price must be above slow EMA
    current_price = close.iloc[-1]
    if current_price <= ema_slow.iloc[-1]:
        return False

    # Condition 5: Fast EMA must have been rising for slope_bars BEFORE the crossover
    ema_fast_before_cross = ema_fast.iloc[-2]
    ema_fast_slope_start = ema_fast.iloc[-(2 + slope_bars)]
    if ema_fast_before_cross <= ema_fast_slope_start:
        return False

    # Condition 7: EMA separation filter
    # Fast EMA must have been BELOW slow EMA for sep_bars consecutive bars before cross
    # This ensures the cross is genuine — not a tiny consolidation wiggle
    sep_pct = params["ema_separation_pct"]
    genuine_separation = True
    for k in range(2, 2 + sep_bars):
        # Check that fast was below slow (by at least sep_pct of price) sep_bars ago
        if len(ema_fast) < k + 1 or len(ema_slow) < k + 1:
            genuine_separation = False
            break
        fast_val = ema_fast.iloc[-(k + 1)]
        slow_val = ema_slow.iloc[-(k + 1)]
        price_ref = close.iloc[-(k + 1)]
        # Fast must have been below slow by at least sep_pct
        if fast_val >= slow_val - sep_pct * price_ref:
            genuine_separation = False
            break

    if not genuine_separation:
        return False

    # Condition 6: Volume confirmation
    vol_period = params["volume_ma_period"]
    vol_mult = params["volume_ma_mult"]
    avg_vol = _volume_ma(df, i, vol_period)
    if avg_vol is not None and avg_vol > 0:
        current_vol = df["Volume"].iloc[i] if "Volume" in df.columns else None
        if current_vol is not None:
            if current_vol < vol_mult * avg_vol:
                return False

    # Condition 2: Weekly BX-Trender (relaxed to -5 for more opportunities)
    bxt_w = _weekly_bx_trender(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_w is None or bxt_w <= params["bxt_weekly_min"]:
        return False

    # Condition 3: Daily BX-Trender > 8 (slightly tighter)
    bxt_series = _daily_bx_trender_series(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_series is None:
        return False

    bxt_now = bxt_series.iloc[-1]
    if bxt_now <= params["bxt_daily_min"]:
        return False

    # Optional: if rising_bars > 0, also require BXT to be rising
    rising_bars = params["bxt_daily_rising_bars"]
    if rising_bars > 0:
        if len(bxt_series) < rising_bars + 2:
            return False
        bxt_prev = bxt_series.iloc[-(rising_bars + 1)]
        if bxt_now <= bxt_prev:
            return False

    return True


# ── Exit signal ───────────────────────────────────────────────────────────────

def get_exit_signal(df, i, entry_price, params):
    """
    Exit when any of:
      1. Hard stop loss (last resort — 12% wide)
      2. ATR trailing stop hit (factor=3.0)
      3. Fast EMA crosses below slow EMA
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

    # 2. ATR trailing stop (factor=3.0)
    trail = _atr_trailing_stop(df_slice, params["atr_length"], params["atr_factor"])
    if current_price < trail.iloc[-1]:
        return True

    # 3. EMA cross below
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return True

    return False