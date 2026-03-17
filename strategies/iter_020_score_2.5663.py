"""
strategy.py — EMA Cross + Weekly BX-Trender + Daily BX-Trender + ATR Trailing Stop
              + Volume Confirmation Filter

Entry (all must be true):
  1. Daily fast EMA freshly crosses above slow EMA    ← entry trigger
  2. Weekly BX-Trender > bxt_weekly_min               ← HTF must be acceptable
  3. Daily BX-Trender > bxt_daily_min AND rising      ← daily momentum confirmed
  4. Price above slow EMA at entry (trend confirmation)
  5. Fast EMA slope positive for 1+ bar before cross  ← momentum building
  6. Volume on crossover bar >= volume_min_mult × avg volume ← NEW: volume confirmation

Exit (first to trigger wins):
  - Hard stop loss (last resort only)
  - ATR trailing stop (give trades room to breathe)
  - Fast EMA crosses below slow EMA

Key change from v8 (iteration 19):
  - Added volume confirmation filter at entry
  - Require crossover bar volume >= 1.0x average (i.e., at least average — not weak)
  - volume_avg_bars: 20-bar lookback for average volume
  - This filters out low-conviction EMA crosses that tend to fail
  - Win rate 39.3% → target 50%+ by rejecting weak-volume crossovers
  - May reduce trade count slightly but quality should improve

Rationale: Win rate is the weakest component (39.3%). The R:R is excellent (7x)
and return is strong (335%). The composite is limited by win rate. Volume at the
crossover bar is a strong quality signal — genuine momentum breakouts tend to 
have elevated volume. Low-volume crosses are often noise/traps. Adding this 
filter should improve the hit rate of entries without fundamentally changing 
the strategy structure.
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
    "bxt_weekly_min": -5,         # weekly can be slightly negative

    # BX-Trender (daily)
    "bxt_daily_min": 8,           # relaxed: avoid over-filtering
    "bxt_daily_rising_bars": 1,   # bars over which daily BXT must be rising

    # Fast EMA pre-slope filter
    "ema_fast_slope_bars": 1,     # 1 bar is sufficient momentum check

    # ATR trailing stop
    "atr_length": 9,
    "atr_factor": 3.2,            # give winners room to run

    # Hard stop loss (last resort only)
    "stop_loss_pct": 0.12,        # wider stop to avoid cutting winners early

    # Volume confirmation filter (NEW)
    "volume_avg_bars": 20,        # lookback for average volume
    "volume_min_mult": 1.0,       # crossover bar volume must be >= this × avg volume

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
      2. Weekly BX-Trender > bxt_weekly_min (HTF context — weekly trend acceptable)
      3. Daily BX-Trender > bxt_daily_min AND rising (daily momentum confirmed)
      4. Current price is above the slow EMA (trend confirmation)
      5. Fast EMA was already rising for slope_bars before the crossover bar
      6. Volume on crossover bar >= volume_min_mult × avg_volume (conviction check)
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]
    rising_bars = params["bxt_daily_rising_bars"]
    slope_bars = params["ema_fast_slope_bars"]
    vol_avg_bars = params["volume_avg_bars"]
    vol_min_mult = params["volume_min_mult"]

    # Need enough history for indicators + slope check + volume avg
    min_bars_needed = max(slow + slope_bars + 5, vol_avg_bars + 2)
    if i < min_bars_needed:
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

    # Condition 6: Volume confirmation — crossover bar volume must be meaningful
    # Check if Volume column exists (some datasets may not have it)
    if "Volume" in df.columns:
        vol_series = df["Volume"].iloc[: i + 1]
        current_vol = vol_series.iloc[-1]
        # Use bars before the current one for the average to avoid self-referencing
        avg_vol = vol_series.iloc[-(vol_avg_bars + 1):-1].mean()
        if avg_vol > 0 and current_vol < vol_min_mult * avg_vol:
            return False

    # Condition 2: Weekly BX-Trender must be above minimum threshold
    bxt_w = _weekly_bx_trender(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_w is None or bxt_w <= params["bxt_weekly_min"]:
        return False

    # Condition 3: Daily BX-Trender — above threshold AND rising
    bxt_series = _daily_bx_trender_series(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_series is None or len(bxt_series) < rising_bars + 2:
        return False

    bxt_now = bxt_series.iloc[-1]

    # Must be above the minimum threshold
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
      1. Hard stop loss (last resort — 12% to avoid stopping out winners early)
      2. ATR trailing stop hit (factor=3.2 — gives trades room to develop)
      3. Fast EMA crosses below slow EMA
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]

    df_slice = df.iloc[: i + 1]
    close = df_slice["Close"]
    current_price = close.iloc[-1]

    # 1. Hard stop loss (wider to give volatile assets room to move)
    if current_price <= entry_price * (1 - params["stop_loss_pct"]):
        return True

    if len(close) < max(slow, params["atr_length"]) + 2:
        return False

    # 2. ATR trailing stop (factor=3.2 — gives winners room to run)
    trail = _atr_trailing_stop(df_slice, params["atr_length"], params["atr_factor"])
    if current_price < trail.iloc[-1]:
        return True

    # 3. EMA cross below
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return True

    return False