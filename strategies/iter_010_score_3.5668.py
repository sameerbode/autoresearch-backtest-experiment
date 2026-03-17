"""
strategy.py — EMA Cross + Weekly BX-Trender + Daily BX-Trender + ATR Trailing Stop

Entry (all must be true):
  1. Daily fast EMA freshly crosses above slow EMA    ← entry trigger
  2. Weekly BX-Trender > bxt_weekly_min               ← HTF must be acceptable
  3. Daily BX-Trender > bxt_daily_min                 ← daily momentum confirmed
  4. Price above slow EMA at entry (trend confirmation)
  5. Fast EMA slope positive for 1+ bar before cross  ← momentum building
  6. Volume on crossover bar > volume_ma_mult × avg volume  ← volume confirmation

Exit (first to trigger wins):
  - Hard stop loss (last resort only)
  - ATR trailing stop (give trades room to breathe)
  - Fast EMA crosses below slow EMA

Key change from v9:
  - Removed the "daily BXT rising" requirement — this was over-filtering and rejecting
    valid setups where BXT was already elevated but flat (still bullish momentum)
  - Relaxed bxt_daily_min: 8 → 3 to allow more quality setups while still requiring
    positive-ish daily momentum
  - Loosened volume filter: 1.0x → 0.8x avg volume — was too strict, rejecting valid
    setups on moderate volume
  - ATR factor: 2.8 → 3.2 — the tighter stop was stopping winners out too early
    (R/R is capped at 3.0 anyway, but win rate suffers from premature exits)

Rationale: Win rate of 37.5% is the composite's weakest component. The "rising BXT"
requirement + high bxt_daily_min + tight ATR was creating a triple-filter that rejected
too many valid setups. With only 16 trades, we need both more trades AND better quality.
Removing the rising filter while keeping BXT > threshold ensures momentum is present
without requiring it to be accelerating at the exact moment of the EMA cross.
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

    # BX-Trender (daily) — relaxed to allow more setups
    "bxt_daily_min": 3,           # was 8 — daily momentum just needs to be present
    "bxt_daily_rising_bars": 0,   # 0 = disabled — don't require BXT to be rising

    # Fast EMA pre-slope filter
    "ema_fast_slope_bars": 1,     # 1-bar slope check before crossover

    # Volume confirmation — slightly loosened from v9
    "volume_ma_period": 20,       # lookback for average volume
    "volume_ma_mult": 0.8,        # was 1.0 — loosened to avoid rejecting moderate-volume entries

    # ATR trailing stop — restored to wider setting to avoid stopping winners early
    "atr_length": 9,
    "atr_factor": 3.2,            # restored from 2.8 — wider gives trends room to develop

    # Hard stop loss (last resort only)
    "stop_loss_pct": 0.12,        # 12% hard stop — wide enough for volatile assets
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


def _volume_ma(df: pd.DataFrame, as_of_bar: int, period: int) -> float:
    """Average volume over the last `period` bars (excluding the current bar)."""
    if "Volume" not in df.columns:
        return None
    vol = df["Volume"].iloc[: as_of_bar + 1]
    if len(vol) < period + 1:
        return None
    # Average of the previous `period` bars (not including current bar)
    return vol.iloc[-(period + 1):-1].mean()


# ── Entry signal ──────────────────────────────────────────────────────────────

def get_entry_signal(df, i, params):
    """
    Enter long when ALL conditions are met:
      1. Fast EMA freshly crosses above slow EMA (today)
      2. Weekly BX-Trender > bxt_weekly_min (HTF context — weekly trend acceptable)
      3. Daily BX-Trender > bxt_daily_min (daily momentum present — no rising req.)
      4. Current price is above the slow EMA (trend confirmation)
      5. Fast EMA was already rising for slope_bars before the crossover bar
      6. Volume on crossover bar > volume_ma_mult × avg volume (conviction check)
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]
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
    ema_fast_before_cross = ema_fast.iloc[-2]
    ema_fast_slope_start = ema_fast.iloc[-(2 + slope_bars)]
    if ema_fast_before_cross <= ema_fast_slope_start:
        return False

    # Condition 6: Volume confirmation — crossover bar must have above-average volume
    vol_period = params["volume_ma_period"]
    vol_mult = params["volume_ma_mult"]
    avg_vol = _volume_ma(df, i, vol_period)
    if avg_vol is not None and avg_vol > 0:
        current_vol = df["Volume"].iloc[i] if "Volume" in df.columns else None
        if current_vol is not None:
            if current_vol < vol_mult * avg_vol:
                return False
    # If volume data is not available, skip this filter (fail open)

    # Condition 2: Weekly BX-Trender must be above minimum threshold
    bxt_w = _weekly_bx_trender(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_w is None or bxt_w <= params["bxt_weekly_min"]:
        return False

    # Condition 3: Daily BX-Trender — just needs to be above threshold
    # (removed "rising" requirement — it was over-filtering valid setups)
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
      1. Hard stop loss (last resort — 12% wide to avoid stopping out winners early)
      2. ATR trailing stop hit (factor=3.2 — restored wider to give trends room)
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

    # 2. ATR trailing stop (factor=3.2 — wider than v9 to reduce premature exits)
    trail = _atr_trailing_stop(df_slice, params["atr_length"], params["atr_factor"])
    if current_price < trail.iloc[-1]:
        return True

    # 3. EMA cross below
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return True

    return False