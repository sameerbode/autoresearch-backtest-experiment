"""
strategy.py — EMA Cross + Weekly BX-Trender + Daily BX-Trender + Volume Confirmation + ATR Trailing Stop

Entry (all must be true):
  1. Daily fast EMA freshly crosses above slow EMA    ← entry trigger
  2. Weekly BX-Trender > bxt_weekly_min               ← HTF not in downtrend
  3. Daily BX-Trender > bxt_daily_min AND rising      ← daily momentum confirmed
  4. Volume on crossover bar > volume_ma_mult × 20-bar avg volume  ← institutional participation

Exit (first to trigger wins):
  - Price closes below ATR trailing stop (ATR length=9, factor=2.5)
  - Fast EMA crosses below slow EMA
  - Hard stop loss (last resort)

Key change from v3:
  - Added volume confirmation: crossover bar must have above-average volume
  - volume_ma_period = 20, volume_ma_mult = 1.0 (at least average volume)
  - bxt_daily_min relaxed back to 0 (volume filter now provides quality gate)
  - bxt_weekly_min relaxed to -10 (recover some trade count)
  - Keep atr_factor=2.5 from v3 (working well)
  - rising_bars kept at 2 for momentum confirmation
  
Rationale: Volume spikes on EMA crossovers indicate genuine breakouts with 
institutional backing. Low-volume crossovers tend to be false breakouts that
hurt win rate. This is the most direct lever to improve 34.5% → 50%+ win rate.
"""

import numpy as np
import pandas as pd

PARAMS = {
    # EMA cross
    "ema_fast": 8,
    "ema_slow": 21,

    # BX-Trender (weekly) — relaxed to -10 to recover trade count
    "bxt_l1": 5,
    "bxt_l2": 20,
    "bxt_l3": 5,
    "bxt_weekly_min": -10,

    # BX-Trender (daily) — relaxed threshold, volume filter provides quality gate
    "bxt_daily_min": 0,          # relaxed from 5 back to 0
    "bxt_daily_rising_bars": 2,  # daily BXT must be higher than N bars ago

    # Volume confirmation — crossover bar must have above-average volume
    "volume_ma_period": 20,      # period for average volume
    "volume_ma_mult": 1.0,       # multiplier: 1.0 = at or above average

    # ATR trailing stop — kept tight from v3
    "atr_length": 9,
    "atr_factor": 2.5,

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


def _volume_confirmed(df: pd.DataFrame, i: int, period: int, mult: float) -> bool:
    """
    Returns True if the volume at bar i is above mult × average volume 
    over the prior `period` bars.
    
    This confirms the EMA crossover has institutional participation —
    low-volume crossovers tend to be false breakouts.
    """
    if "Volume" not in df.columns:
        return True  # graceful fallback if no volume data

    if i < period + 1:
        return False

    # Use prior bars (not including current) for the average to avoid bias
    vol_series = df["Volume"].iloc[:i + 1]
    current_vol = vol_series.iloc[-1]

    if current_vol <= 0:
        return True  # skip filter if volume data is zero/missing

    # Average volume over prior `period` bars (excluding current bar)
    avg_vol = vol_series.iloc[-period - 1:-1].mean()

    if avg_vol <= 0:
        return True

    return current_vol >= mult * avg_vol


# ── Entry signal ──────────────────────────────────────────────────────────────

def get_entry_signal(df, i, params):
    """
    Enter long when ALL conditions are met:
      1. Fast EMA freshly crosses above slow EMA (today)
      2. Weekly BX-Trender > bxt_weekly_min (HTF context — not in deep downtrend)
      3. Daily BX-Trender > bxt_daily_min AND rising vs N bars ago
      4. Volume on crossover bar >= volume_ma_mult × 20-bar avg volume
         (confirms institutional participation, filters false breakouts)
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]
    rising_bars = params["bxt_daily_rising_bars"]

    if i < slow + 5:
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

    # Condition 4: Volume confirmation — crossover bar must have above-average volume
    if not _volume_confirmed(df, i, params["volume_ma_period"], params["volume_ma_mult"]):
        return False

    return True


# ── Exit signal ───────────────────────────────────────────────────────────────

def get_exit_signal(df, i, entry_price, params):
    """
    Exit when any of:
      1. Hard stop loss (last resort, price dropped significantly)
      2. ATR trailing stop hit (factor=2.5 — tight enough to cut losers)
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