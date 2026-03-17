"""
strategy.py — EMA Cross + Weekly BX-Trender + Daily BX-Trender + ATR Trailing Stop

Entry (all must be true):
  1. Daily fast EMA freshly crosses above slow EMA    ← entry trigger
  2. Weekly BX-Trender > bxt_weekly_min               ← HTF must be acceptable
  3. Daily BX-Trender > bxt_daily_min AND rising      ← daily momentum confirmed
  4. Price above slow EMA at entry (trend confirmation)
  5. Fast EMA slope positive for 1+ bar before cross  ← momentum building
  6. Volume on crossover bar > volume_ma_mult × avg volume  ← NEW: volume confirmation

Exit (first to trigger wins):
  - Hard stop loss (last resort only)
  - ATR trailing stop (give trades room to breathe)
  - Fast EMA crosses below slow EMA

Key change from v8:
  - Added volume confirmation filter: crossover bar must have above-average volume
    (volume_ma_mult=1.0 means just above average, filters silent/low-conviction crosses)
  - Slightly tightened ATR factor: 3.2 → 2.8 to reduce give-back on losers
    (win rate was 39.3% — the wide trailing stop was allowing too many losers to run deep)
  - Volume filter rationale: low-volume EMA crossovers are notoriously unreliable;
    requiring above-average volume confirms institutional participation

Rationale: Win rate of 39.3% is the score's weakest component. R/R is already capped
at 3.0 in the composite. The fastest path to higher score is improving win rate from
39.3% toward 50%+. Volume confirmation is a classic momentum filter — crossovers on
strong volume have much higher follow-through rates than silent, low-volume crosses.
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
    "bxt_daily_min": 8,           # relaxed: allow more setups through
    "bxt_daily_rising_bars": 1,   # bars over which daily BXT must be rising

    # Fast EMA pre-slope filter
    "ema_fast_slope_bars": 1,     # 1-bar slope check before crossover

    # Volume confirmation — NEW in v9
    "volume_ma_period": 20,       # lookback for average volume
    "volume_ma_mult": 1.0,        # crossover bar volume must be > mult × avg volume
                                  # 1.0 = just above average; try 0.8–1.5

    # ATR trailing stop — tightened slightly to reduce give-back on losers
    "atr_length": 9,
    "atr_factor": 2.8,            # reduced from 3.2: tighter trailing stop improves win rate

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
      3. Daily BX-Trender > bxt_daily_min AND rising (daily momentum confirmed)
      4. Current price is above the slow EMA (trend confirmation)
      5. Fast EMA was already rising for slope_bars before the crossover bar
      6. Volume on crossover bar > volume_ma_mult × avg volume (NEW: conviction check)
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
    ema_fast_before_cross = ema_fast.iloc[-2]
    ema_fast_slope_start = ema_fast.iloc[-(2 + slope_bars)]
    if ema_fast_before_cross <= ema_fast_slope_start:
        return False

    # Condition 6: Volume confirmation — crossover bar must have above-average volume
    # This filters out low-conviction, silent crossovers
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
      2. ATR trailing stop hit (factor=2.8 — tightened to reduce give-back on losers)
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

    # 2. ATR trailing stop (factor=2.8 — tighter than v8 to improve win rate)
    trail = _atr_trailing_stop(df_slice, params["atr_length"], params["atr_factor"])
    if current_price < trail.iloc[-1]:
        return True

    # 3. EMA cross below
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return True

    return False