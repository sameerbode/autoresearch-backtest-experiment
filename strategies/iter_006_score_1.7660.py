"""
strategy.py — EMA Cross + Weekly BX-Trender + Daily BX-Trender + ATR Trailing Stop

Entry (all must be true):
  1. Daily fast EMA freshly crosses above slow EMA    ← entry trigger
  2. Weekly BX-Trender > bxt_weekly_min               ← HTF not in downtrend
  3. Daily BX-Trender > bxt_daily_min AND rising      ← daily momentum confirmed
  4. Price above slow EMA at entry (trend confirmation)

Exit (first to trigger wins):
  - Hard stop loss (last resort)
  - ATR trailing stop (factor=2.9 — give trades room to breathe, was 2.0 which was too tight)
  - Fast EMA crosses below slow EMA

Key change from v5:
  - ATR factor raised: 2.0 → 2.9 (2.0 was too tight, stopping out valid trends prematurely)
  - bxt_daily_min raised: 5 → 15 (require stronger daily momentum to improve entry quality)
  - bxt_weekly_min raised: -10 → 0 (only enter when weekly trend is neutral or positive)
  - Added price-above-slow-EMA filter (ensures we're entering in established uptrend)
  - rising_bars reduced: 2 → 1 (less strict on timing, momentum threshold does the work)
  
Rationale: Win rate of 28.6% is the critical failure. Most trades are losers.
The 2.0 ATR factor stops out trades that just need more room, while the high BXT
threshold (15) filters out weak crossovers that tend to fail. Weekly BXT > 0 ensures
we're only entering when the higher timeframe is at least neutral. Price above slow EMA
adds trend confirmation — crossovers that happen below the slow EMA are often false starts.
These changes aim to reduce trade count but significantly improve trade quality.
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
    "bxt_weekly_min": 0,          # raised from -10: require neutral/positive weekly trend

    # BX-Trender (daily)
    "bxt_daily_min": 15,          # raised from 5: require stronger daily momentum
    "bxt_daily_rising_bars": 1,   # reduced from 2: threshold does the filtering work

    # ATR trailing stop — loosened to give trades room to breathe
    "atr_length": 9,
    "atr_factor": 2.9,            # raised from 2.0: was cutting winners too early

    # Hard stop loss (last resort only)
    "stop_loss_pct": 0.10,        # slightly wider to match ATR factor increase
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
      2. Weekly BX-Trender > bxt_weekly_min (HTF context — neutral or positive)
      3. Daily BX-Trender > bxt_daily_min (stronger momentum required) AND rising
      4. Current price is above the slow EMA (trend confirmation)
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

    # Condition 4: Price must be above slow EMA (avoid false crossovers in downtrend)
    # Note: at a crossover, fast > slow, but close might still be near/below slow EMA
    # This ensures the actual close confirms the uptrend direction
    current_price = close.iloc[-1]
    if current_price <= ema_slow.iloc[-1]:
        return False

    # Condition 2: Weekly BX-Trender must be above minimum threshold (0 = neutral or positive)
    bxt_w = _weekly_bx_trender(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_w is None or bxt_w <= params["bxt_weekly_min"]:
        return False

    # Condition 3: Daily BX-Trender — above threshold AND rising
    bxt_series = _daily_bx_trender_series(df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"])
    if bxt_series is None or len(bxt_series) < rising_bars + 2:
        return False

    bxt_now = bxt_series.iloc[-1]

    # Must be above the minimum threshold (raised to 15 for stronger momentum filter)
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
      1. Hard stop loss (last resort, price dropped significantly)
      2. ATR trailing stop hit (factor=2.9 — gives trades room to develop)
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