"""
strategy.py — EMA Cross + Weekly BX-Trender + Daily BX-Trender + ATR Trailing Stop

Entry (all must be true):
  1. Daily fast EMA freshly crosses above slow EMA    ← entry trigger
  2. Weekly BX-Trender > bxt_weekly_min               ← HTF must be acceptable
  3. Daily BX-Trender > bxt_daily_min AND rising      ← daily momentum confirmed
  4. Price above slow EMA at entry (trend confirmation)
  5. Fast EMA slope positive for 1+ bar before cross  ← momentum building
  6. Volume on crossover bar > N-bar average volume   ← conviction filter
  7. Crossover bar closes in upper half of its range  ← bullish price action

Key change from v16:
  - ATR factor tightened: 3.2 → 2.5 (cut losers faster, improve win rate)
  - Added early exit: if daily BX-Trender drops below bxt_exit_min AFTER entry,
    exit early rather than waiting for full ATR stop to be hit
  - bxt_exit_min: -15 — only exits clearly failed momentum trades
  - Rationale: Win rate was 29.2% — the primary drag on composite score.
    With ATR factor 3.2, losing trades were giving back too much before stopping out.
    Tighter ATR (2.5) cuts losers while winners should still run via EMA cross exit.
    The BXT exit filter adds a momentum-based early exit for trades that fail quickly.
    
Rationale: Win rate of 29.2% with avg R/R of 7.6 means we're letting losers run
almost as long as winners before the trailing stop catches them. Tightening the
ATR factor to 2.5 should convert moderate losses into quick cuts (improving win rate)
while large winners will exit via EMA cross (which is a slower, more profitable exit).
The daily BXT early exit adds a second momentum-based cut for trades that fail fast.
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
    "bxt_weekly_min": -8,         # slightly more flexible weekly threshold

    # BX-Trender (daily)
    "bxt_daily_min": 5,           # entry filter — daily BXT must be above this
    "bxt_daily_rising_bars": 1,   # bars over which daily BXT must be rising

    # Early exit: if daily BXT drops below this level after entry, exit
    # Set to -15 so we only exit clearly failed momentum trades
    "bxt_exit_min": -15,

    # Fast EMA pre-slope filter
    "ema_fast_slope_bars": 1,     # 1 bar is sufficient momentum check

    # Volume confirmation
    "volume_ma_period": 20,       # period for volume moving average
    "volume_min_ratio": 0.7,      # relaxed — allow moderate volume crosses

    # Candle close position filter
    "close_range_min_pct": 0.35,  # close must be in top 65% of bar range

    # ATR trailing stop — TIGHTENED from 3.2 to 2.5 to cut losers faster
    "atr_length": 9,
    "atr_factor": 2.5,            # tighter: cut losers before they run too far

    # Hard stop loss (last resort only)
    "stop_loss_pct": 0.12,        # wider stop to avoid stopping out winners
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
      2. Weekly BX-Trender > bxt_weekly_min (HTF context)
      3. Daily BX-Trender > bxt_daily_min AND rising
      4. Current price is above the slow EMA
      5. Fast EMA was already rising for slope_bars before the crossover
      6. Volume on crossover bar >= volume_min_ratio × N-bar volume MA
      7. Close is in the upper portion of today's high-low range
    """
    fast = params["ema_fast"]
    slow = params["ema_slow"]
    rising_bars = params["bxt_daily_rising_bars"]
    slope_bars = params["ema_fast_slope_bars"]
    vol_period = params["volume_ma_period"]
    vol_ratio = params["volume_min_ratio"]
    close_range_min = params["close_range_min_pct"]

    # Need enough history for indicators + slope check + volume MA
    min_bars = max(slow, vol_period) + slope_bars + 5
    if i < min_bars:
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

    # Condition 6: Volume confirmation — crossover bar must have above-average volume
    if "Volume" in df.columns:
        volume = df["Volume"].iloc[: i + 1]
        if len(volume) >= vol_period + 1:
            vol_ma = volume.iloc[-(vol_period + 1):-1].mean()  # avg of prior N bars
            current_vol = volume.iloc[-1]
            if vol_ma > 0 and current_vol < vol_ratio * vol_ma:
                return False

    # Condition 7: Candle close position — close must be in upper portion of bar range
    bar_high = df["High"].iloc[i]
    bar_low  = df["Low"].iloc[i]
    bar_range = bar_high - bar_low
    if bar_range > 0:
        close_position = (current_price - bar_low) / bar_range
        if close_position < close_range_min:
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
      2. ATR trailing stop hit (factor=2.5 — tighter to cut losers faster)
      3. Daily BX-Trender drops below bxt_exit_min (momentum failure early exit)
      4. Fast EMA crosses below slow EMA
    
    Exit priority: hard stop → ATR trail → momentum failure → EMA cross
    The ATR trail is tighter (2.5 vs prior 3.2) so losing trades get cut sooner.
    Winners tend to stay above the ATR trail longer and exit via EMA cross.
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

    # 2. ATR trailing stop (factor=2.5 — tighter than before to cut losers faster)
    trail = _atr_trailing_stop(df_slice, params["atr_length"], params["atr_factor"])
    if current_price < trail.iloc[-1]:
        return True

    # 3. Daily BX-Trender momentum failure exit
    # If daily BXT drops deeply negative after entry, momentum has clearly failed
    bxt_exit_min = params.get("bxt_exit_min", -15)
    bxt_series = _daily_bx_trender_series(
        df, i, params["bxt_l1"], params["bxt_l2"], params["bxt_l3"]
    )
    if bxt_series is not None and len(bxt_series) >= 2:
        bxt_now = bxt_series.iloc[-1]
        if bxt_now < bxt_exit_min:
            return True

    # 4. EMA cross below — trend has reversed
    ema_fast_series = _ema(close, fast)
    ema_slow_series = _ema(close, slow)
    if ema_fast_series.iloc[-1] < ema_slow_series.iloc[-1]:
        return True

    return False