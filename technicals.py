"""
technicals.py — Calculate technical indicators and convert them to a 0–100 score.

What each indicator tells you (beginner primer):
  RSI  — Relative Strength Index: measures buying/selling momentum.
          30–70 is normal; <30 = oversold (possible bounce), >70 = overbought.
  MACD — Moving Average Convergence/Divergence: shows whether momentum is
          building or fading. Histogram > 0 means the bullish trend is gaining.
  SMA  — Simple Moving Average: smooths price over N days.
          Price above 20/50/200 SMA = short/medium/long-term uptrend.
  Volume ratio — compares recent 5-day volume to 20-day average.
          A spike (> 1.5×) often signals institutional buying or news.
  52-week position — where the price sits between its annual low and high.
          0 = at the 52-week low, 1 = at the 52-week high.
"""
import logging

import pandas as pd
import ta

from config import (
    TECH_WEIGHT_MACD,
    TECH_WEIGHT_MOMENTUM,
    TECH_WEIGHT_RSI,
    TECH_WEIGHT_TREND,
    TECH_WEIGHT_VOLUME,
)

logger = logging.getLogger(__name__)


def score_all_technicals(price_data: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Run technical analysis on every ticker. Returns scores keyed by symbol."""
    scores = {}
    for ticker, df in price_data.items():
        try:
            scores[ticker] = _analyze_one(df)
        except Exception as exc:
            logger.debug(f"{ticker}: indicator error — {exc}")
    return scores


def _analyze_one(df: pd.DataFrame) -> dict:
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi_series = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_obj = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    macd_line = macd_obj.macd()
    macd_hist = macd_obj.macd_diff()

    macd_val  = float(macd_line.iloc[-1]) if not macd_line.empty else 0.0
    macd_h    = float(macd_hist.iloc[-1]) if not macd_hist.empty else 0.0

    # ── Moving Averages ───────────────────────────────────────────────────────
    sma20  = _last(ta.trend.SMAIndicator(close=close, window=20).sma_indicator())
    sma50  = _last(ta.trend.SMAIndicator(close=close, window=50).sma_indicator()) if len(close) >= 50  else sma20
    sma200 = _last(ta.trend.SMAIndicator(close=close, window=200).sma_indicator()) if len(close) >= 200 else sma50
    price  = float(close.iloc[-1])

    # ── Volume ────────────────────────────────────────────────────────────────
    today_vol = float(volume.iloc[-1])
    avg_vol5  = float(volume.tail(5).mean())
    avg_vol20 = float(volume.tail(20).mean())
    avg_vol60 = float(volume.tail(60).mean()) if len(volume) >= 60 else avg_vol20

    # Three timeframe-appropriate ratios:
    #   day    — today's volume vs 20-day average (intraday surge / unusual day)
    #   swing  — 5-day vs 20-day (recent week vs typical month)  ← used for ranking
    #   long   — 20-day vs 60-day (recent month vs prior quarter, accumulation)
    vol_ratio_day  = (today_vol / avg_vol20) if avg_vol20 > 0 else 1.0
    vol_ratio      = (avg_vol5  / avg_vol20) if avg_vol20 > 0 else 1.0   # swing window
    vol_ratio_long = (avg_vol20 / avg_vol60) if avg_vol60 > 0 else 1.0

    # Direction-aware volume: split last 20 days' volume into "up" and "down"
    # bars based on day-over-day close change. up_pct + down_pct ≈ 1.0
    # (flat days are excluded).
    up_pct, down_pct = _up_down_volume_pct(df, lookback=20)

    # ── 52-week position ──────────────────────────────────────────────────────
    window = min(len(close), 252)
    hi52   = float(close.tail(window).max())
    lo52   = float(close.tail(window).min())
    rng52  = hi52 - lo52
    pos52  = (price - lo52) / rng52 if rng52 > 0 else 0.5

    # 5-day return in percent
    ret5d  = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0

    # ── Volume scores (direction-aware) ───────────────────────────────────────
    # Base volume intensity (same as before: vol_ratio drives the magnitude).
    vol_sc_base = _score_volume(vol_ratio)
    # Amplify the side that matches the dominant direction of recent volume.
    # 0.5 + up_pct gives a 0.5×–1.5× multiplier (capped at 100).
    vol_sc_bull = min(100.0, vol_sc_base * (0.5 + up_pct))
    vol_sc_bear = min(100.0, vol_sc_base * (0.5 + down_pct))

    # ── Bullish component scores (each 0–100) ────────────────────────────────
    rsi_sc  = _score_rsi(rsi)
    macd_sc = _score_macd(macd_h, macd_val)
    tnd_sc  = _score_trend(price, sma20, sma50, sma200)
    mom_sc  = _score_momentum(pos52, ret5d)

    bull_composite = (
        rsi_sc      * TECH_WEIGHT_RSI      +
        macd_sc     * TECH_WEIGHT_MACD     +
        tnd_sc      * TECH_WEIGHT_TREND    +
        vol_sc_bull * TECH_WEIGHT_VOLUME   +
        mom_sc      * TECH_WEIGHT_MOMENTUM
    )

    # ── Bearish component scores (mirror of bullish, looking for downtrends) ─
    rsi_b   = _score_rsi_bearish(rsi)
    macd_b  = _score_macd_bearish(macd_h, macd_val)
    tnd_b   = _score_trend_bearish(price, sma20, sma50, sma200)
    mom_b   = _score_momentum_bearish(pos52, ret5d)

    bear_composite = (
        rsi_b       * TECH_WEIGHT_RSI      +
        macd_b      * TECH_WEIGHT_MACD     +
        tnd_b       * TECH_WEIGHT_TREND    +
        vol_sc_bear * TECH_WEIGHT_VOLUME   +
        mom_b       * TECH_WEIGHT_MOMENTUM
    )

    # ── Day-trade suitability score (0–100) ──────────────────────────────────
    # Computed for ALL stocks (cheap, price/volume only — no fundamentals).
    # Used to rank a separate "top day-trade picks" list.
    atr_series = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    atr     = float(atr_series.iloc[-1]) if not atr_series.empty else price * 0.015
    atr_pct = (atr / price) * 100
    daily_rng_pct = float(((high - low) / close).tail(20).mean() * 100)

    liquidity_pts = min(25, (avg_vol20 / 1_000_000) * 5)               # 5M shares → 25
    intensity_pts = min(20, max(0.0, vol_ratio_day - 1.0) * 25)        # 1.8× → 20
    atr_pts       = min(30, atr_pct * 7.5)                              # 4% ATR → 30
    range_pts     = min(25, daily_rng_pct * 5)                          # 5% range → 25
    day_score     = liquidity_pts + intensity_pts + atr_pts + range_pts

    return {
        "technical_score":  round(bull_composite, 2),
        "bearish_score":    round(bear_composite, 2),
        "last_price":       round(price, 2),
        "rsi":              round(rsi, 2),
        "rsi_score":        round(rsi_sc, 2),
        "macd_hist":        round(macd_h, 4),
        "macd_score":       round(macd_sc, 2),
        "sma20":            round(sma20, 2),
        "sma50":            round(sma50, 2),
        "sma200":           round(sma200, 2),
        "trend_score":      round(tnd_sc, 2),
        "vol_ratio":        round(vol_ratio, 2),       # 5d / 20d (swing window)
        "vol_ratio_day":    round(vol_ratio_day, 2),   # today / 20d (day window)
        "vol_ratio_long":   round(vol_ratio_long, 2),  # 20d / 60d (long window)
        "vol_up_pct":       round(up_pct, 3),
        "vol_down_pct":     round(down_pct, 3),
        "volume_score":     round(vol_sc_bull, 2),
        "bear_volume_score": round(vol_sc_bear, 2),
        "week52_pos":       round(pos52, 3),
        "ret5d_pct":        round(ret5d, 2),
        "momentum_score":   round(mom_sc, 2),
        # Day-trade suitability (price/volume only, no fundamentals)
        "atr":              round(atr, 2),
        "atr_pct":          round(atr_pct, 2),
        "daily_range_pct":  round(daily_rng_pct, 2),
        "day_score":        round(day_score, 2),
    }


# ── Scoring functions (0–100 each) ────────────────────────────────────────────

def _score_rsi(rsi: float) -> float:
    """
    Momentum strategy: best zone is RSI 50–65 (uptrend, not yet overbought).
    """
    if 50 <= rsi <= 65:  return 85.0
    if 65 < rsi <= 70:   return 70.0
    if 45 <= rsi < 50:   return 60.0
    if 40 <= rsi < 45:   return 45.0
    if 30 <= rsi < 40:   return 30.0
    if rsi > 70:         return 50.0   # overbought — caution
    return 20.0                         # < 30 — weak / oversold


def _score_macd(hist: float, macd: float) -> float:
    """
    Reward positive histogram (bullish momentum building).
    Best signal: histogram AND MACD line both above zero.
    """
    if hist > 0 and macd > 0:  return 85.0
    if hist > 0 and macd < 0:  return 65.0   # improving but still negative
    if hist < 0 and macd > 0:  return 40.0   # fading but positive territory
    return 20.0                                # both negative — downtrend


def _score_trend(price: float, sma20: float, sma50: float, sma200: float) -> float:
    """
    Award 25 pts per moving-average alignment level.
    Perfect bull stack: price > SMA20 > SMA50 > SMA200 = 100.
    """
    score = 0.0
    if price > sma20:              score += 25
    if price > sma50:              score += 25
    if price > sma200:             score += 25
    if sma20 > sma50 > sma200:     score += 25   # "golden" alignment
    return score


def _score_volume(ratio: float) -> float:
    """Elevated volume vs 20-day avg suggests institutional activity."""
    if ratio >= 2.0:   return 90.0
    if ratio >= 1.5:   return 75.0
    if ratio >= 1.0:   return 55.0
    if ratio >= 0.7:   return 35.0
    return 15.0


def _score_momentum(pos52: float, ret5d: float) -> float:
    """
    Reward stocks near their 52-week high with positive short-term momentum.
    pos52: 0.0 = at annual low, 1.0 = at annual high.
    """
    pos_pts = pos52 * 60                            # 0–60 from 52-wk position
    ret_pts = min(max(ret5d * 4, -20), 40)          # ±5% → ±20 pts, capped at 40
    return max(0.0, min(100.0, pos_pts + ret_pts))


def _last(series: pd.Series) -> float:
    """Safe last-value helper."""
    if series is None or series.empty:
        return 0.0
    val = series.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _up_down_volume_pct(df: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    """
    Split the last `lookback` days' volume into up-day vs down-day buckets
    (based on day-over-day close change) and return the percentages.

    Returns (up_pct, down_pct) summing to ~1.0; flat days are excluded.
    Used to make the volume score direction-aware: a 2× volume spike on
    declining prices is bearish confirmation, not bullish.
    """
    close  = df["Close"].astype(float).tail(lookback + 1)
    volume = df["Volume"].astype(float).tail(lookback)
    diff   = close.diff().tail(lookback)

    up_vol   = float(volume[diff > 0].sum())
    down_vol = float(volume[diff < 0].sum())
    total    = up_vol + down_vol
    if total <= 0:
        return 0.5, 0.5
    return up_vol / total, down_vol / total


# ── Bearish scoring (mirror of bullish, looking for downtrends) ──────────────

def _score_rsi_bearish(rsi: float) -> float:
    """
    Bearish RSI zones:
      30–45 = downtrending with room to fall further (best for puts)
      75+   = topping/overbought, possible mean reversion (decent for puts)
      < 30  = oversold, dangerous for puts (likely to bounce — penalize)
    """
    if 30 <= rsi <= 45:  return 85.0
    if 45 < rsi <= 55:   return 50.0
    if rsi > 75:         return 60.0   # overbought, possible reversal setup
    if rsi > 60:         return 25.0   # still in uptrend zone
    return 30.0                         # < 30 oversold — risky to short


def _score_macd_bearish(hist: float, macd: float) -> float:
    """Reward negative histogram (bearish momentum building)."""
    if hist < 0 and macd < 0:  return 90.0
    if hist < 0 and macd > 0:  return 65.0   # weakening uptrend
    if hist > 0 and macd < 0:  return 35.0   # bouncing inside downtrend
    return 15.0                                # both positive = bullish


def _score_trend_bearish(price: float, sma20: float, sma50: float, sma200: float) -> float:
    """
    Award 25 pts per moving-average alignment level (downward).
    Perfect bear stack: price < SMA20 < SMA50 < SMA200 = 100.
    """
    score = 0.0
    if price < sma20:              score += 25
    if price < sma50:              score += 25
    if price < sma200:             score += 25
    if sma20 < sma50 < sma200:     score += 25   # "death" alignment
    return score


def _score_momentum_bearish(pos52: float, ret5d: float) -> float:
    """
    Reward stocks near 52-week LOW with negative short-term momentum.
    pos52: 0.0 = at annual low, 1.0 = at annual high — invert.
    """
    pos_pts = (1 - pos52) * 60                          # 0–60 from inverse 52-wk position
    ret_pts = min(max(-ret5d * 4, -20), 40)             # -5% return → +20 pts
    return max(0.0, min(100.0, pos_pts + ret_pts))
