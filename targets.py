"""
targets.py — Price target calculations and trade-style suitability scoring.

Price targets (three timeframes):
  Day trade   : Entry ± ATR(14) × multiplier  (ATR = average daily range)
  Swing trade : 3× ATR above price OR next Fibonacci resistance (1–4 weeks)
  Long hold   : Analyst consensus price target from yfinance, or trend projection

Trade style suitability (0–100 score, ★ rating):
  Day trading  — needs high intraday range + very high volume
  Swing trading— needs clean technical setup + medium-term momentum
  Long holding — needs price above 200 SMA + solid fundamentals

Learner glossary:
  ATR  — Average True Range: average daily price movement in dollars.
         E.g. ATR=$5 means the stock typically moves $5 in a day.
  Fibonacci retracement — traders use the ratios 0.382, 0.5, 0.618 to
         find price levels where reversals often happen.
  P/E  — Price-to-Earnings ratio: price ÷ earnings per share.
         < 20 = cheap, 20–40 = normal growth, > 40 = expensive or high growth.
"""
import logging

import pandas as pd
import ta

logger = logging.getLogger(__name__)


def calculate_price_targets(df: pd.DataFrame, fundamentals: dict) -> dict:
    """
    Return day / swing / long price targets, stops, and risk:reward for a stock.
    """
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    price = float(close.iloc[-1])

    # ── ATR (Average True Range) ──────────────────────────────────────────────
    # Measures the typical daily move — the foundation for all our targets.
    atr_series = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    atr     = float(atr_series.iloc[-1]) if not atr_series.empty else price * 0.015
    atr_pct = (atr / price) * 100

    # ── Key price levels ──────────────────────────────────────────────────────
    resistance_20 = float(high.tail(20).max())
    support_20    = float(low.tail(20).min())
    window_52     = min(len(close), 252)
    resistance_52 = float(high.tail(window_52).max())

    # ── Day Trade Targets ─────────────────────────────────────────────────────
    # Risk:reward target = 1.5:1 (risk 1× ATR to make 1.5× ATR).
    day_target = round(price + 1.5 * atr, 2)
    day_stop   = round(price - 1.0 * atr, 2)

    # ── Swing Trade Target (1–4 weeks) ────────────────────────────────────────
    # If price recently broke above 20-day resistance, project the next level
    # using a Fibonacci 0.618 extension. Otherwise use 3× ATR from price.
    if price >= resistance_20 * 0.98:    # at or above recent resistance
        range_of_move = resistance_20 - support_20
        fib_target    = resistance_20 + 0.618 * range_of_move
        swing_target  = round(min(fib_target, price * 1.25), 2)   # cap at +25%
    else:
        swing_target = round(min(price + 3.0 * atr, resistance_20), 2)

    swing_stop = round(price - 1.5 * atr, 2)

    # ── Long-term Target ──────────────────────────────────────────────────────
    # Prefer analyst consensus (the average of all Wall St. price targets).
    analyst_mean = fundamentals.get("targetMeanPrice")
    analyst_high = fundamentals.get("targetHighPrice")
    analyst_low  = fundamentals.get("targetLowPrice")

    # long_target_basis is the short label for the "Target Src" column. The
    # fallback is not an analyst number, and the table must say so.
    if analyst_mean and float(analyst_mean) > price:
        long_target       = round(float(analyst_mean), 2)
        long_target_src   = "analyst consensus"
        long_target_basis = "Analyst"
    else:
        long_target     = round(max(resistance_52, price * 1.15), 2)
        long_target_src = "52-week high / 15% projection"
        basis    = "52w high" if resistance_52 >= price * 1.15 else "+15% rule"
        coverage = "analysts below" if analyst_mean else "no analysts"
        long_target_basis = f"{basis} ({coverage})"
        analyst_high = analyst_low = None

    return {
        "current_price":      price,
        "atr":                round(atr, 2),
        "atr_pct":            round(atr_pct, 2),
        # Day
        "day_target":         day_target,
        "day_stop":           day_stop,
        "day_target_pct":     round((day_target / price - 1) * 100, 2),
        "day_risk_pct":       round((1 - day_stop / price) * 100, 2),
        "day_rr":             "1 : 1.5",
        # Swing
        "swing_target":       swing_target,
        "swing_stop":         swing_stop,
        "swing_target_pct":   round((swing_target / price - 1) * 100, 2),
        "swing_risk_pct":     round((1 - swing_stop / price) * 100, 2),
        # Long
        "long_target":        long_target,
        "long_target_pct":    round((long_target / price - 1) * 100, 2),
        "long_target_src":    long_target_src,
        "long_target_basis":  long_target_basis,
        "analyst_mean":       round(float(analyst_mean), 2) if analyst_mean else None,
        "analyst_high":       round(float(analyst_high), 2) if analyst_high else None,
        "analyst_low":        round(float(analyst_low), 2)  if analyst_low  else None,
        # Levels
        "support_20":         round(support_20, 2),
        "resistance_20":      round(resistance_20, 2),
        "resistance_52w":     round(resistance_52, 2),
    }


def calculate_bearish_levels(df: pd.DataFrame) -> dict:
    """
    For BEARISH picks, project DOWNWARD targets (where the stock might fall to)
    instead of the bullish upward targets in calculate_price_targets().

    Levels returned:
      day_target_down   = price − 1.5 × ATR  (one-day downside projection)
      swing_target_down = max(price − 3 × ATR, 20-day support)  (1–4 wk decline)
      long_target_down  = max(price × 0.85, 52-wk low)  (sustained downtrend)
      day_stop / swing_stop  = upper levels where the bearish thesis breaks down
                               (if price climbs above, exit the put)
    """
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    price = float(close.iloc[-1])

    atr_series = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    atr     = float(atr_series.iloc[-1]) if not atr_series.empty else price * 0.015
    atr_pct = (atr / price) * 100

    support_20    = float(low.tail(20).min())
    resistance_20 = float(high.tail(20).max())
    window_52     = min(len(close), 252)
    support_52    = float(low.tail(window_52).min())

    day_target   = round(max(0.0, price - 1.5 * atr), 2)
    swing_target = round(max(price - 3.0 * atr, support_20), 2)
    long_target  = round(max(price * 0.85, support_52), 2)

    day_stop   = round(price + 1.0 * atr, 2)
    swing_stop = round(price + 1.5 * atr, 2)

    return {
        "current_price":     price,
        "atr":               round(atr, 2),
        "atr_pct":           round(atr_pct, 2),
        # Downward targets
        "day_target":        day_target,
        "day_stop":           day_stop,
        "day_target_pct":    round((day_target / price - 1) * 100, 2),
        "day_risk_pct":      round((day_stop / price - 1) * 100, 2),
        "day_rr":            "1 : 1.5",
        "swing_target":      swing_target,
        "swing_stop":         swing_stop,
        "swing_target_pct":  round((swing_target / price - 1) * 100, 2),
        "swing_risk_pct":    round((swing_stop / price - 1) * 100, 2),
        "long_target":       long_target,
        "long_target_pct":   round((long_target / price - 1) * 100, 2),
        "long_target_src":   "ATR / 52-week-low projection",
        # Levels
        "support_20":        round(support_20, 2),
        "resistance_20":     round(resistance_20, 2),
        "support_52w":       round(support_52, 2),
    }


def score_trade_styles(df: pd.DataFrame, tech: dict, fundamentals: dict) -> dict:
    """
    Rate the stock's suitability for day / swing / long strategies.

    Each timeframe uses a volume window appropriate to its horizon:
      Day   — today's volume vs 20-day average  (intraday surge candidate)
      Swing — 5-day vs 20-day                   (recent week of interest)
      Long  — 20-day vs 60-day                  (sustained accumulation)
    """
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    price  = float(close.iloc[-1])

    atr_series = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    atr     = float(atr_series.iloc[-1]) if not atr_series.empty else price * 0.015
    atr_pct = (atr / price) * 100

    avg_vol20      = float(volume.tail(20).mean())
    daily_rng      = float(((high - low) / close).tail(20).mean() * 100)
    vol_ratio_day  = tech.get("vol_ratio_day",  1.0)   # today vs 20d
    vol_ratio_swing = tech.get("vol_ratio",     1.0)   # 5d vs 20d
    vol_ratio_long = tech.get("vol_ratio_long", 1.0)   # 20d vs 60d

    # ── Day Trade (0–100) ─────────────────────────────────────────────────────
    # Liquidity (25) + intensity-today-vs-20d (20) + ATR% (30) + daily range (25)
    liquidity_pts = min(25, (avg_vol20 / 1_000_000) * 5)               # 5M shares → 25 pts
    intensity_pts = min(20, max(0.0, vol_ratio_day - 1.0) * 25)        # 1.8× → 20 pts
    atr_pts       = min(30, atr_pct * 7.5)                              # 4% ATR → 30 pts
    range_pts     = min(25, daily_rng * 5)                              # 5% range → 25 pts
    day_score     = liquidity_pts + intensity_pts + atr_pts + range_pts

    # ── Swing Trade (0–100) ───────────────────────────────────────────────────
    rsi         = tech.get("rsi", 50)
    rsi_pts     = 25 if 45 <= rsi <= 68 else (12 if 35 <= rsi < 45 else 5)
    tech_pts    = tech.get("technical_score", 50) * 0.55
    vol_r_pts   = min(20, vol_ratio_swing * 10)
    swing_score = min(100, tech_pts + rsi_pts + vol_r_pts)

    # ── Long Hold (0–100) ─────────────────────────────────────────────────────
    # 200 SMA (25) + P/E (22) + revenue growth (22) + margins (16) + accumulation (15)
    sma200       = tech.get("sma200", price)
    above_pts    = 25 if price > sma200 else 0

    pe           = fundamentals.get("forwardPE") or fundamentals.get("trailingPE")
    rev_growth   = fundamentals.get("revenueGrowth") or 0
    margins      = fundamentals.get("profitMargins") or 0

    pe_pts       = 22 if pe and 0 < pe < 25 else (15 if pe and 0 < pe < 40 else 6)
    growth_pts   = min(22, float(rev_growth) * 130) if rev_growth else 8
    margin_pts   = min(16, float(margins) * 80)     if margins    else 6
    # Accumulation: 20-day avg up vs 60-day avg signals institutions building.
    # 1.20× = full 15 pts; 1.0× = 0 pts; below 1.0 = 0 (distribution, not accumulation).
    accum_pts    = min(15, max(0.0, vol_ratio_long - 1.0) * 75)
    long_score   = min(100, above_pts + pe_pts + growth_pts + margin_pts + accum_pts)

    return {
        "day_score":        round(day_score, 1),
        "day_stars":        _stars(day_score),
        "day_label":        _label(day_score),
        "day_note":         _day_note(avg_vol20, atr_pct, vol_ratio_day),
        "swing_score":      round(swing_score, 1),
        "swing_stars":      _stars(swing_score),
        "swing_label":      _label(swing_score),
        "swing_note":       _swing_note(rsi, tech.get("macd_hist", 0)),
        "long_score":       round(long_score, 1),
        "long_stars":       _stars(long_score),
        "long_label":       _label(long_score),
        "long_note":        _long_note(price, sma200, pe, rev_growth, vol_ratio_long),
        "avg_volume":       int(avg_vol20),
        "atr_pct":          round(atr_pct, 2),
        "daily_range_pct":  round(daily_rng, 2),
        "vol_ratio_day":    round(vol_ratio_day, 2),
        "vol_ratio_long":   round(vol_ratio_long, 2),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stars(score: float) -> str:
    s = max(0, min(5, round(score / 20)))
    return "★" * s + "☆" * (5 - s)


def _label(score: float) -> str:
    if score >= 75: return "Excellent"
    if score >= 60: return "Good"
    if score >= 40: return "Fair"
    if score >= 20: return "Poor"
    return "Not Suitable"


def _day_note(avg_vol: float, atr_pct: float, vol_ratio_day: float = 1.0) -> str:
    vol_m = avg_vol / 1_000_000
    today_note = ""
    if vol_ratio_day >= 1.5:
        today_note = f" Today's volume is {vol_ratio_day:.1f}× the 20-day average — unusual interest."
    elif vol_ratio_day < 0.7:
        today_note = f" Today's volume is only {vol_ratio_day:.1f}× normal — quiet session."
    base = (
        "Good intraday liquidity and range." if vol_m > 1 and atr_pct > 1.5
        else "Low volume or tight range — day trading is difficult."
    )
    return f"Avg volume {vol_m:.1f}M shares/day, ATR {atr_pct:.1f}% of price. {base}{today_note}"


def _swing_note(rsi: float, macd_hist: float) -> str:
    rsi_str  = "RSI in momentum zone" if 45 <= rsi <= 68 else f"RSI {rsi:.0f} — outside ideal swing zone"
    macd_str = "MACD bullish" if macd_hist > 0 else "MACD bearish"
    return f"{rsi_str}, {macd_str}."


def _long_note(price: float, sma200: float, pe, rev_growth, vol_ratio_long: float = 1.0) -> str:
    trend = "Above 200 SMA (long-term uptrend)" if price > sma200 else "Below 200 SMA (long-term downtrend — caution)"
    pe_str = f"P/E {pe:.1f}" if pe else "P/E N/A"
    gr_str = f"Revenue growth {float(rev_growth)*100:.0f}%/yr" if rev_growth else ""
    parts  = [trend, pe_str]
    if gr_str:
        parts.append(gr_str)
    if vol_ratio_long >= 1.15:
        parts.append(f"20d/60d volume {vol_ratio_long:.2f}× — sustained accumulation")
    elif vol_ratio_long <= 0.85:
        parts.append(f"20d/60d volume {vol_ratio_long:.2f}× — interest fading")
    return ", ".join(parts) + "."
