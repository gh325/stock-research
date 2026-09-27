"""
scorer.py — Combines technical and sentiment scores into a final ranking.

Two-phase design:
  Phase 1: Score ALL stocks by technicals → pick top N candidates (fast)
  Phase 2: Run sentiment only on those N candidates (avoids 4 000 news fetches)
  Phase 3: Merge both scores → final sorted list → top 5 picks
"""
import logging

from config import TOP_CANDIDATES, TOP_PICKS, WEIGHT_SENTIMENT, WEIGHT_TECHNICAL

logger = logging.getLogger(__name__)


def get_top_candidates(
    technical_scores: dict[str, dict], n: int = TOP_CANDIDATES
) -> list[str]:
    """
    Sort all tickers by technical score and return the top-N symbols.
    These are the only stocks we'll spend time fetching news for.
    """
    ranked = sorted(
        technical_scores.items(),
        key=lambda x: x[1].get("technical_score", 0.0),
        reverse=True,
    )
    top = [ticker for ticker, _ in ranked[:n]]
    logger.info(f"Phase 1: {len(technical_scores)} stocks → top {len(top)} candidates")
    return top


def final_ranking(
    technical_scores: dict[str, dict],
    sentiment_scores: dict[str, dict],
) -> list[dict]:
    """
    Merge technical and sentiment scores into one combined ranking.
    Returns all stocks sorted best → worst.
    """
    merged: list[dict] = []

    for ticker, tech in technical_scores.items():
        sent = sentiment_scores.get(ticker, {"sentiment_score": 50.0, "headline_count": 0})

        tech_score = tech.get("technical_score", 0.0)
        sent_score = sent.get("sentiment_score", 50.0)
        combined   = tech_score * WEIGHT_TECHNICAL + sent_score * WEIGHT_SENTIMENT

        merged.append({
            "ticker":           ticker,
            "combined_score":   round(combined, 2),
            "technical_score":  round(tech_score, 2),
            "sentiment_score":  round(sent_score, 2),
            # Technical detail for the report
            "last_price":       tech.get("last_price", 0.0),
            "rsi":              tech.get("rsi", 0.0),
            "macd_hist":        tech.get("macd_hist", 0.0),
            "trend_score":      tech.get("trend_score", 0.0),
            "vol_ratio":        tech.get("vol_ratio", 0.0),
            "vol_ratio_day":    tech.get("vol_ratio_day", 1.0),
            "vol_ratio_long":   tech.get("vol_ratio_long", 1.0),
            "vol_up_pct":       tech.get("vol_up_pct", 0.5),
            "vol_down_pct":     tech.get("vol_down_pct", 0.5),
            "ret5d_pct":        tech.get("ret5d_pct", 0.0),
            "week52_pos":       tech.get("week52_pos", 0.0),
            "atr_pct":          tech.get("atr_pct", 0.0),
            "daily_range_pct":  tech.get("daily_range_pct", 0.0),
            "day_score":        tech.get("day_score", 0.0),
            # Sentiment detail
            "headline_count":    sent.get("headline_count", 0),
            "avg_compound":      sent.get("avg_compound", 0.0),
            "sample_headlines":  sent.get("headlines", []),
        })

    merged.sort(key=lambda x: x["combined_score"], reverse=True)

    # Deduplicate dual-class shares (e.g. GOOGL vs GOOG, BRK-A vs BRK-B).
    # Keep only the first (highest-scoring) entry per base company name.
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in merged:
        base = row["ticker"].split("-")[0].rstrip("AB")   # GOOG, BRK, etc.
        if base not in seen:
            seen.add(base)
            deduped.append(row)

    logger.info(f"Final ranking: {len(deduped)} stocks after dedup (from {len(merged)})")
    return deduped


def select_top_picks(
    ranked: list[dict],
    n: int = TOP_PICKS,
    max_per_sector: int | None = None,
    sector_fetcher=None,
) -> list[dict]:
    """
    Return top N picks with 1-based rank field added.

    If max_per_sector is set, enforces sector concentration cap (e.g. 2 per
    GICS sector) by lazily fetching each candidate's sector via the supplied
    sector_fetcher(ticker) callback.

    A 20-year-veteran insists on this: 5 picks all in semis is one factor
    bet, not five diversified picks.
    """
    # Copy each dict: the same candidate can land in several tables, and a
    # shared dict lets a later table's rank overwrite this one.
    if max_per_sector is None:
        picks = [{**r} for r in ranked[:n]]
        for i, pick in enumerate(picks, 1):
            pick["rank"] = i
        return picks

    picks: list[dict] = []
    sector_counts: dict[str, int] = {}

    for r in ranked:
        if len(picks) >= n:
            break

        sector = r.get("sector")
        if sector is None and sector_fetcher is not None:
            try:
                sector = sector_fetcher(r["ticker"]) or "Unknown"
            except Exception:
                sector = "Unknown"
            r["sector"] = sector

        if sector and sector_counts.get(sector, 0) >= max_per_sector:
            continue
        sector_counts[sector] = sector_counts.get(sector or "Unknown", 0) + 1
        picks.append({**r})

    for i, pick in enumerate(picks, 1):
        pick["rank"] = i
    logger.info(f"Sector cap applied: {len(picks)} picks across {len(sector_counts)} sectors")
    return picks


# ── Bearish picks (puts on stocks with active downtrends) ────────────────────

def get_top_bearish_candidates(
    technical_scores: dict[str, dict], n: int = 20
) -> list[str]:
    """
    Sort all tickers by BEARISH technical score and return the top N symbols.
    These get the news fetch + sentiment scoring for the bearish ranking.
    """
    ranked = sorted(
        technical_scores.items(),
        key=lambda x: x[1].get("bearish_score", 0.0),
        reverse=True,
    )
    top = [ticker for ticker, _ in ranked[:n]]
    logger.info(f"Bearish phase 1: → top {len(top)} bearish candidates")
    return top


def bearish_final_ranking(
    technical_scores: dict[str, dict],
    sentiment_scores: dict[str, dict],
) -> list[dict]:
    """
    Like final_ranking() but uses bearish_score and INVERTS sentiment so that
    very negative news (low sentiment_score) becomes a high bearish score.
    """
    merged: list[dict] = []

    for ticker, tech in technical_scores.items():
        sent = sentiment_scores.get(ticker, {"sentiment_score": 50.0, "headline_count": 0})

        bear_score = tech.get("bearish_score", 0.0)
        sent_score = sent.get("sentiment_score", 50.0)
        bear_sent  = 100.0 - sent_score   # invert: bad news → high bearish score
        combined   = bear_score * WEIGHT_TECHNICAL + bear_sent * WEIGHT_SENTIMENT

        merged.append({
            "ticker":             ticker,
            "combined_score":     round(combined, 2),
            "bearish_score":      round(bear_score, 2),
            "sentiment_score":    round(sent_score, 2),
            "bear_sentiment":     round(bear_sent, 2),
            # Technical detail for the report
            "last_price":         tech.get("last_price", 0.0),
            "rsi":                tech.get("rsi", 0.0),
            "macd_hist":          tech.get("macd_hist", 0.0),
            "trend_score":        tech.get("trend_score", 0.0),
            "vol_ratio":          tech.get("vol_ratio", 0.0),
            "vol_ratio_day":      tech.get("vol_ratio_day", 1.0),
            "vol_ratio_long":     tech.get("vol_ratio_long", 1.0),
            "vol_up_pct":         tech.get("vol_up_pct", 0.5),
            "vol_down_pct":       tech.get("vol_down_pct", 0.5),
            "ret5d_pct":          tech.get("ret5d_pct", 0.0),
            "atr_pct":            tech.get("atr_pct", 0.0),
            "daily_range_pct":    tech.get("daily_range_pct", 0.0),
            "day_score":          tech.get("day_score", 0.0),
            "week52_pos":         tech.get("week52_pos", 0.0),
            "sma20":              tech.get("sma20", 0.0),
            "sma50":              tech.get("sma50", 0.0),
            "sma200":             tech.get("sma200", 0.0),
            "headline_count":     sent.get("headline_count", 0),
            "avg_compound":       sent.get("avg_compound", 0.0),
            "sample_headlines":   sent.get("headlines", []),
        })

    merged.sort(key=lambda x: x["combined_score"], reverse=True)

    # Same dedup as bullish: collapse dual-class shares
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in merged:
        base = row["ticker"].split("-")[0].rstrip("AB")
        if base not in seen:
            seen.add(base)
            deduped.append(row)

    return deduped


def select_bearish_picks(
    ranked: list[dict], n: int = 3, min_score: float = 55.0
) -> list[dict]:
    """
    Top N bearish picks, but ONLY if combined_score >= min_score.
    This prevents the report from surfacing weak bearish setups in a strong
    bull market — a beginner shouldn't be buying speculative puts on
    sideways stocks.
    """
    qualifiers = [r for r in ranked if r["combined_score"] >= min_score]
    picks = qualifiers[:n]
    for i, pick in enumerate(picks, 1):
        pick["rank"] = i
    logger.info(f"Bearish picks: {len(picks)} qualified above score {min_score}")
    return picks


# ── Day-trade picks (separate lists ranked by day-suitability) ───────────────

def select_top_day_bullish(
    ranked: list[dict],
    n: int = 3,
    min_day_score: float = 60.0,
    min_combined: float = 50.0,
) -> list[dict]:
    """
    Top N BULLISH stocks ranked for INTRADAY trading.

    Composite = 60% × day_score + 30% × technical_score + 10% × sentiment_score.
    Filters out stocks with day_score < 60 (illiquid or low-volatility).
    Same stock can also appear in swing/long top 5 — that's fine, different
    timeframes value different things.
    """
    scored = []
    for r in ranked:
        day_sc  = r.get("day_score", 0.0)
        tech_sc = r.get("technical_score", 0.0)
        sent_sc = r.get("sentiment_score", 50.0)
        if day_sc < min_day_score:
            continue
        if r.get("combined_score", 0.0) < min_combined:
            continue
        composite = day_sc * 0.6 + tech_sc * 0.3 + sent_sc * 0.1
        scored.append((composite, r))
    scored.sort(key=lambda x: -x[0])
    picks = [{**r, "day_combined_score": round(s, 2)} for s, r in scored[:n]]
    for i, pick in enumerate(picks, 1):
        pick["rank"] = i
    logger.info(f"Bullish day-trade picks: {len(picks)} qualified above day_score {min_day_score}")
    return picks


def select_momentum_continuation_picks(
    ranked: list[dict],
    n: int = 3,
    min_rsi: float = 70.0,
    min_trend_score: float = 75.0,
) -> list[dict]:
    """
    Top N stocks in CONFIRMED STRONG UPTRENDS that are 'overbought' on RSI
    but still trending. The standard scoring penalizes RSI > 70; this list
    surfaces them deliberately because:

      An RSI 80 stock that's above all SMAs and making new 52-week highs is
      a CONTINUATION signal, not exhaustion. (Think NVDA / MU / AMD during
      the AI rally — they stayed overbought for months.)

    Filter:  rsi >= 70  AND  trend_score >= 75  (price above all 3 SMAs)
    Rank by: technical_score + 0.3 × ret5d_pct  (favouring active momentum)

    This list is PARALLEL to the swing/long top 5 — it does not replace it,
    just adds a second view for trend-following names.
    """
    qualifiers = []
    for r in ranked:
        rsi   = r.get("rsi", 50.0)
        trend = r.get("trend_score", 0.0)
        if rsi >= min_rsi and trend >= min_trend_score:
            momentum_score = (
                r.get("technical_score", 0.0) + 0.3 * r.get("ret5d_pct", 0.0)
            )
            qualifiers.append((momentum_score, r))

    qualifiers.sort(key=lambda x: -x[0])
    picks = [{**r, "momentum_score": round(m, 2)} for m, r in qualifiers[:n]]
    for i, pick in enumerate(picks, 1):
        pick["rank"] = i
    logger.info(
        f"Momentum continuation: {len(picks)} of {len(qualifiers)} qualifying "
        f"picks (rsi >= {min_rsi}, trend_score >= {min_trend_score})"
    )
    return picks


def _has_real_fundamentals(fund: dict) -> bool:
    """
    Long-term gate: a 1yr+ hold should be an actual business, not a
    momentum micro-cap. Require BOTH:

    1. Profitable today: profitMargins > 0. A positive forward P/E alone let
       names like NET (-8% margin, fwd P/E 208) through on analysts' hopes.
       A missing margin does not fail the gate; the P/E test still applies.
    2. Positive earnings: a positive trailing OR forward P/E.

    Unprofitable names (e.g. QTTB, AGEN, VEEE, NET, SAIL) are excluded from
    the long-term list; they still surface in the swing table.
    """
    if not fund:
        return False
    margin = fund.get("profitMargins")
    try:
        if margin is not None and float(margin) <= 0:
            return False
    except (TypeError, ValueError):
        pass
    for key in ("forwardPE", "trailingPE"):
        v = fund.get(key)
        try:
            if v is not None and float(v) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def select_long_term_picks(
    ranked: list[dict],
    n: int = 5,
    max_per_sector: int | None = 2,
    sector_fetcher=None,
    fundamentals_fetcher=None,
) -> list[dict]:
    """
    Top N stocks suited for long-term holding (1 year+).

    Composite = 50% technical_score + 35% trend_score + 15% accumulation
    where accumulation = min(vol_ratio_long × 20, 20).

    trend_score ≥ 25 is the gate — means price is at least above the 200 SMA.
    Stocks below all SMAs are not suitable long-term buys.

    When ``fundamentals_fetcher`` is supplied, a second gate requires positive
    earnings (see ``_has_real_fundamentals``) so unprofitable momentum micro-caps
    do not masquerade as long-term holds.

    These picks are intentionally distinct from swing/long top 5:
    - They weight trend confirmation (200 SMA) more heavily
    - They reward sustained volume accumulation (20d vs 60d)
    - RSI is not penalised here — great long-term stocks can stay overbought for months
    """
    scored = []
    for r in ranked:
        tech  = r.get("technical_score", 0.0)
        trend = r.get("trend_score", 0.0)
        vol_l = r.get("vol_ratio_long", 1.0)
        if trend < 25:          # must be above at least the 200 SMA
            continue
        composite = tech * 0.50 + trend * 0.35 + min(vol_l * 20, 20) * 0.15
        scored.append((composite, r))

    scored.sort(key=lambda x: -x[0])
    candidates = [r for _, r in scored]

    def _passes_fundamentals(r: dict) -> bool:
        """Lazy positive-earnings gate; passes through when no fetcher given."""
        if fundamentals_fetcher is None:
            return True
        try:
            return _has_real_fundamentals(fundamentals_fetcher(r["ticker"]) or {})
        except Exception:
            return False

    if max_per_sector is None:
        picks = []
        for r in candidates:
            if len(picks) >= n:
                break
            if not _passes_fundamentals(r):
                continue
            picks.append({**r})  # copy: see select_top_picks
        for i, pick in enumerate(picks, 1):
            pick["rank"] = i
        return picks

    picks: list[dict] = []
    sector_counts: dict[str, int] = {}

    for r in candidates:
        if len(picks) >= n:
            break
        if not _passes_fundamentals(r):
            continue
        sector = r.get("sector")
        if sector is None and sector_fetcher is not None:
            try:
                sector = sector_fetcher(r["ticker"]) or "Unknown"
            except Exception:
                sector = "Unknown"
            r["sector"] = sector
        if sector and sector_counts.get(sector, 0) >= max_per_sector:
            continue
        sector_counts[sector] = sector_counts.get(sector or "Unknown", 0) + 1
        picks.append({**r})  # copy: see select_top_picks

    for i, pick in enumerate(picks, 1):
        pick["rank"] = i
    logger.info(f"Long-term picks: {len(picks)} picks across {len(sector_counts)} sectors")
    return picks


def select_top_day_bearish(
    bear_ranked: list[dict],
    n: int = 3,
    min_day_score: float = 60.0,
    min_combined: float = 55.0,
) -> list[dict]:
    """
    Top N BEARISH stocks ranked for INTRADAY trading.

    Composite = 60% × day_score + 30% × bearish_score + 10% × bear_sentiment.
    """
    scored = []
    for r in bear_ranked:
        day_sc    = r.get("day_score", 0.0)
        bear_sc   = r.get("bearish_score", 0.0)
        bear_sent = r.get("bear_sentiment", 50.0)
        if day_sc < min_day_score:
            continue
        if r.get("combined_score", 0.0) < min_combined:
            continue
        composite = day_sc * 0.6 + bear_sc * 0.3 + bear_sent * 0.1
        scored.append((composite, r))
    scored.sort(key=lambda x: -x[0])
    picks = [{**r, "day_combined_score": round(s, 2)} for s, r in scored[:n]]
    for i, pick in enumerate(picks, 1):
        pick["rank"] = i
    logger.info(f"Bearish day-trade picks: {len(picks)} qualified above day_score {min_day_score}")
    return picks
