"""
sentiment.py — News sentiment scoring using VADER.

VADER (Valence Aware Dictionary for Sentiment Reasoning) was built specifically
for short social/financial text. It works offline — no API key needed.

Scores:
  compound  -1.0 → +1.0  (negative → positive)
  normalised 0   → 100   (what we store and rank on)

For the overall market, we also parse free RSS feeds to get a broad mood reading.
"""
import logging

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NEWS_FEEDS

logger   = logging.getLogger(__name__)
_vader   = SentimentIntensityAnalyzer()


def score_sentiment_for_tickers(
    ticker_news: dict[str, list[str]]
) -> dict[str, dict]:
    """
    Score sentiment for each ticker.

    Input:  { "AAPL": ["Apple beats earnings", "iPhone demand surges", ...] }
    Output: { "AAPL": { "sentiment_score": 72.5, "avg_compound": 0.45, ... } }
    """
    return {ticker: _score_headlines(headlines) for ticker, headlines in ticker_news.items()}


def fetch_market_rss_sentiment() -> dict:
    """
    Parse general financial RSS feeds and return a market-wide sentiment reading.
    Used as context in the final report ("market is broadly Bullish today").
    """
    headlines: list[str] = []
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                if title:
                    headlines.append(title)
        except Exception as exc:
            logger.warning(f"RSS parse error ({url}): {exc}")

    if not headlines:
        return {"market_sentiment": 50.0, "article_count": 0, "market_mood": "Neutral"}

    result = _score_headlines(headlines)
    return {
        "market_sentiment": result["sentiment_score"],
        "article_count":    result["headline_count"],
        "market_mood":      _mood_label(result["avg_compound"]),
    }


def _score_headlines(headlines: list[str]) -> dict:
    """Run VADER on each headline and return an averaged, normalised score."""
    if not headlines:
        return {"sentiment_score": 50.0, "headline_count": 0, "avg_compound": 0.0, "headlines": []}

    compounds = [
        _vader.polarity_scores(h)["compound"]
        for h in headlines
        if h.strip()
    ]

    if not compounds:
        return {"sentiment_score": 50.0, "headline_count": 0, "avg_compound": 0.0, "headlines": []}

    avg = sum(compounds) / len(compounds)
    # Map VADER compound (-1 to +1) → 0 to 100
    score = (avg + 1.0) / 2.0 * 100.0

    return {
        "sentiment_score": round(score, 2),
        "headline_count":  len(compounds),
        "avg_compound":    round(avg, 4),
        "headlines":       headlines[:5],   # sample stored for the report
    }


def _mood_label(compound: float) -> str:
    if compound >= 0.30:    return "Bullish"
    if compound >= 0.05:    return "Slightly Bullish"
    if compound <= -0.30:   return "Bearish"
    if compound <= -0.05:   return "Slightly Bearish"
    return "Neutral"
