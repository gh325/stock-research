"""
config.py — Central configuration for the stock research pipeline.
Edit this file to tune scoring weights, filters, and schedule.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH    = DATA_DIR / "paper_trades.db"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Universe filters ──────────────────────────────────────────────────────────
# Stocks below these thresholds are dropped before any analysis.
MIN_PRICE      = 5.0        # USD — filters penny stocks
MIN_AVG_VOLUME = 500_000    # shares/day — filters illiquid names
# Calendar days of OHLCV history to download. 400 calendar days ≈ 275
# trading days: enough for a real 200-day SMA and a true 52-week range.
# (At 90 days, technicals silently fell back to the 50 SMA for "200 SMA".)
LOOKBACK_DAYS  = 400

# ── Pipeline sizing ───────────────────────────────────────────────────────────
TOP_CANDIDATES     = 50   # stocks sent to sentiment phase (top-N by technicals)
TOP_PICKS          = 5    # final recommendations
DOWNLOAD_BATCH_SIZE = 100  # tickers per yfinance batch request

# ── Scoring weights (must each sum to 1.0) ────────────────────────────────────
# Final score = technical * WEIGHT_TECHNICAL + sentiment * WEIGHT_SENTIMENT
WEIGHT_TECHNICAL = 0.60
WEIGHT_SENTIMENT = 0.40

# Sub-weights within the technical score. They live in params/scoring.json so
# `python main.py evolve` can propose changes without touching code. These
# defaults apply only if that file is missing.
PARAMS_PATH = BASE_DIR / "params" / "scoring.json"
DEFAULT_TECH_WEIGHTS = {
    "rsi":      0.20,   # Relative Strength Index — momentum
    "macd":     0.25,   # MACD — trend-following momentum
    "trend":    0.25,   # Price vs moving averages — trend direction
    "volume":   0.15,   # Volume vs 20-day avg — institutional interest
    "momentum": 0.15,   # 52-week position + recent return
}


def load_tech_weights(path: Path = PARAMS_PATH) -> dict[str, float]:
    """Technical sub-weights from params/scoring.json, validated to sum to 1."""
    import json
    if not path.exists():
        return dict(DEFAULT_TECH_WEIGHTS)
    weights = json.loads(path.read_text())["tech_weights"]
    if set(weights) != set(DEFAULT_TECH_WEIGHTS) or abs(sum(weights.values()) - 1) > 1e-6:
        raise ValueError(f"{path}: tech_weights must have keys "
                         f"{sorted(DEFAULT_TECH_WEIGHTS)} and sum to 1.0")
    return {k: float(v) for k, v in weights.items()}


_TECH_WEIGHTS        = load_tech_weights()
TECH_WEIGHT_RSI      = _TECH_WEIGHTS["rsi"]
TECH_WEIGHT_MACD     = _TECH_WEIGHTS["macd"]
TECH_WEIGHT_TREND    = _TECH_WEIGHTS["trend"]
TECH_WEIGHT_VOLUME   = _TECH_WEIGHTS["volume"]
TECH_WEIGHT_MOMENTUM = _TECH_WEIGHTS["momentum"]

# ── Scheduler ────────────────────────────────────────────────────────────────
SCHEDULE_HOUR   = 21   # 9:00 PM local time (after US market close)
SCHEDULE_MINUTE = 0

# ── Optional API keys ─────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── News RSS feeds (free, no key required) ────────────────────────────────────
NEWS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
]

# ── Cache TTLs ────────────────────────────────────────────────────────────────
TICKER_CACHE_TTL = 7 * 86_400    # 7 days — ticker list changes slowly
PRICE_CACHE_TTL  = 14 * 3_600    # 14 hours — re-fetch after market close
