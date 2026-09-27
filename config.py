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

# Sub-weights within the technical score
TECH_WEIGHT_RSI      = 0.20   # Relative Strength Index — momentum
TECH_WEIGHT_MACD     = 0.25   # MACD — trend-following momentum
TECH_WEIGHT_TREND    = 0.25   # Price vs moving averages — trend direction
TECH_WEIGHT_VOLUME   = 0.15   # Volume vs 20-day avg — institutional interest
TECH_WEIGHT_MOMENTUM = 0.15   # 52-week position + recent return

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
