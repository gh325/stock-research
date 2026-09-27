"""
ingest.py — Data ingestion layer.

Responsibilities:
  1. Fetch the full list of US-listed common stocks (NASDAQ FTP → Wikipedia fallback)
  2. Batch-download 90-day OHLCV price history via yfinance
  3. Fetch news headlines per ticker via yfinance
"""
import functools
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from config import (
    DATA_DIR,
    DOWNLOAD_BATCH_SIZE,
    LOOKBACK_DAYS,
    MIN_AVG_VOLUME,
    MIN_PRICE,
    PRICE_CACHE_TTL,
    TICKER_CACHE_TTL,
)

logger = logging.getLogger(__name__)

TICKER_CACHE = DATA_DIR / "tickers.json"
# Lookback in the name: a cache written with a shorter history is never reused
PRICE_CACHE  = DATA_DIR / f"prices_{LOOKBACK_DAYS}d.pkl"
WATCHLIST_DIR = Path(__file__).parent / "watchlists"


def load_watchlist(name: str) -> list[str]:
    """
    Load a watchlist from `watchlists/<name>.txt`.
    Format: one ticker per line, '#' for comments, blank lines ignored.
    Returns the tickers (uppercased) in file order, deduped.
    """
    path = WATCHLIST_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")
    tickers: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        t = line.split("#", 1)[0].strip().upper()   # allow inline comments
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)
    logger.info(f"Watchlist '{name}': loaded {len(tickers)} tickers")
    return tickers


def load_watchlist_header(name: str, key: str) -> str | None:
    """
    Read a `# key: value` header line from watchlists/<name>.txt.
    Headers used today:
      # benchmark: CIBR, HACK, BUG   theme ETFs (watchlist_check, ETF panel)
      # sector_cap: off              single-sector list — skip the 2-per-sector cap
    """
    path = WATCHLIST_DIR / f"{name}.txt"
    prefix = f"# {key.lower()}:"
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


# ── Ticker Universe ───────────────────────────────────────────────────────────

def get_ticker_universe() -> list[str]:
    """
    Return all US common stock tickers (NASDAQ + NYSE + NYSE Arca).
    The list is cached locally for 7 days so we don't hammer the source.
    """
    if _cache_fresh(TICKER_CACHE, TICKER_CACHE_TTL):
        logger.info("Loading ticker list from cache")
        return json.loads(TICKER_CACHE.read_text())

    tickers = _fetch_nasdaq_tickers()
    if not tickers:
        logger.warning("NASDAQ FTP failed — falling back to S&P 500 from Wikipedia")
        tickers = _get_sp500_tickers()

    if not tickers:
        return _get_hardcoded_sp100()   # never cache the hardcoded list

    TICKER_CACHE.write_text(json.dumps(tickers))
    logger.info(f"Ticker universe: {len(tickers)} symbols saved to cache")
    return tickers


def _parse_nasdaq_traded(text: str) -> list[str]:
    """Parse nasdaqtraded.txt content into a filtered common-stock ticker list."""
    from io import StringIO
    df = pd.read_csv(StringIO(text), sep="|")
    mask = (
        (df["Test Issue"] == "N") &
        (df["ETF"] == "N") &
        df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)
    )
    return df.loc[mask, "Symbol"].dropna().tolist()


def _fetch_nasdaq_tickers() -> list[str]:
    """
    Fetch all US common stocks from the NASDAQ symbol directory.
    Tries the HTTPS mirror first, then falls back to plain FTP (the HTTPS
    endpoint is blocked or rate-limited on some networks while FTP stays up).
    Filters out ETFs, warrants, and test issues.
    """
    https_url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    try:
        resp = requests.get(https_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        tickers = _parse_nasdaq_traded(resp.text)
        logger.info(f"NASDAQ FTP (HTTPS): fetched {len(tickers)} common stocks")
        return tickers
    except Exception as exc:
        logger.warning(f"NASDAQ HTTPS mirror failed ({exc}); trying plain FTP")

    ftp_url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt"
    try:
        import urllib.request
        with urllib.request.urlopen(ftp_url, timeout=30) as fh:
            text = fh.read().decode("utf-8", errors="replace")
        tickers = _parse_nasdaq_traded(text)
        logger.info(f"NASDAQ FTP (plain): fetched {len(tickers)} common stocks")
        return tickers
    except Exception as exc:
        logger.error(f"NASDAQ plain-FTP fetch failed: {exc}")
        return []


def _get_sp500_tickers() -> list[str]:
    """
    Scrape S&P 500 constituents from Wikipedia (fallback).
    Returns ~503 large-cap US stocks.
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        from io import StringIO
        tables = pd.read_html(StringIO(resp.text))
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Wikipedia S&P 500: fetched {len(tickers)} tickers")
        return tickers
    except Exception as exc:
        logger.error(f"Wikipedia S&P 500 fetch failed: {exc}")
        return _get_hardcoded_sp100()


def _get_hardcoded_sp100() -> list[str]:
    """Last-resort fallback: S&P 100 hardcoded. Used if all network sources fail."""
    logger.warning("All network sources failed — using hardcoded S&P 100 list")
    return [
        "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","TSLA","BRK-B","UNH",
        "LLY","JPM","V","XOM","AVGO","MA","JNJ","PG","COST","HD","MRK","ABBV",
        "CVX","CRM","BAC","AMD","KO","PEP","ADBE","WMT","TMO","ACN","MCD","CSCO",
        "ABT","NFLX","LIN","DHR","TXN","NEE","ORCL","QCOM","PM","UPS","INTU",
        "AMGN","RTX","SPGI","CAT","GS","HON","IBM","LOW","ELV","MDT","DE","MS",
        "BLK","ISRG","SBUX","GE","AXP","BKNG","ADI","REGN","VRTX","ZTS","CI",
        "MO","TJX","CB","SO","DUK","MDLZ","SCHW","C","USB","PLD","MMC","CME",
        "EOG","PGR","BSX","BDX","NOC","GD","ITW","ETN","SHW","APD","ICE","AON",
        "F","GM","KHC","MET","PNC","TGT","COF","AIG","PYPL","INTC",
    ]


# ── Price Data ────────────────────────────────────────────────────────────────

def fetch_price_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for all tickers, batched to respect yfinance rate limits.
    Returns { "AAPL": DataFrame(Open/High/Low/Close/Volume), ... }

    First run: 15–25 minutes for ~4 000 tickers.
    Subsequent runs within 14 hours: instant (uses cache).
    """
    if _cache_fresh(PRICE_CACHE, PRICE_CACHE_TTL):
        logger.info("Loading price data from cache")
        return pd.read_pickle(PRICE_CACHE)

    logger.info(f"Downloading price data for {len(tickers)} tickers …")
    result: dict[str, pd.DataFrame] = {}
    batches = [
        tickers[i : i + DOWNLOAD_BATCH_SIZE]
        for i in range(0, len(tickers), DOWNLOAD_BATCH_SIZE)
    ]

    for idx, batch in enumerate(batches, 1):
        logger.info(f"  Batch {idx}/{len(batches)} ({len(batch)} tickers)")
        try:
            raw = yf.download(
                " ".join(batch),
                period=f"{LOOKBACK_DAYS}d",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            for ticker in batch:
                df = _extract_ticker_df(raw, ticker, len(batch))
                if df is not None and len(df) >= 20:
                    result[ticker] = df
        except Exception as exc:
            logger.warning(f"Batch {idx} download error: {exc}")

        time.sleep(0.3)   # gentle rate limiting between batches

    result = _filter_quality(result)
    pd.to_pickle(result, PRICE_CACHE)
    logger.info(f"Price cache saved: {len(result)} quality stocks")
    return result


def _extract_ticker_df(
    raw: pd.DataFrame, ticker: str, batch_size: int
) -> pd.DataFrame | None:
    """Extract a single ticker's DataFrame from a yfinance multi-ticker download."""
    try:
        if batch_size == 1:
            df = raw.copy()
        else:
            df = raw.xs(ticker, axis=1, level=0).copy()
        return df.dropna(how="all")
    except (KeyError, TypeError):
        return None


def _filter_quality(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Drop penny stocks (< $5) and illiquid names (< 500k avg daily volume).
    These filters prevent low-quality signals from polluting the rankings.
    """
    filtered = {}
    for ticker, df in data.items():
        try:
            last_close = float(df["Close"].iloc[-1])
            avg_vol    = float(df["Volume"].tail(20).mean())
            if last_close >= MIN_PRICE and avg_vol >= MIN_AVG_VOLUME:
                filtered[ticker] = df
        except Exception:
            continue
    removed = len(data) - len(filtered)
    logger.info(f"Quality filter: kept {len(filtered)}, removed {removed} stocks")
    return filtered


# ── News ──────────────────────────────────────────────────────────────────────

def fetch_news_for_tickers(tickers: list[str]) -> dict[str, list[str]]:
    """
    Fetch recent news headlines for each ticker using yfinance.
    Called only for the top ~50 technical candidates to save time.
    Returns { "AAPL": ["Headline 1", "Headline 2", ...], ... }
    """
    news: dict[str, list[str]] = {}
    for ticker in tickers:
        try:
            items = yf.Ticker(ticker).news or []
            headlines = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                # yfinance 1.x: title lives inside item['content']['title']
                content = item.get("content", {})
                title = (
                    item.get("title")
                    or (content.get("title", "") if isinstance(content, dict) else "")
                )
                if title:
                    headlines.append(str(title))
            news[ticker] = headlines
        except Exception:
            news[ticker] = []
        time.sleep(0.1)
    return news


# ── Fundamentals ─────────────────────────────────────────────────────────────

# Keys we care about from yfinance's .info dict
_FUNDAMENTAL_KEYS = [
    "sector", "industry", "fullName", "longName",
    "trailingPE", "forwardPE", "priceToBook",
    "revenueGrowth", "earningsGrowth", "profitMargins", "grossMargins",
    "dividendYield", "marketCap", "beta",
    "targetMeanPrice", "targetHighPrice", "targetLowPrice",
    "numberOfAnalystOpinions", "recommendationKey",
    "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
]


@functools.lru_cache(maxsize=512)
def fetch_fundamentals(ticker: str) -> dict:
    """
    Pull key fundamental data for a single ticker via yfinance .info.
    Called only for the top 5 picks — .info is a slow per-ticker call.

    Cached per-process to avoid re-fetching for the same ticker (sector lookup,
    bullish enrichment, bearish enrichment can all hit the same ticker). Each
    miss creates an HTTP connection pool, so caching prevents file-descriptor
    exhaustion on long runs.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return {k: info.get(k) for k in _FUNDAMENTAL_KEYS}
    except Exception as exc:
        logger.warning(f"{ticker} fundamentals fetch error: {exc}")
        return {}


@functools.lru_cache(maxsize=512)
def fetch_earnings_beat_miss(ticker: str) -> dict:
    """
    Fetch the most recent quarterly EPS beat or miss via yfinance.

    Returns {"beat": bool, "beat_pct": float, "eps_actual": float, "eps_estimate": float}
    or {} if data unavailable or estimates not on record.

    A large EPS beat often drives institutional re-rating that keeps the stock
    elevated for weeks — exactly the catalyst the screener's VADER layer misses.
    """
    try:
        yft = yf.Ticker(ticker)
        df  = None
        for attr in ("earnings_history", "get_earnings_history"):
            try:
                candidate = getattr(yft, attr)
                if callable(candidate):
                    candidate = candidate()
                if candidate is not None and hasattr(candidate, "empty") and not candidate.empty:
                    df = candidate
                    break
            except Exception:
                continue

        if df is None or df.empty:
            return {}

        row = df.iloc[0]

        def _get(r, *keys):
            for k in keys:
                try:
                    v = r[k]
                    if v is not None and str(v).lower() not in ("nan", "none", ""):
                        return float(v)
                except (KeyError, TypeError, ValueError):
                    continue
            return None

        actual   = _get(row, "epsActual", "actual", "Actual")
        estimate = _get(row, "epsEstimate", "estimate", "Estimate")

        if actual is None or estimate is None:
            return {}

        # Beat/miss % is only meaningful off a real positive earnings expectation.
        # Pre-profit names (estimate <= 0) or near-zero bases (e.g. $0.01) make the
        # percentage explode into noise (+100%, -160%) — suppress those entirely.
        # For those, the move is revenue/guidance-driven, which the pipeline already
        # captures via revenue growth.
        if estimate < 0.10:
            return {}

        beat_pct = (actual - estimate) / estimate * 100
        return {
            "beat":         beat_pct > 0,
            "beat_pct":     round(beat_pct, 1),
            "eps_actual":   round(actual, 2),
            "eps_estimate": round(estimate, 2),
        }
    except Exception as exc:
        logger.debug(f"{ticker} earnings beat/miss fetch: {exc}")
        return {}


@functools.lru_cache(maxsize=512)
def fetch_next_earnings_date(ticker: str):
    """
    Get the next earnings announcement date for a ticker, or None if unavailable.

    yfinance has changed this API across versions; we try `.calendar` (returns
    a dict in 0.2.50+). Returns a `datetime` or None.
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if not cal:
            return None
        # yfinance 0.2.50+ returns a dict
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date") or cal.get("earnings_date")
            if isinstance(ed, list) and ed:
                val = ed[0]
                if hasattr(val, "year"):
                    return val
                try:
                    return pd.to_datetime(val).to_pydatetime()
                except Exception:
                    return None
        return None
    except Exception as exc:
        logger.debug(f"{ticker} earnings date fetch error: {exc}")
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_fresh(path: Path, max_age: float) -> bool:
    """Return True if the file exists and is younger than max_age seconds."""
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < max_age
