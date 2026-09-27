"""Sector scan and watchlist check logic. Pure functions only — no network."""
from datetime import datetime

import numpy as np
import pandas as pd

import ingest
import sector_scan
import watchlist_check as wc
from conftest import make_ohlcv


# ── Watchlist files ──────────────────────────────────────────────────────────

def test_every_watchlist_loads_and_is_deduped() -> None:
    for path in ingest.WATCHLIST_DIR.glob("*.txt"):
        tickers = ingest.load_watchlist(path.stem)
        assert tickers, f"{path.name} is empty"
        assert len(tickers) == len(set(tickers)), f"{path.name} has duplicates"
        assert all(t == t.upper() and " " not in t for t in tickers)


def test_benchmark_header() -> None:
    assert wc.load_benchmarks("cyber") == ["CIBR", "HACK", "BUG"]
    assert wc.load_benchmarks("custom") == []


def test_sector_cap_header() -> None:
    assert ingest.load_watchlist_header("energy", "sector_cap") == "off"
    assert ingest.load_watchlist_header("ai_ecosystem", "sector_cap") is None


# ── Watchlist check: per-ticker status ───────────────────────────────────────

def test_no_data_is_remove() -> None:
    status, reasons = wc.assess_ticker({"has_data": False})
    assert status == "REMOVE" and "no price data" in reasons[0]


def test_illiquid_is_remove(uptrend) -> None:
    m = wc.ticker_metrics(make_ohlcv(uptrend["Close"].values, volume=1_000), bench_ret3m=0)
    assert wc.assess_ticker(m)[0] == "REMOVE"


def test_broken_and_lagging_is_bench(downtrend) -> None:
    m = wc.ticker_metrics(downtrend, bench_ret3m=10)
    status, reasons = wc.assess_ticker(m)
    assert status == "BENCH" and "lagging theme" in reasons[0]


def test_earnings_and_negative_news_are_watch(uptrend) -> None:
    m = wc.ticker_metrics(uptrend, bench_ret3m=0)
    status, reasons = wc.assess_ticker(
        m, earnings=datetime(2026, 9, 30),
        sentiment={"headline_count": 5, "sentiment_score": 20},
        today=datetime(2026, 9, 26),
    )
    assert status == "WATCH"
    assert any("earnings 09/30" in r for r in reasons)
    assert any("negative news" in r for r in reasons)


def test_thin_news_is_ignored() -> None:
    # Gentle oscillation: RSI near 50, no trend, nothing else to flag. (A
    # perfectly flat series has no down days, so RSI reads 100 = overbought.)
    wave = 100 + np.sin(np.arange(275) / 3)
    m = wc.ticker_metrics(make_ohlcv(wave), bench_ret3m=0)
    status, _ = wc.assess_ticker(m, sentiment={"headline_count": 1, "sentiment_score": 5})
    assert status == "KEEP"


def test_theme_classification(uptrend, downtrend) -> None:
    assert wc.classify_theme(uptrend["Close"], downtrend["Close"])["status"] == "Leading"
    assert wc.classify_theme(downtrend["Close"], uptrend["Close"])["status"] == "Broken"


# ── Sector scan buckets ──────────────────────────────────────────────────────

def _metrics(**over: float) -> dict:
    base = {"rs1m": 0.0, "rs3m": 0.0, "rs6m": 0.0, "ret1m": 0.0, "vs200": 0.0, "accel": 0.0}
    return {**base, **over}


def test_bucket_trending() -> None:
    assert sector_scan.classify(_metrics(rs3m=5, rs6m=10, vs200=8)) == "Trending"


def test_bucket_emerging_requires_rising_price() -> None:
    early = _metrics(rs1m=3, accel=2, ret1m=2, rs6m=-5)
    assert sector_scan.classify(early) == "Emerging"
    # Heavy volume on a falling price is distribution, not an early turn.
    assert sector_scan.classify({**early, "ret1m": -1}) is None


def test_bucket_breaking() -> None:
    assert sector_scan.classify(_metrics(rs3m=-5, rs6m=-10, vs200=-8)) == "Breaking"


def test_etf_metrics_needs_200_days() -> None:
    short = make_ohlcv(np.linspace(100, 110, 150))
    assert sector_scan.etf_metrics(short["Close"], short["Volume"], short["Close"]) is None


def test_etf_metrics_relative_strength(uptrend) -> None:
    flat = pd.Series(100.0, index=uptrend.index)
    m = sector_scan.etf_metrics(uptrend["Close"], uptrend["Volume"], flat)
    assert m is not None and m["rs3m"] > 0 and m["vs200"] > 0
    assert sector_scan.classify(m) == "Trending"
