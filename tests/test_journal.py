"""Research journal: recording, grading and scoring predictions. No network."""
import numpy as np
import pandas as pd
import pytest

import journal


@pytest.fixture
def db(tmp_path):
    return tmp_path / "journal.db"


def _series(values, start="2026-01-02") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def _pred(etf="CIBR", bucket="Trending", made_on="2026-01-02", source="live") -> dict:
    return {"made_on": made_on, "etf": etf, "theme": "Cybersecurity", "bucket": bucket,
            "expect": journal.EXPECT[bucket], "horizon": 5, "source": source}


# ── Predictions from a scan ──────────────────────────────────────────────────

def test_predictions_cover_every_bucketed_etf_and_skip_the_middle() -> None:
    scan_all = pd.DataFrame([
        {"etf": "CIBR", "theme": "Cyber", "bucket": "Trending"},
        {"etf": "EWJ",  "theme": "Japan", "bucket": "Emerging"},
        {"etf": "TAN",  "theme": "Solar", "bucket": "Breaking"},
        {"etf": "XLP",  "theme": "Staples", "bucket": None},
    ])
    preds = journal.predictions_from_scan(scan_all, made_on="2026-09-27", horizon=20)
    assert {(p["etf"], p["expect"]) for p in preds} == {
        ("CIBR", "outperform"), ("EWJ", "outperform"), ("TAN", "underperform")}


# ── Recording ────────────────────────────────────────────────────────────────

def test_record_is_idempotent(db) -> None:
    assert journal.record([_pred()], db) == 1
    assert journal.record([_pred()], db) == 0      # same day, ETF, horizon, source
    assert journal.record([_pred(source="backfill")], db) == 1


def test_record_live_skips_etfs_with_an_open_window(db) -> None:
    scan = {"all": pd.DataFrame([{"etf": "CIBR", "theme": "Cyber", "bucket": "Trending"},
                                 {"etf": "TAN",  "theme": "Solar", "bucket": "Breaking"}])}
    assert journal.record_live(scan, db, made_on="2026-09-01") == 2
    assert journal.record_live(scan, db, made_on="2026-09-02") == 0   # windows still open
    assert journal.record_live(scan, db, made_on="2026-10-15") == 2   # ~30 trading days later


# ── Grading ──────────────────────────────────────────────────────────────────

def test_grade_one_uses_entry_on_or_after_made_on_and_horizon_rows_later() -> None:
    etf = _series([100, 101, 102, 103, 104, 105, 110])
    spy = _series([100] * 7)
    etf_ret, spy_ret = journal.grade_one(etf, spy, "2026-01-02", horizon=5)
    assert etf_ret == pytest.approx(5.0) and spy_ret == pytest.approx(0.0)


def test_grade_one_returns_none_before_the_horizon_passes() -> None:
    assert journal.grade_one(_series([100, 101]), _series([100, 100]), "2026-01-02", 5) is None


def test_grade_marks_correct_and_wrong_calls(db) -> None:
    journal.record([_pred("UP", "Trending"), _pred("DOWN", "Breaking"),
                    _pred("MISS", "Trending")], db)
    prices = {
        "UP":   _series([100, 101, 102, 103, 104, 106]),   # +6% vs SPY flat → correct
        "DOWN": _series([100, 99, 98, 97, 96, 94]),        # -6%, predicted to lag → correct
        "MISS": _series([100, 99, 98, 97, 96, 95]),        # -5%, predicted to lead → wrong
        "SPY":  _series([100] * 6),
    }
    assert journal.grade(db, fetch=lambda tickers, start: prices) == 3
    stats = journal.summary(db)
    trending = stats[(stats.bucket == "Trending") & (stats.source == "live")].iloc[0]
    breaking = stats[(stats.bucket == "Breaking") & (stats.source == "live")].iloc[0]
    assert trending.n == 2 and trending.correct == 1
    assert breaking.n == 1 and breaking.correct == 1
    assert journal.grade(db, fetch=lambda tickers, start: prices) == 0   # already graded


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_wilson_interval_brackets_the_rate_and_narrows_with_n() -> None:
    lo_small, hi_small = journal.wilson(6, 10)
    lo_big, hi_big     = journal.wilson(600, 1000)
    assert lo_small < 0.6 < hi_small and lo_big < 0.6 < hi_big
    assert (hi_big - lo_big) < (hi_small - lo_small)


# ── Backfill (pure part) ─────────────────────────────────────────────────────

def test_backfill_uses_only_data_up_to_each_date() -> None:
    n = 320
    rising = _series(np.linspace(100, 200, n), start="2025-01-01")
    flat   = _series(np.full(n, 100.0), start="2025-01-01")
    vol    = _series(np.full(n, 1e6), start="2025-01-01")
    dates  = [rising.index[260], rising.index[280]]
    preds  = journal.backfill_predictions({"UPX": (rising, vol)}, spy=flat,
                                          dates=dates, horizon=20)
    assert [p["made_on"] for p in preds] == [d.strftime("%Y-%m-%d") for d in dates]
    assert all(p["bucket"] == "Trending" and p["source"] == "backfill" for p in preds)
