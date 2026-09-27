"""
journal.py — Research journal: does the sector scan actually predict anything?

Every sector scan makes implicit calls: Trending and Emerging themes should
beat SPY, and Breaking themes should lag it. The journal writes each call down
as a testable prediction, grades it once the horizon has passed, and reports
hit rates with confidence intervals. That is how the research improves: we
keep what has worked and discount what hasn't, based on evidence rather than
on stories.

  python main.py journal backfill   # grade ~2 years of non-overlapping scans now
  python main.py journal grade      # grade live predictions whose horizon has passed
  python main.py journal report     # hit rate per bucket, with 95% intervals

Predictions live in the tracker DB (config.DB_PATH, git-ignored). Headline
sentiment is not part of a prediction, because it has no history to backfill.
"""
import logging
import sqlite3
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from config import DB_PATH

logger  = logging.getLogger(__name__)
console = Console(width=160)

HORIZON = 20                                  # trading days per prediction
EXPECT  = {"Trending": "outperform", "Emerging": "outperform", "Breaking": "underperform"}
MIN_N   = 30                                  # below this, a hit rate is anecdote

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    made_on   TEXT NOT NULL,
    etf       TEXT NOT NULL,
    theme     TEXT,
    bucket    TEXT NOT NULL,
    expect    TEXT NOT NULL,
    horizon   INTEGER NOT NULL,
    source    TEXT NOT NULL,
    graded_on TEXT,
    etf_ret   REAL,
    spy_ret   REAL,
    correct   INTEGER,
    UNIQUE (made_on, etf, horizon, source)
)
"""


def _conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


# ── Making predictions (pure) ────────────────────────────────────────────────

def predictions_from_scan(scan_all: pd.DataFrame, made_on: str, horizon: int = HORIZON,
                          source: str = "live") -> list[dict]:
    """One prediction per ETF the scan put in a bucket. The unbucketed middle makes no call."""
    return [
        {"made_on": made_on, "etf": r.etf, "theme": r.theme, "bucket": r.bucket,
         "expect": EXPECT[r.bucket], "horizon": horizon, "source": source}
        for r in scan_all.itertuples() if r.bucket in EXPECT
    ]


def backfill_predictions(etfs: dict[str, tuple[pd.Series, pd.Series]], spy: pd.Series,
                         dates: list[pd.Timestamp], horizon: int = HORIZON) -> list[dict]:
    """
    Re-run the scan's bucketing as of each past date, using only prices up to
    that date. `etfs` maps ticker -> (close, volume).
    """
    from sector_scan import THEME_ETFS, classify, etf_metrics

    preds = []
    for d in dates:
        spy_d = spy[spy.index <= d]
        for etf, (close, volume) in etfs.items():
            m = etf_metrics(close[close.index <= d], volume[volume.index <= d], spy_d)
            bucket = classify(m) if m else None
            if bucket in EXPECT:
                preds.append({"made_on": d.strftime("%Y-%m-%d"), "etf": etf,
                              "theme": THEME_ETFS.get(etf, ""), "bucket": bucket,
                              "expect": EXPECT[bucket], "horizon": horizon, "source": "backfill"})
    return preds


def record(preds: list[dict], db_path: Path = DB_PATH) -> int:
    """Insert predictions and skip duplicates. Returns how many were new."""
    with _conn(db_path) as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO predictions (made_on, etf, theme, bucket, expect, horizon, source) "
            "VALUES (:made_on, :etf, :theme, :bucket, :expect, :horizon, :source)", preds)
        return conn.total_changes - before


# ── Grading ──────────────────────────────────────────────────────────────────

def grade_one(etf_close: pd.Series, spy_close: pd.Series, made_on: str,
              horizon: int) -> tuple[float, float] | None:
    """
    % return of the ETF and of SPY, from the first close on or after `made_on`
    to `horizon` trading days later. None if that day hasn't happened yet.
    """
    def ret(s: pd.Series) -> float | None:
        s = s.dropna()
        s = s[s.index.tz_localize(None) >= pd.Timestamp(made_on)] if s.index.tz else \
            s[s.index >= pd.Timestamp(made_on)]
        if len(s) <= horizon:
            return None
        return (float(s.iloc[horizon]) / float(s.iloc[0]) - 1) * 100

    etf_ret, spy_ret = ret(etf_close), ret(spy_close)
    return None if etf_ret is None or spy_ret is None else (etf_ret, spy_ret)


def _fetch_closes(tickers: list[str], start: str) -> dict[str, pd.Series]:
    import yfinance as yf
    raw = yf.download(" ".join(tickers), start=start, interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            out[t] = (raw.copy() if len(tickers) == 1 else raw[t])["Close"].dropna()
        except (KeyError, TypeError):
            continue
    return out


def grade(db_path: Path = DB_PATH,
          fetch: Callable[[list[str], str], dict[str, pd.Series]] = _fetch_closes) -> int:
    """Grade every ungraded prediction whose horizon has passed. Returns the count graded."""
    with _conn(db_path) as conn:
        rows = conn.execute("SELECT id, made_on, etf, expect, horizon FROM predictions "
                            "WHERE correct IS NULL").fetchall()
        if not rows:
            return 0
        closes = fetch(sorted({r[2] for r in rows} | {"SPY"}), min(r[1] for r in rows))
        spy, graded, today = closes.get("SPY"), 0, date.today().isoformat()
        for pid, made_on, etf, expect, horizon in rows:
            if spy is None or etf not in closes:
                continue
            res = grade_one(closes[etf], spy, made_on, horizon)
            if res is None:
                continue
            etf_ret, spy_ret = res
            correct = etf_ret > spy_ret if expect == "outperform" else etf_ret < spy_ret
            conn.execute("UPDATE predictions SET graded_on=?, etf_ret=?, spy_ret=?, correct=? "
                         "WHERE id=?", (today, etf_ret, spy_ret, int(correct), pid))
            graded += 1
        return graded


# ── Scoring ──────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a hit rate of k out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half   = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def summary(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Hit rate, interval and signed edge per (source, bucket) over graded predictions."""
    with _conn(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions WHERE correct IS NOT NULL", conn)
    if df.empty:
        return pd.DataFrame(columns=["source", "bucket", "n", "correct", "hit_rate",
                                     "ci_low", "ci_high", "edge"])
    # Edge: relative return in the predicted direction (positive = the call paid off)
    sign = np.where(df.expect == "outperform", 1, -1)
    df["edge"] = sign * (df.etf_ret - df.spy_ret)
    out = (df.groupby(["source", "bucket"])
             .agg(n=("correct", "size"), correct=("correct", "sum"), edge=("edge", "mean"))
             .reset_index())
    out["hit_rate"] = out.correct / out.n
    out[["ci_low", "ci_high"]] = [wilson(int(k), int(n)) for k, n in zip(out.correct, out.n)]
    return out


def _verdict(row: pd.Series) -> str:
    if row.n < MIN_N:
        return "[dim]too few to judge[/dim]"
    if row.ci_low > 0.5:
        return "[green]works: keep trusting it[/green]"
    if row.ci_high < 0.5:
        return "[red]anti-signal: fade it or drop it[/red]"
    return "[yellow]no edge yet: treat as noise[/yellow]"


def render_report(db_path: Path = DB_PATH) -> pd.DataFrame:
    stats = summary(db_path)
    console.rule("[bold]Research Journal: do the scan's calls come true?")
    if stats.empty:
        console.print("  No graded predictions yet. Run `python main.py journal backfill`.\n")
        return stats
    tbl = Table(box=box.SIMPLE_HEAVY, show_edge=False)
    for col in ("Source", "Bucket", "Call", "N", "Hit rate", "95% range", "Avg edge", "Lesson"):
        tbl.add_column(col, justify="right" if col in ("N", "Hit rate", "95% range", "Avg edge") else "left")
    for r in stats.itertuples():
        tbl.add_row(r.source, r.bucket, f"{EXPECT[r.bucket]} SPY", str(r.n),
                    f"{r.hit_rate*100:.0f}%", f"{r.ci_low*100:.0f}–{r.ci_high*100:.0f}%",
                    f"{r.edge:+.2f}%", _verdict(pd.Series(r._asdict())))
    console.print(tbl)
    console.print(f"  [dim]A coin flip is 50%. A bucket earns trust only when its whole 95% range "
                  f"clears 50%, with n ≥ {MIN_N}. Edge = relative return in the predicted direction "
                  f"over {HORIZON} trading days. Calls never overlap in time, but ETFs on the same "
                  f"date move together (for example XLE and XOP), so the true range is somewhat "
                  f"wider.[/dim]\n")
    return stats


# ── Network wrappers ─────────────────────────────────────────────────────────

def _recent_live_etfs(db_path: Path, made_on: str, horizon: int) -> set[str]:
    """ETFs with a live call whose window is still open (about horizon trading days)."""
    since = (pd.Timestamp(made_on) - pd.Timedelta(days=int(horizon * 7 / 5))).strftime("%Y-%m-%d")
    with _conn(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT etf FROM predictions WHERE source='live' "
                            "AND horizon=? AND made_on > ? AND made_on <= ?",
                            (horizon, since, made_on)).fetchall()
    return {r[0] for r in rows}


def record_live(scan_result: dict, db_path: Path = DB_PATH, made_on: str | None = None,
                horizon: int = HORIZON) -> int:
    """
    Called after every sector scan. Logs today's calls, but skips any ETF that
    already has a call inside the current window. The scan runs daily, and
    overlapping 20-day calls are not independent evidence.
    """
    made_on = made_on or date.today().isoformat()
    skip    = _recent_live_etfs(db_path, made_on, horizon)
    preds   = [p for p in predictions_from_scan(scan_result["all"], made_on, horizon)
               if p["etf"] not in skip]
    return record(preds, db_path)


def backfill(weeks: int = 104, horizon: int = HORIZON, db_path: Path = DB_PATH) -> int:
    """
    Replay the scan over the past `weeks` from price history, record the calls
    and grade them. Scan dates are `horizon` trading days apart, so no two
    calls on the same ETF overlap.
    """
    import yfinance as yf

    from sector_scan import THEME_ETFS

    tickers = list(THEME_ETFS) + ["SPY"]
    raw = yf.download(" ".join(tickers), period="3y", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    spy  = raw["SPY"]["Close"].dropna()
    etfs = {}
    for t in THEME_ETFS:
        try:
            etfs[t] = (raw[t]["Close"].dropna(), raw[t]["Volume"].dropna())
        except KeyError:
            continue
    # Non-overlapping scan dates whose horizon has fully played out. Each scan
    # needs 200+ days of history, so the earliest usable date is row 260.
    last  = len(spy) - 1 - horizon
    dates = [spy.index[i] for i in range(last, max(last - weeks * 5, 260), -horizon)][::-1]
    new   = record(backfill_predictions(etfs, spy, dates, horizon), db_path)
    closes = {t: c for t, (c, _) in etfs.items()} | {"SPY": spy}
    graded = grade(db_path, fetch=lambda tickers, start: closes)
    logger.info(f"Journal backfill: {new} new predictions, {graded} graded")
    return graded
