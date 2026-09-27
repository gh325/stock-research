"""
tracker.py — SQLite paper trade log.

Tracks simulated buy/sell decisions so you can measure how well the
recommendations actually perform over time. No real money involved.

Tables:
  recommendations  — every pipeline run's top-5 picks (auto-populated)
  paper_trades     — manual buy/sell entries you log via the CLI
"""
import logging
import sqlite3
from datetime import date, timedelta

from config import DB_PATH

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date        TEXT    NOT NULL,
                ticker          TEXT    NOT NULL,
                rank            INTEGER NOT NULL,
                price_at_rec    REAL,
                combined_score  REAL,
                technical_score REAL,
                sentiment_score REAL
            );

            CREATE TABLE IF NOT EXISTS paper_trades (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker     TEXT    NOT NULL,
                action     TEXT    NOT NULL,   -- BUY or SELL
                price      REAL    NOT NULL,
                shares     INTEGER NOT NULL DEFAULT 1,
                trade_date TEXT    NOT NULL,
                notes      TEXT
            );

            CREATE TABLE IF NOT EXISTS analyst_targets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                run_date    TEXT    NOT NULL,
                target_mean REAL    NOT NULL,
                UNIQUE(ticker, run_date)
            );
        """)


def save_recommendations(picks: list[dict], run_date: str | None = None) -> None:
    """Persist the pipeline's top picks so you can track accuracy over time."""
    run_date = run_date or date.today().isoformat()
    init_db()
    with _conn() as conn:
        for pick in picks:
            conn.execute(
                """
                INSERT INTO recommendations
                    (run_date, ticker, rank, price_at_rec,
                     combined_score, technical_score, sentiment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_date,
                    pick["ticker"],
                    pick.get("rank", 0),
                    pick.get("last_price"),
                    pick.get("combined_score"),
                    pick.get("technical_score"),
                    pick.get("sentiment_score"),
                ),
            )


def log_trade(
    ticker: str,
    action: str,
    price: float,
    shares: int = 1,
    notes: str = "",
) -> None:
    """Record a paper buy or sell in the database."""
    action = action.upper()
    if action not in ("BUY", "SELL"):
        raise ValueError("action must be BUY or SELL")
    init_db()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO paper_trades (ticker, action, price, shares, trade_date, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker, action, price, shares, date.today().isoformat(), notes),
        )
    logger.info(f"Paper {action}: {shares}× {ticker} @ ${price:.2f}")


def get_open_positions() -> list[dict]:
    """
    Return positions with a non-zero net share count.
    (Does not fetch current prices — do that externally to show P&L.)
    """
    init_db()
    with _conn() as conn:
        cursor = conn.execute(
            """
            SELECT
                ticker,
                SUM(CASE WHEN action='BUY'  THEN  shares ELSE 0 END) -
                SUM(CASE WHEN action='SELL' THEN  shares ELSE 0 END)  AS net_shares,
                AVG(CASE WHEN action='BUY'  THEN  price  END)          AS avg_buy_price
            FROM paper_trades
            GROUP BY ticker
            HAVING net_shares > 0
            ORDER BY ticker
            """
        )
        return [dict(row) for row in cursor.fetchall()]


def get_recommendation_history() -> list[dict]:
    """Return all past pipeline recommendations, newest first."""
    init_db()
    with _conn() as conn:
        cursor = conn.execute(
            "SELECT * FROM recommendations ORDER BY run_date DESC, rank ASC"
        )
        return [dict(row) for row in cursor.fetchall()]


def store_analyst_target(ticker: str, target_mean: float | None, run_date: str | None = None) -> None:
    """
    Persist today's analyst consensus target for a ticker.
    Called every run so we can track target drift over time.
    Silently skips if target_mean is None or zero.
    """
    if not target_mean or float(target_mean) <= 0:
        return
    run_date = run_date or date.today().isoformat()
    init_db()
    with _conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analyst_targets (ticker, run_date, target_mean)
            VALUES (?, ?, ?)
            """,
            (ticker, run_date, float(target_mean)),
        )


def get_analyst_target_velocity(
    ticker: str,
    current_target: float | None,
    days: int = 30,
    min_pct: float = 3.0,
) -> dict | None:
    """
    Compare current_target against the oldest stored target within `days`.
    Call this BEFORE store_analyst_target so we compare against prior history.

    Returns {"pct_change": float, "from_target": float, "days_ago": int}
    or None if insufficient history or change below min_pct noise floor.

    Examples:
      +12% in 14 days → analysts raised their targets after an earnings beat
      -8%  in 21 days → coverage cut targets, street is ahead of the move
    """
    if not current_target or float(current_target) <= 0:
        return None
    current_target = float(current_target)
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    today  = date.today().isoformat()
    init_db()
    with _conn() as conn:
        cur = conn.execute(
            """
            SELECT target_mean, run_date FROM analyst_targets
            WHERE ticker = ? AND run_date >= ? AND run_date < ? AND target_mean > 0
            ORDER BY run_date ASC LIMIT 1
            """,
            (ticker, cutoff, today),
        )
        row = cur.fetchone()

    if not row:
        return None
    from_t   = float(row["target_mean"])
    days_ago = (date.today() - date.fromisoformat(row["run_date"])).days
    if from_t == 0:
        return None
    pct = (current_target - from_t) / from_t * 100
    if abs(pct) < min_pct:
        return None
    return {
        "pct_change":  round(pct, 1),
        "from_target": round(from_t, 2),
        "days_ago":    max(days_ago, 1),
    }


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
