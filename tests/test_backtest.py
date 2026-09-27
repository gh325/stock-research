"""Backtest wiring. No network: only checks where it reads from."""
import backtest
import config
import tracker


def test_backtest_reads_the_db_tracker_writes() -> None:
    # Regression: backtest read data/tracker.db while tracker wrote
    # data/paper_trades.db, so `main.py backtest` never found any picks.
    assert backtest.DB_PATH == config.DB_PATH == tracker.DB_PATH


def _point_dbs_at(monkeypatch, path) -> None:
    monkeypatch.setattr(backtest, "DB_PATH", path)
    monkeypatch.setattr(tracker, "DB_PATH", path)


def test_backtest_survives_a_db_without_recommendations(monkeypatch, tmp_path) -> None:
    # Regression: on a fresh install the journal creates the DB with only its
    # `predictions` table, and backtest crashed with "no such table:
    # recommendations".
    import sqlite3
    db = tmp_path / "paper_trades.db"
    sqlite3.connect(db).execute("CREATE TABLE predictions (id INTEGER)").connection.commit()
    _point_dbs_at(monkeypatch, db)
    backtest.run_backtest()          # must not raise


def test_backtest_on_a_missing_db_does_not_crash(monkeypatch, tmp_path) -> None:
    _point_dbs_at(monkeypatch, tmp_path / "paper_trades.db")
    backtest.run_backtest()
