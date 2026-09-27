"""Backtest wiring. No network: only checks where it reads from."""
import backtest
import config
import tracker


def test_backtest_reads_the_db_tracker_writes() -> None:
    # Regression: backtest read data/tracker.db while tracker wrote
    # data/paper_trades.db, so `main.py backtest` never found any picks.
    assert backtest.DB_PATH == config.DB_PATH == tracker.DB_PATH
