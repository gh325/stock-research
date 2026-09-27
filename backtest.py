"""
backtest.py — Validate recommendations against what actually happened.

Pulls saved recommendations from the tracker DB (config.DB_PATH) and fetches each stock's actual
closing price N days later. Reports:
  - Hit rate (% of picks that were UP after N days)
  - Average return per pick
  - Best / worst pick
  - vs SPY benchmark return for the same period

Without this, the scoring weights are unvalidated guessing. With this, you
have data to tune the scorer or stop trusting it.

Usage:
  python main.py backtest                    # default: 5 days forward, last 30d
  python main.py backtest --days 10           # 10-day forward window
  python main.py backtest --since 30          # look at recs from last 30d
"""
import logging
import sqlite3
from datetime import datetime, timedelta

import yfinance as yf
from rich.console import Console
from rich.table import Table

from config import DB_PATH

logger  = logging.getLogger(__name__)
console = Console()


def run_backtest(days_forward: int = 5, since_days: int = 30) -> None:
    """
    Replay saved recommendations from the last `since_days` and check whether
    each pick's price `days_forward` later was higher (hit) or lower (miss).
    Compares against SPY's same-period return.
    """
    if not DB_PATH.exists():
        console.print(f"[yellow]No tracker DB at {DB_PATH} — run the pipeline first.[/yellow]")
        return

    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.today() - timedelta(days=since_days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        """
        SELECT run_date, rank, ticker, price_at_rec, combined_score
        FROM recommendations
        WHERE run_date >= ?
        ORDER BY run_date, rank
        """,
        (cutoff,),
    ).fetchall()
    conn.close()

    if not rows:
        console.print(f"[yellow]No recommendations in the last {since_days} days.[/yellow]")
        return

    console.rule(f"[bold blue]Backtest — last {since_days}d, {days_forward}d forward window")
    console.print(f"  Found {len(rows)} recommendation rows to replay\n")

    results: list[dict] = []
    spy_cache: dict[str, float] = {}

    for run_date, rank, ticker, price_at_rec, combined_score in rows:
        target_date = (
            datetime.strptime(run_date, "%Y-%m-%d") + timedelta(days=days_forward)
        ).strftime("%Y-%m-%d")

        # Don't backtest into the future
        if datetime.strptime(target_date, "%Y-%m-%d") > datetime.today():
            continue

        try:
            df = yf.Ticker(ticker).history(start=run_date, period="1mo")
            if df.empty:
                continue
            # Find the closing price on or after target_date
            df = df[df.index.strftime("%Y-%m-%d") >= target_date]
            if df.empty:
                continue
            future_price = float(df["Close"].iloc[0])
            entry_price  = float(price_at_rec) if price_at_rec else float(yf.Ticker(ticker).history(start=run_date, period="5d")["Close"].iloc[0])
            ret_pct      = (future_price / entry_price - 1) * 100

            # SPY benchmark for same window
            spy_key = f"{run_date}_{target_date}"
            if spy_key not in spy_cache:
                spy_df = yf.Ticker("SPY").history(start=run_date, period="1mo")
                if not spy_df.empty:
                    spy_entry  = float(spy_df["Close"].iloc[0])
                    spy_target = spy_df[spy_df.index.strftime("%Y-%m-%d") >= target_date]
                    if not spy_target.empty:
                        spy_future = float(spy_target["Close"].iloc[0])
                        spy_cache[spy_key] = (spy_future / spy_entry - 1) * 100
                    else:
                        spy_cache[spy_key] = 0
                else:
                    spy_cache[spy_key] = 0
            spy_ret = spy_cache[spy_key]

            results.append({
                "run_date": run_date,
                "rank":     rank,
                "ticker":   ticker,
                "entry":    entry_price,
                "exit":     future_price,
                "ret_pct":  ret_pct,
                "spy_ret":  spy_ret,
                "alpha":    ret_pct - spy_ret,
                "hit":      ret_pct > 0,
                "score":    combined_score,
            })
        except Exception as exc:
            logger.debug(f"{ticker} backtest fetch error: {exc}")

    if not results:
        console.print("[yellow]No completable backtest rows (all targets in the future or fetch errors).[/yellow]")
        return

    # ── Aggregate stats ─────────────────────────────────────────────────────
    n          = len(results)
    hits       = sum(1 for r in results if r["hit"])
    avg_ret    = sum(r["ret_pct"] for r in results) / n
    avg_spy    = sum(r["spy_ret"] for r in results) / n
    avg_alpha  = sum(r["alpha"]   for r in results) / n
    best       = max(results, key=lambda r: r["ret_pct"])
    worst      = min(results, key=lambda r: r["ret_pct"])

    summary = Table(title=f"Backtest Summary ({n} picks, {days_forward}d forward)", show_lines=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Hit rate (% positive)",   f"{hits/n*100:.1f}% ({hits}/{n})")
    summary.add_row("Avg pick return",         f"{avg_ret:+.2f}%")
    summary.add_row("Avg SPY return",          f"{avg_spy:+.2f}%")
    summary.add_row("Avg alpha (pick − SPY)",  f"{avg_alpha:+.2f}%")
    summary.add_row("Best pick",                f"{best['ticker']} {best['ret_pct']:+.2f}% (run {best['run_date']})")
    summary.add_row("Worst pick",               f"{worst['ticker']} {worst['ret_pct']:+.2f}% (run {worst['run_date']})")
    console.print(summary)

    # ── Per-pick detail (last 20) ──────────────────────────────────────────
    detail = Table(title=f"Most Recent {min(20, n)} Picks", show_lines=False, highlight=True)
    detail.add_column("Run Date", no_wrap=True)
    detail.add_column("Rank", justify="center")
    detail.add_column("Ticker", style="bold yellow")
    detail.add_column("Entry",  justify="right")
    detail.add_column("Exit",   justify="right")
    detail.add_column("Return", justify="right")
    detail.add_column("SPY",    justify="right")
    detail.add_column("Alpha",  justify="right")
    for r in results[-20:]:
        ret_color = "green" if r["ret_pct"] > 0 else "red"
        alpha_color = "green" if r["alpha"] > 0 else "red"
        detail.add_row(
            r["run_date"], str(r["rank"]), r["ticker"],
            f"${r['entry']:.2f}", f"${r['exit']:.2f}",
            f"[{ret_color}]{r['ret_pct']:+.2f}%[/{ret_color}]",
            f"{r['spy_ret']:+.2f}%",
            f"[{alpha_color}]{r['alpha']:+.2f}%[/{alpha_color}]",
        )
    console.print(detail)

    # ── Honest interpretation ───────────────────────────────────────────────
    console.print()
    if avg_alpha > 1.0:
        console.print(f"[green]Picks beat SPY by {avg_alpha:.2f}% on average — directional edge above noise.[/green]")
    elif avg_alpha > 0:
        console.print(f"[yellow]Picks beat SPY by only {avg_alpha:.2f}% — within noise band; not yet validated.[/yellow]")
    else:
        console.print(f"[red]Picks UNDERPERFORMED SPY by {-avg_alpha:.2f}% — buying SPY would have been better.[/red]")
    console.print(f"[dim]Sample size {n} is {'small (need 50+)' if n < 50 else 'reasonable'} — interpret accordingly.[/dim]")
