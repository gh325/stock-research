"""
monitor.py — Intraday alert monitor for top picks or a full watchlist.

Polls tickers every N seconds during US market hours (9:30 AM – 4:00 PM ET).

TWO MODES
─────────
1. Picks mode (default):
   Watches the 5–10 picks from the latest pipeline run (top5_picks_*.json).
   Uses pre-computed day/swing targets from targets.py.

   python main.py monitor
   python main.py monitor --interval 60

2. Watchlist mode (--watchlist):
   Watches every ticker in a watchlist file (e.g. ai_ecosystem = 92 tickers).
   Computes ATR-based intraday targets on the fly at startup.
   Shows a compact top-movers heartbeat — not 92 lines per poll.

   python main.py monitor --watchlist ai_ecosystem
   python main.py monitor --watchlist ai_chips --interval 60

ALERTS (fire at most once per ticker per session)
─────────────────────────────────────────────────
  SURGE      +3% intraday on >1.5× average volume  → possible entry
  BREAKOUT   price crosses yesterday's high on >1.5× volume → momentum confirmed
  DROP       -3% intraday                           → re-evaluate / stop
  DAY TARGET price hits ATR-based intraday target   → consider taking profit
  DAY STOP   price hits ATR-based intraday stop     → exit signal

Output: coloured Rich terminal + macOS notification (Ping sound).
Allow Terminal in System Settings → Notifications if alerts don't appear.
"""
import json
import logging
import subprocess
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import yfinance as yf
from rich.console import Console

from config import OUTPUT_DIR

logger  = logging.getLogger(__name__)
console = Console()

ET            = ZoneInfo("America/New_York")
MARKET_OPEN   = dtime(9, 30)
MARKET_CLOSE  = dtime(16, 0)

DEFAULT_INTERVAL = 300   # 5 minutes
ALERT_PCT_UP     = 3.0
ALERT_PCT_DOWN   = -3.0
VOL_MULT         = 1.5
ATR_PERIODS      = 14    # days for ATR calculation
TOP_MOVERS_N     = 5     # gainers/losers to show per heartbeat


def monitor_watchlist(tickers: list[str], poll_interval: int = DEFAULT_INTERVAL) -> None:
    """
    Watch every ticker in a watchlist for intraday breakout/breakdown signals.

    On startup fetches 20d daily data for each ticker to compute:
      - ATR-based day target  (yesterday close + 1.5 × ATR)
      - ATR-based day stop    (yesterday close − 1.0 × ATR)
      - Yesterday's high/low  (breakout / breakdown level)
      - 20-day average volume (volume comparison)

    Each poll prints a compact top-movers summary and fires alerts.
    """
    if not tickers:
        console.print("[yellow]No tickers to watch.[/yellow]")
        return

    console.rule("[bold blue]Watchlist Monitor — initialising")
    console.print(f"Loading baseline data for [bold]{len(tickers)}[/bold] tickers … (takes ~30s)")

    baselines: dict[str, dict] = {}
    failed: list[str] = []

    # Batch fetch daily data for ATR / levels
    raw = yf.download(
        tickers,
        period="25d",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = raw
            else:
                df = raw[ticker] if ticker in raw.columns.get_level_values(0) else None

            if df is None or df.empty or len(df) < ATR_PERIODS + 1:
                failed.append(ticker)
                continue

            df = df.dropna(subset=["Close"])
            closes  = df["Close"].values
            highs   = df["High"].values
            lows    = df["Low"].values
            volumes = df["Volume"].values

            # True range → ATR
            tr = [max(h - l, abs(h - cp), abs(l - cp))
                  for h, l, cp in zip(highs[1:], lows[1:], closes[:-1])]
            atr = sum(tr[-ATR_PERIODS:]) / ATR_PERIODS

            prev_close = float(closes[-1])
            prev_high  = float(highs[-1])
            prev_low   = float(lows[-1])
            avg_vol    = float(volumes[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())

            baselines[ticker] = {
                "atr":        atr,
                "prev_close": prev_close,
                "prev_high":  prev_high,
                "prev_low":   prev_low,
                "avg_vol":    avg_vol,
                "day_target": prev_close + 1.5 * atr,
                "day_stop":   prev_close - 1.0 * atr,
            }
        except Exception as exc:
            logger.warning(f"{ticker} baseline failed: {exc}")
            failed.append(ticker)

    watch = [t for t in tickers if t in baselines]
    console.print(
        f"[green]Ready:[/green] {len(watch)} tickers  |  "
        f"[dim]skipped {len(failed)}: {', '.join(failed[:8])}{'…' if len(failed) > 8 else ''}[/dim]"
        if failed else f"[green]Ready:[/green] {len(watch)} tickers"
    )
    console.print(
        f"Poll: [cyan]{poll_interval}s[/cyan]  |  "
        f"Surge: [green]+{ALERT_PCT_UP}%[/green] on {VOL_MULT}× vol  |  "
        f"Drop: [red]{ALERT_PCT_DOWN}%[/red]  |  "
        f"Breakout: above yesterday's high on {VOL_MULT}× vol"
    )
    console.print("[dim]Ctrl+C to stop.[/dim]\n")

    fired: set[str] = set()

    try:
        while True:
            now_et    = datetime.now(ET).time()
            now_local = datetime.now().strftime("%H:%M:%S")

            if not (MARKET_OPEN <= now_et <= MARKET_CLOSE):
                console.print(
                    f"[dim]{now_local} | Market closed (ET {now_et.strftime('%H:%M')}). "
                    f"Sleeping 60s …[/dim]"
                )
                time.sleep(60)
                continue

            movers: list[tuple[float, float, str]] = []  # (pct, vol_ratio, ticker)
            for ticker in watch:
                try:
                    pct, vol_ratio = _check_watchlist_ticker(
                        ticker, baselines[ticker], fired
                    )
                    movers.append((pct, vol_ratio, ticker))
                except Exception as exc:
                    logger.debug(f"{ticker} poll failed: {exc}")

            # Compact heartbeat — top N gainers / losers only
            movers.sort(key=lambda x: x[0], reverse=True)
            gainers  = [m for m in movers if m[0] > 0][:TOP_MOVERS_N]
            losers   = [m for m in movers if m[0] < 0][-TOP_MOVERS_N:][::-1]

            parts: list[str] = [f"[dim]{now_local}[/dim] | [bold]{len(watch)} tickers[/bold]"]
            if gainers:
                g_str = "  ".join(
                    f"[green]{t} +{p:.1f}%[/green]" for p, _, t in gainers
                )
                parts.append(f"↑ {g_str}")
            if losers:
                l_str = "  ".join(
                    f"[red]{t} {p:.1f}%[/red]" for p, _, t in losers
                )
                parts.append(f"↓ {l_str}")
            console.print("  ".join(parts))

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")


def _check_watchlist_ticker(
    ticker: str, baseline: dict, fired: set
) -> tuple[float, float]:
    """
    Fetch latest intraday quote, fire alerts, return (pct_from_prev_close, vol_ratio).
    """
    yft = yf.Ticker(ticker)
    df  = yft.history(period="1d", interval="5m")
    if df.empty:
        return 0.0, 0.0

    last_p       = float(df["Close"].iloc[-1])
    intraday_vol = float(df["Volume"].sum())
    prev_close   = baseline["prev_close"]
    avg_vol      = baseline["avg_vol"]
    day_target   = baseline["day_target"]
    day_stop     = baseline["day_stop"]
    prev_high    = baseline["prev_high"]
    vol_ratio    = intraday_vol / avg_vol if avg_vol > 0 else 0.0
    pct          = (last_p / prev_close - 1) * 100

    # SURGE: +3% intraday from prev close, above-avg volume
    key = f"{ticker}:surge"
    if pct >= ALERT_PCT_UP and vol_ratio >= VOL_MULT and key not in fired:
        _alert(
            f"{ticker} SURGE +{pct:.1f}%  vol {vol_ratio:.1f}× avg  "
            f"@ ${last_p:.2f} — possible day-trade entry  "
            f"[target ${day_target:.2f} | stop ${day_stop:.2f}]"
        )
        fired.add(key)

    # BREAKOUT: above yesterday's high on volume
    key = f"{ticker}:breakout"
    if last_p > prev_high and vol_ratio >= VOL_MULT and key not in fired:
        _alert(
            f"{ticker} BREAKOUT above prev high ${prev_high:.2f}  "
            f"vol {vol_ratio:.1f}× avg  @ ${last_p:.2f} — momentum confirmed"
        )
        fired.add(key)

    # DROP: -3% from prev close
    key = f"{ticker}:drop"
    if pct <= ALERT_PCT_DOWN and key not in fired:
        _alert(
            f"{ticker} DROP {pct:.1f}%  @ ${last_p:.2f} — "
            f"breakdown watch  [stop ${day_stop:.2f}]"
        )
        fired.add(key)

    # DAY TARGET hit
    key = f"{ticker}:day_tgt"
    if last_p >= day_target and key not in fired:
        _alert(
            f"{ticker} HIT DAY TARGET ${day_target:.2f} @ ${last_p:.2f} — "
            f"consider taking intraday profit"
        )
        fired.add(key)

    # DAY STOP hit
    key = f"{ticker}:day_stop"
    if last_p <= day_stop and key not in fired:
        _alert(
            f"{ticker} HIT DAY STOP ${day_stop:.2f} @ ${last_p:.2f} — "
            f"exit signal"
        )
        fired.add(key)

    return pct, vol_ratio


def monitor_top_picks(poll_interval: int = DEFAULT_INTERVAL) -> None:
    """Main monitor loop. Blocks until Ctrl+C."""
    picks = _load_latest_picks()
    if not picks:
        console.print("[yellow]No saved picks found. Run the pipeline first: python main.py run[/yellow]")
        return

    tickers = [p["ticker"] for p in picks]
    targets = {p["ticker"]: p.get("targets", {}) for p in picks}

    fired: set[str] = set()

    console.rule("[bold blue]Intraday Monitor")
    console.print(f"[bold]Watching:[/bold] {', '.join(tickers)}")
    console.print(
        f"Poll: [cyan]{poll_interval}s[/cyan]  |  "
        f"Surge alert: [green]+{ALERT_PCT_UP}%[/green] on {VOL_MULT}× volume  |  "
        f"Drop alert: [red]{ALERT_PCT_DOWN}%[/red]"
    )
    console.print("[dim]Ctrl+C to stop.[/dim]\n")

    try:
        while True:
            now_et    = datetime.now(ET).time()
            now_local = datetime.now().strftime("%H:%M:%S")

            if not (MARKET_OPEN <= now_et <= MARKET_CLOSE):
                console.print(
                    f"[dim]{now_local} | Market closed (ET {now_et.strftime('%H:%M')}). "
                    f"Sleeping 60s …[/dim]"
                )
                time.sleep(60)
                continue

            statuses: list[str] = []
            for ticker in tickers:
                try:
                    statuses.append(_check_ticker(ticker, targets[ticker], fired))
                except Exception as exc:
                    logger.warning(f"{ticker} check failed: {exc}")
                    statuses.append(f"[dim]{ticker} ERR[/dim]")

            console.print(f"[dim]{now_local}[/dim] | " + "  ".join(statuses))
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")


def _check_ticker(ticker: str, tgt: dict, fired: set) -> str:
    """Fetch intraday data, fire alerts if criteria met, return heartbeat string."""
    yft = yf.Ticker(ticker)
    df  = yft.history(period="1d", interval="5m")
    if df.empty or len(df) < 2:
        return f"[dim]{ticker} no-data[/dim]"

    open_p = float(df["Open"].iloc[0])
    last_p = float(df["Close"].iloc[-1])
    pct    = (last_p / open_p - 1) * 100

    # 5-day avg daily volume (excluding today's partial bar)
    daily   = yft.history(period="6d", interval="1d")
    avg_vol = (
        float(daily["Volume"].iloc[:-1].tail(5).mean())
        if len(daily) >= 2 else 0
    )
    intraday_vol = float(df["Volume"].sum())
    vol_ratio    = intraday_vol / avg_vol if avg_vol > 0 else 0

    # ── Alert conditions (each fires at most once per session) ──────────────
    key = f"{ticker}:surge"
    if pct >= ALERT_PCT_UP and vol_ratio >= VOL_MULT and key not in fired:
        _alert(
            f"{ticker} SURGE +{pct:.1f}%  vol {vol_ratio:.1f}× avg  "
            f"@ ${last_p:.2f} — possible day-trade entry"
        )
        fired.add(key)

    key = f"{ticker}:drop"
    if pct <= ALERT_PCT_DOWN and key not in fired:
        _alert(f"{ticker} DROP {pct:.1f}%  @ ${last_p:.2f} — re-evaluate / stop-loss")
        fired.add(key)

    day_tgt = tgt.get("day_target", 0)
    key     = f"{ticker}:day_tgt"
    if day_tgt and last_p >= day_tgt and key not in fired:
        _alert(
            f"{ticker} HIT DAY TARGET ${day_tgt:.2f} @ ${last_p:.2f} — "
            f"consider taking day-trade profit"
        )
        fired.add(key)

    swing_tgt = tgt.get("swing_target", 0)
    key       = f"{ticker}:swing_tgt"
    if swing_tgt and last_p >= swing_tgt and key not in fired:
        _alert(
            f"{ticker} HIT SWING TARGET ${swing_tgt:.2f} @ ${last_p:.2f} — "
            f"consider taking swing-trade profit"
        )
        fired.add(key)

    # ── Heartbeat status string ─────────────────────────────────────────────
    arrow = "↑" if pct > 0.1 else ("↓" if pct < -0.1 else "→")
    color = "green" if pct >= 1 else ("red" if pct <= -1 else "white")
    return f"[{color}]{ticker} {arrow}{abs(pct):.1f}%[/{color}]"


def _alert(message: str) -> None:
    """Print colourful terminal alert + macOS notification."""
    console.print(f"\n[bold black on yellow] 🔔 ALERT [/bold black on yellow] [bold]{message}[/bold]\n")
    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "{message}" with title "Stock Research Alert" '
                f'sound name "Ping"',
            ],
            timeout=5,
            capture_output=True,
        )
    except Exception as exc:
        logger.debug(f"osascript notification failed: {exc}")


def _load_latest_picks() -> list[dict]:
    """Load the most recent top5_picks_*.json from output/."""
    files = sorted(OUTPUT_DIR.glob("top5_picks_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return []
    data = json.loads(files[0].read_text())
    return data.get("top_5_picks", [])
