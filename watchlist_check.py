"""
watchlist_check.py — Pre-run health check for a watchlist.

A watchlist goes stale quietly: tickers get acquired, themes roll over, a
name breaks trend while its peers lead. This check runs before every
`--watchlist` run and flags what a veteran would adjust before trading it.

Checks
  Market   — regime (SPY vs 200 SMA, VIX) from macro.get_macro_snapshot
  Theme    — the watchlist's benchmark ETF vs SPY: Leading / Emerging /
             Fading / Broken
  Ticker   — data and liquidity, trend vs the theme ETF, stretch, earnings
             within EARNINGS_WINDOW_DAYS, news sentiment
  Trending — top holdings of the benchmark ETFs that are missing from the
             watchlist and are leading their theme

The check is advisory. It never edits the watchlist file.

Benchmarks come from a header line in the watchlist file:
    # benchmark: CIBR, HACK, BUG
The first ETF is the theme benchmark. All of them seed the "missing
leaders" suggestions. With no header, the check uses SPY and skips
suggestions.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from rich import box
from rich.console import Console
from rich.table import Table

from config import MIN_AVG_VOLUME, MIN_PRICE, OUTPUT_DIR

logger  = logging.getLogger(__name__)
console = Console(width=160)   # wide enough that tables never wrap

EARNINGS_WINDOW_DAYS = 7      # earnings this close = event risk on any hold
STRETCH_RSI          = 75     # matches the pipeline's overbought flag
STRETCH_SMA200_PCT   = 50     # +50% above 200 SMA = stretched
LAG_RS_PCT           = -10    # 3mo return vs theme ETF below this = laggard
LEAD_RS_PCT          = 10     # 3mo return vs theme ETF above this = leader
NEG_SENTIMENT        = 35     # 0–100 VADER scale; below this = negative news
MIN_HEADLINES        = 3      # fewer headlines = sentiment too thin to trust
MAX_SUGGESTIONS      = 5
_STATUS_ORDER        = {"REMOVE": 0, "BENCH": 1, "WATCH": 2, "KEEP": 3}


# ── Watchlist metadata ───────────────────────────────────────────────────────

def load_benchmarks(name: str) -> list[str]:
    """Read the `# benchmark: A, B` header from watchlists/<name>.txt."""
    from ingest import load_watchlist_header
    raw = load_watchlist_header(name, "benchmark") or ""
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


# ── Indicators (pure — no network) ───────────────────────────────────────────

def _ret(close: pd.Series, n: int) -> float | None:
    if len(close) <= n:
        return None
    return (float(close.iloc[-1]) / float(close.iloc[-1 - n]) - 1) * 100


def _rsi(close: pd.Series, n: int = 14) -> float | None:
    if len(close) <= n:
        return None
    delta = close.diff()
    up    = delta.clip(lower=0).ewm(alpha=1 / n).mean().iloc[-1]
    down  = (-delta.clip(upper=0)).ewm(alpha=1 / n).mean().iloc[-1]
    if down == 0:
        return 100.0
    return float(100 - 100 / (1 + up / down))


def _sma_pct(close: pd.Series, n: int) -> float | None:
    if len(close) < n:
        return None
    return (float(close.iloc[-1]) / float(close.rolling(n).mean().iloc[-1]) - 1) * 100


def ticker_metrics(df: pd.DataFrame | None, bench_ret3m: float | None) -> dict:
    """Trend, stretch and liquidity numbers for one ticker's OHLCV frame."""
    if df is None or df.empty or len(df.dropna(subset=["Close"])) < 20:
        return {"has_data": False}
    df    = df.dropna(subset=["Close"])
    close = df["Close"]
    ret3m = _ret(close, 63)
    return {
        "has_data":  True,
        "price":     float(close.iloc[-1]),
        "avg_vol":   float(df["Volume"].tail(20).mean()),
        "rows":      len(close),
        "rsi":       _rsi(close),
        "vs50":      _sma_pct(close, 50),
        "vs200":     _sma_pct(close, 200),
        "ret1m":     _ret(close, 21),
        "ret3m":     ret3m,
        "rs3m":      (ret3m - bench_ret3m) if ret3m is not None and bench_ret3m is not None else None,
    }


def classify_theme(etf_close: pd.Series, spy_close: pd.Series) -> dict:
    """Leading / Emerging / Fading / Broken for a theme ETF vs SPY."""
    rs1m  = (_ret(etf_close, 21) or 0) - (_ret(spy_close, 21) or 0)
    rs3m  = (_ret(etf_close, 63) or 0) - (_ret(spy_close, 63) or 0)
    vs200 = _sma_pct(etf_close, 200)
    above = vs200 is None or vs200 > 0

    if above and rs3m > 0 and rs1m >= 0:
        status, note = "Leading",  "theme beats SPY on 1mo and 3mo — tailwind"
    elif above and rs3m > 0:
        status, note = "Fading",   "3mo leader losing steam this month — tighten stops"
    elif rs1m > 0:
        status, note = "Emerging", "turning up vs SPY — early, wait for confirmation"
    elif not above:
        status, note = "Broken",   "below 200 SMA and lagging SPY — size down or skip"
    else:
        status, note = "Lagging",  "above 200 SMA but trailing SPY"
    return {"status": status, "note": note, "rs1m": rs1m, "rs3m": rs3m, "vs200": vs200}


def assess_ticker(m: dict, earnings=None, sentiment: dict | None = None,
                  today: datetime | None = None) -> tuple[str, list[str]]:
    """
    Return (status, reasons). Status is the worst that applies:
    REMOVE (pipeline can't use it) > BENCH (broken vs theme) > WATCH > KEEP.
    """
    today   = today or datetime.now()
    status  = "KEEP"
    reasons: list[str] = []

    def bump(new: str, why: str) -> None:
        nonlocal status
        reasons.append(why)
        if _STATUS_ORDER[new] < _STATUS_ORDER[status]:
            status = new

    if not m.get("has_data"):
        bump("REMOVE", "no price data — delisted, acquired or renamed?")
        return status, reasons

    if m["price"] < MIN_PRICE:
        bump("REMOVE", f"price ${m['price']:.2f} < ${MIN_PRICE:.0f} — pipeline drops it")
    if m["avg_vol"] < MIN_AVG_VOLUME:
        bump("REMOVE", f"avg volume {m['avg_vol']/1e3:.0f}k < {MIN_AVG_VOLUME/1e3:.0f}k — pipeline drops it")

    vs50, vs200, rs3m = m.get("vs50"), m.get("vs200"), m.get("rs3m")
    if vs200 is None:
        reasons.append(f"only {m['rows']} days of history — no 200 SMA yet")
    elif vs200 < 0 and (vs50 or 0) < 0 and rs3m is not None and rs3m < LAG_RS_PCT:
        bump("BENCH", f"below 50 & 200 SMA, lagging theme {rs3m:+.0f}% (3mo)")

    rsi = m.get("rsi")
    if rsi is not None and rsi > STRETCH_RSI:
        bump("WATCH", f"RSI {rsi:.0f} — overbought")
    if vs200 is not None and vs200 > STRETCH_SMA200_PCT:
        bump("WATCH", f"{vs200:+.0f}% vs 200 SMA — stretched")

    if earnings is not None:
        ed = pd.Timestamp(earnings).to_pydatetime().replace(tzinfo=None)
        if today.date() <= ed.date() <= (today + timedelta(days=EARNINGS_WINDOW_DAYS)).date():
            bump("WATCH", f"earnings {ed:%m/%d} — event risk")

    if sentiment and sentiment.get("headline_count", 0) >= MIN_HEADLINES \
            and sentiment.get("sentiment_score", 50) < NEG_SENTIMENT:
        bump("WATCH", f"negative news ({sentiment['sentiment_score']:.0f}/100)")

    if status == "KEEP" and rs3m is not None and rs3m > LEAD_RS_PCT and (vs50 or 0) > 0:
        reasons.append(f"theme leader {rs3m:+.0f}% vs ETF (3mo)")

    return status, reasons


# ── Network helpers ──────────────────────────────────────────────────────────

def _download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    raw = yf.download(" ".join(tickers), period="1y", interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = (raw.copy() if len(tickers) == 1 else raw[t]).dropna(how="all")
        except (KeyError, TypeError):
            out[t] = pd.DataFrame()
    return out


def _close(data: dict[str, pd.DataFrame], ticker: str) -> pd.Series:
    """Close series for one downloaded ticker; empty if it failed to download."""
    return data.get(ticker, pd.DataFrame()).get("Close", pd.Series(dtype=float)).dropna()


def _top_holdings(etf: str) -> list[str]:
    try:
        return [str(t).upper() for t in yf.Ticker(etf).funds_data.top_holdings.index]
    except Exception as exc:
        logger.debug(f"{etf} holdings fetch error: {exc}")
        return []


def _events(tickers: list[str]) -> tuple[dict, dict]:
    """Next earnings date and news sentiment per ticker, fetched in parallel."""
    import ingest
    import sentiment

    def one(t: str):
        headlines = ingest.fetch_news_for_tickers([t]).get(t, [])
        return t, ingest.fetch_next_earnings_date(t), headlines

    earnings, news = {}, {}
    # 8 workers: fast enough for ~90 tickers without exhausting sockets
    with ThreadPoolExecutor(max_workers=8) as pool:
        for t, ed, headlines in pool.map(one, tickers):
            earnings[t] = ed
            news[t]     = headlines
    return earnings, sentiment.score_sentiment_for_tickers(news)


# ── Entry point ──────────────────────────────────────────────────────────────

def check_watchlist(name: str, macro: dict | None = None) -> dict:
    """Run every check for watchlists/<name>.txt and return the findings."""
    import ingest
    import macro as macro_mod

    tickers    = ingest.load_watchlist(name)
    benchmarks = load_benchmarks(name)
    primary    = benchmarks[0] if benchmarks else "SPY"
    macro      = macro or macro_mod.get_macro_snapshot()

    holdings: list[str] = []
    for etf in benchmarks:
        holdings += [h for h in _top_holdings(etf) if h not in tickers and h not in holdings]

    data  = _download(list(dict.fromkeys(tickers + holdings + benchmarks + ["SPY"])))
    spy   = _close(data, "SPY")
    bench = _close(data, primary)
    bench_ret3m = _ret(bench, 63)
    theme = classify_theme(bench, spy) if len(bench) and len(spy) else None

    metrics  = {t: ticker_metrics(data.get(t), bench_ret3m) for t in tickers}
    live     = [t for t in tickers if metrics[t]["has_data"]]
    earnings, sents = _events(live)

    rows = []
    for t in tickers:
        m = metrics[t]
        status, reasons = assess_ticker(m, earnings.get(t), sents.get(t))
        rows.append({
            "ticker": t, "status": status, "reasons": "; ".join(reasons),
            "price": m.get("price"), "rsi": m.get("rsi"), "vs200": m.get("vs200"),
            "rs3m_vs_theme": m.get("rs3m"),
            "sentiment": (sents.get(t) or {}).get("sentiment_score"),
        })
    rows.sort(key=lambda r: (_STATUS_ORDER[r["status"]], -(r["rs3m_vs_theme"] or -999)))

    suggestions = []
    for h in holdings:
        m = ticker_metrics(data.get(h), bench_ret3m)
        if not m["has_data"] or m["price"] < MIN_PRICE or m["avg_vol"] < MIN_AVG_VOLUME:
            continue
        if (m["vs50"] or 0) > 0 and (m["vs200"] or 0) > 0 and (m["rs3m"] or 0) > 0:
            suggestions.append({"ticker": h, "rs3m_vs_theme": m["rs3m"],
                                "vs200": m["vs200"], "rsi": m["rsi"]})
    suggestions.sort(key=lambda s: -s["rs3m_vs_theme"])

    return {
        "name": name, "benchmark": primary, "macro": macro, "theme": theme,
        "rows": rows, "suggestions": suggestions[:MAX_SUGGESTIONS],
    }


def render(result: dict) -> None:
    """Print the check as a short rich report."""
    macro, theme = result["macro"], result["theme"]
    console.rule(f"[bold]Watchlist check: {result['name']}")

    if macro.get("available"):
        console.print(f"  Market: [{macro['regime_color']}]{macro['regime']}[/]  "
                      f"SPY {macro['spy_dist_200_pct']:+.1f}% vs 200 SMA  VIX {macro['vix']}")
        if macro["regime"] != "Risk-On":
            console.print("  [yellow]→ Not Risk-On: halve position sizes on this run.[/yellow]")
    if theme:
        color = {"Leading": "green", "Emerging": "cyan", "Fading": "yellow"}.get(theme["status"], "red")
        console.print(f"  Theme ({result['benchmark']}): [{color}]{theme['status']}[/]  "
                      f"1mo vs SPY {theme['rs1m']:+.1f}%  3mo vs SPY {theme['rs3m']:+.1f}%  — {theme['note']}")

    counts = pd.Series([r["status"] for r in result["rows"]]).value_counts().to_dict()
    console.print("  Tickers: " + "  ".join(f"{k} {counts.get(k, 0)}" for k in _STATUS_ORDER))

    flagged = [r for r in result["rows"] if r["status"] != "KEEP" or r["reasons"]]
    if flagged:
        tbl = Table(box=box.SIMPLE_HEAVY, show_edge=False)
        for col in ("Ticker", "Status", "RSI", "vs 200", "3mo vs theme", "Why"):
            tbl.add_column(col, no_wrap=col != "Why")
        style = {"REMOVE": "red", "BENCH": "yellow", "WATCH": "cyan", "KEEP": "green"}

        def fmt(v: float | None, spec: str) -> str:
            return "—" if v is None else spec.format(v)

        for r in flagged:
            tbl.add_row(r["ticker"], f"[{style[r['status']]}]{r['status']}[/]",
                        fmt(r["rsi"], "{:.0f}"), fmt(r["vs200"], "{:+.0f}%"),
                        fmt(r["rs3m_vs_theme"], "{:+.0f}%"), r["reasons"])
        console.print(tbl)

    if result["suggestions"]:
        adds = ", ".join(f"{s['ticker']} ({s['rs3m_vs_theme']:+.0f}%)" for s in result["suggestions"])
        console.print(f"  [bold]Trending but missing[/bold] (in benchmark ETFs, beating the theme): {adds}")
    console.print("  [dim]Advisory only — edit the watchlist file to act on this.[/dim]\n")


def save(result: dict) -> str:
    path = OUTPUT_DIR / f"watchlist_check_{result['name']}_{datetime.now():%Y-%m-%d}.csv"
    pd.DataFrame(result["rows"]).to_csv(path, index=False)
    return str(path)


def run_check(name: str, macro: dict | None = None) -> dict | None:
    """check + render + save. Never raises — a failed check must not block a run."""
    try:
        result = check_watchlist(name, macro)
        render(result)
        save(result)
        return result
    except Exception as exc:
        logger.warning(f"Watchlist check failed for '{name}': {exc}")
        return None
