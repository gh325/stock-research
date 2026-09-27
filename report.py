"""
report.py — Save pipeline results to the output/ directory.

Files generated per run:
  top5_picks_YYYY-MM-DD.json      Full machine-readable detail
  top5_picks_YYYY-MM-DD.csv       Spreadsheet summary with notes
  full_rankings_YYYY-MM-DD.csv    All top-50 candidates ranked
  report_YYYY-MM-DD.txt           Human-readable report with legend
"""
import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import OUTPUT_DIR

logger  = logging.getLogger(__name__)
console = Console()

# ── Legend text (shown once at top of every report) ───────────────────────────
LEGEND = """\
TRADING STYLE GUIDE
───────────────────────────────────────────────────────────────────
Day Trade   Buy AND sell within the SAME day. Never hold overnight.
            Needs high daily volume (>1 M shares) and wide price
            swings (ATR). Target = entry + 1.5 × ATR. Stop = entry − 1× ATR.
            ★ rating driven by: volume, ATR %, average daily range.

Swing Trade Hold for 3–20 DAYS to capture a multi-day price move.
            Uses RSI (50–68 ideal), MACD crossover, and resistance
            levels. Target set by Fibonacci extension or next resistance.
            ★ rating driven by: technical score, RSI zone, volume.

Long Hold   Hold for MONTHS to YEARS based on company fundamentals.
            Uses price vs 200-day SMA, P/E ratio, revenue growth,
            and analyst consensus price targets.
            ★ rating driven by: trend (200 SMA), P/E, growth, margins.

OPTIONS QUICK GUIDE
───────────────────────────────────────────────────────────────────
CALL option  Right to BUY 100 shares at the strike price.
             Buy a call when you expect the stock to RISE.
             Profit = intrinsic value at expiry − premium paid.
             Break-even = strike + premium.

PUT option   Right to SELL 100 shares at the strike price.
             Buy a put when you expect the stock to FALL (or as a hedge).
             Break-even = strike − premium.

1 contract  = 100 shares. Cost shown is total per contract.
IV (Implied Volatility): low IV (<30%) = cheap options; high IV (>60%) = expensive.
DTE: Days To Expiration. Shorter DTE = faster time decay (theta burns premium faster).
───────────────────────────────────────────────────────────────────\
"""


def save_results(
    top_picks: list[dict],
    all_ranked: list[dict],
    market_sentiment: dict,
    day_bullish: list[dict] | None = None,
    day_bearish: list[dict] | None = None,
    momentum_continuation: list[dict] | None = None,
    long_term: list[dict] | None = None,
    macro: dict | None = None,
    etf_benchmarks: dict | None = None,
    tag: str | None = None,
) -> dict[str, Path]:
    # tag (the watchlist name) goes after the date so runs on different
    # watchlists on the same day don't overwrite each other's files:
    #   stock_swinglong_2026-09-26_cyber.csv
    today = date.today().isoformat() + (f"_{tag}" if tag else "")
    paths = {
        "json":   _save_json(
            top_picks, market_sentiment, today,
            day_bullish=day_bullish or [],
            day_bearish=day_bearish or [],
            momentum_continuation=momentum_continuation or [],
            long_term=long_term or [],
            macro=macro or {},
            etf_benchmarks=etf_benchmarks or {},
        ),
        "csv":         _save_csv_top5(top_picks, today),
        "full":        _save_csv_full(all_ranked, today),
        "report":      _save_text_report(top_picks, market_sentiment, today),
        "stock_swing": _save_csv_stock_swinglong(top_picks, today),
    }
    if day_bullish:
        paths["stock_intraday"] = _save_csv_stock_intraday(day_bullish, today)
    if day_bullish:
        paths["day_bull_csv"] = _save_csv_picks(
            day_bullish, today, "day_bullish",
            why_label="Why It Ranked High",
            best_label="Best Play",
        )
    if day_bearish:
        paths["day_bear_csv"] = _save_csv_picks(
            day_bearish, today, "day_bearish",
            why_label="Why It's Bearish",
            best_label="Best Play (PUT)",
            target_arrow="↓",
        )
    if momentum_continuation:
        paths["mom_csv"] = _save_csv_picks(
            momentum_continuation, today, "momentum_continuation",
            why_label="Why It Ranked High",
            best_label="Best Play",
        )
    if long_term:
        paths["stock_longterm"] = _save_csv_stock_longterm(long_term, today)
    return paths


def print_summary(
    top_picks: list[dict],
    market_sentiment: dict,
    day_bullish: list[dict] | None = None,
    day_bearish: list[dict] | None = None,
    momentum_continuation: list[dict] | None = None,
    long_term: list[dict] | None = None,
    macro: dict | None = None,
    etf_benchmarks: dict | None = None,
    stocks_only: bool = False,
) -> None:
    """Print macro panel + thematic ETFs + picks tables.

    When stocks_only=True, render ONLY the 3 stock-only tables:
      1. Intraday Stocks  2. Swing/Long Stocks  3. Long-Term Stocks
    Options, momentum, and bearish tables are suppressed.
    """
    from rich.console import Console as _Console
    from rich.panel import Panel as _Panel
    mood  = market_sentiment.get("market_mood", "N/A")
    score = market_sentiment.get("market_sentiment", 50.0)

    wide = _Console(width=320)

    # ── Macro regime panel ───────────────────────────────────────────────────
    if macro and macro.get("available"):
        regime       = macro["regime"]
        regime_color = macro["regime_color"]
        spy_dist     = macro.get("spy_dist_200_pct")
        spy_above    = macro.get("spy_above_200")
        vix          = macro.get("vix")
        dxy_5d       = macro.get("dxy_5d_pct", 0)
        macro_lines = [
            f"[bold]Regime:[/bold] [{regime_color}]{regime}[/{regime_color}]   "
            f"[bold]News mood:[/bold] {mood} ({score:.0f}/100)",
            "",
            f"  [bold]SPY[/bold]   ${macro['spy_close']}   "
            f"vs 200-SMA: [{'green' if spy_above else 'red'}]"
            f"{'+' if spy_dist and spy_dist >= 0 else ''}{spy_dist:.1f}%[/]   "
            f"1d {macro['spy_1d']:+.2f}%   5d {macro['spy_5d']:+.2f}%   "
            f"1mo {macro['spy_1mo']:+.2f}%   3mo {macro['spy_3mo']:+.2f}%   "
            f"YTD {macro['spy_ytd']:+.1f}%",
        ]
        if vix is not None:
            vix_label = "low (calm)" if vix < 15 else "moderate" if vix < 22 else "elevated" if vix < 30 else "high (fear)"
            macro_lines.append(f"  [bold]VIX[/bold]   {vix:.1f}   ({vix_label})")
        if macro.get("dxy") is not None:
            macro_lines.append(f"  [bold]DXY[/bold]   {macro['dxy']:.2f}   5d {dxy_5d:+.2f}%")
        wide.print(_Panel("\n".join(macro_lines), title="[bold]Market Regime[/bold]", border_style="blue"))

    # ── Thematic ETF benchmarks ──────────────────────────────────────────────
    if etf_benchmarks:
        etf_tbl = Table(
            title="Thematic ETF Benchmarks (what you'd buy if you didn't pick stocks)",
            show_lines=False, highlight=True,
        )
        etf_tbl.add_column("ETF", style="bold cyan")
        etf_tbl.add_column("Price",  justify="right")
        etf_tbl.add_column("5d %",   justify="right")
        etf_tbl.add_column("1mo %",  justify="right")
        etf_tbl.add_column("3mo %",  justify="right")
        etf_tbl.add_column("Theme")
        from macro import ETF_THEMES as themes
        for ticker, data in etf_benchmarks.items():
            etf_tbl.add_row(
                ticker,
                f"${data['close']:.2f}",
                f"{data['ret_5d']:+.2f}%",
                f"{data['ret_1mo']:+.2f}%",
                f"{data['ret_3mo']:+.2f}%",
                themes.get(ticker, ""),
            )
        wide.print(etf_tbl)
        wide.print()

    # ── Table 1: Stock-only intraday top 3 ──────────────────────────────────
    if day_bullish:
        wide.print(
            "[bold green]★ Top 3 Intraday Stocks[/bold green] "
            "[dim](stock picks only — buy and sell the same day, no options)[/dim]\n"
        )
        wide.print(_build_stock_intraday_table(day_bullish))
        wide.print()

    # ── Table 2: Options day-trade top 3 (with CALL recommendations) ────────
    if day_bullish and not stocks_only:
        wide.print(
            "[bold green]★ Top 3 Bullish Day-Trade Picks — Options[/bold green] "
            "[dim](same 3 stocks as above, now with CALL option plays)[/dim]\n"
        )
        wide.print(_build_picks_table(
            day_bullish,
            title="Top 3 Bullish Day-Trade Picks (CALL recommendations)",
            why_header="Why It Ranked High",
            best_style="bold green",
            target_arrow="",
        ))
        wide.print()

    # ── Table 3: Stock-only swing / long top 5 ──────────────────────────────
    wide.print(
        "[bold blue]Top 5 Swing / Long Stocks[/bold blue] "
        "[dim](stock picks only — hold days to months, no options)[/dim]\n"
    )
    wide.print(_build_stock_swinglong_table(top_picks))
    wide.print()

    # ── Table 4: Options swing / long top 5 (with CALL recommendations) ─────
    if not stocks_only:
        wide.print(
            "[bold blue]Top 5 Bullish Swing / Long Picks — Options[/bold blue] "
            "[dim](same 5 stocks as above, now with CALL option plays)[/dim]\n"
        )
        wide.print(_build_picks_table(
            top_picks,
            title="Top 5 Bullish Swing / Long Picks (CALL recommendations)",
            why_header="Why It Ranked High",
            best_style="bold green",
            target_arrow="",
        ))
        wide.print()

    # ── Table 5: Long-Term Stocks top 5 (1 year+) ──────────────────────────
    if long_term:
        wide.print(
            "[bold cyan]📈 Top 5 Long-Term Stocks[/bold cyan] "
            "[dim](hold 1 year+ — fundamentals-first, trend-confirmed, no options)[/dim]\n"
        )
        wide.print(_build_stock_longterm_table(long_term))
        wide.print()

    # ── Table 6: Momentum Continuation top 3 (riding established trends) ───
    if momentum_continuation and not stocks_only:
        wide.print(
            "[bold magenta]🚀 Top 3 Momentum Continuation Picks[/bold magenta] "
            "[dim](RSI > 70 AND price above all SMAs — overbought but trending. "
            "Standard scoring penalises these; this list surfaces them deliberately.)[/dim]\n"
        )
        wide.print(_build_picks_table(
            momentum_continuation,
            title="Top 3 Momentum Continuation Picks (CALL recommendations)",
            why_header="Why It Ranked High",
            best_style="bold magenta",
            target_arrow="",
        ))
        wide.print()

    # ── Table 4: Bearish day-trade top 3 ────────────────────────────────────
    if stocks_only:
        return
    if day_bearish:
        wide.print(
            "[bold red]⚠ Top 3 Bearish Day-Trade Picks[/bold red] "
            "[dim](advanced — buying puts profits when stocks FALL. "
            "Higher risk, only consider if you understand the mechanics.)[/dim]\n"
        )
        wide.print(_build_picks_table(
            day_bearish,
            title="Top 3 Bearish Day-Trade Picks (PUT recommendations)",
            why_header="Why It's Bearish",
            best_style="bold red",
            target_arrow=" ↓",
            header_style="bold red",
        ))
    else:
        wide.print(
            "[dim]No qualifying bearish day-trade setups today "
            "(market may be too bullish to surface clean put candidates).[/dim]"
        )


def _build_picks_table(
    picks: list[dict],
    title: str,
    why_header: str,
    best_style: str,
    target_arrow: str = "",
    header_style: str | None = None,
) -> Table:
    """
    Build the picks table — exactly 13 columns, locked layout.

    Order (do NOT modify, rename, or reorder without explicit user approval):
      # | Ticker | Current | Day Target | Swing Target | Long Target |
      Best Play | Why ... | Today vs 20d | 20d vs 60d | Day ★ |
      What This Tells You | Why
    """
    tbl = Table(
        title=title,
        show_lines=True,
        highlight=True,
        header_style=header_style or "bold",
    )
    tbl.add_column("#",            style="bold cyan",   justify="center", no_wrap=True)
    tbl.add_column("Ticker",       style="bold yellow", no_wrap=True)
    tbl.add_column("Current",      justify="right",     no_wrap=True)
    tbl.add_column(f"Day Target{target_arrow}",   justify="right", no_wrap=True)
    tbl.add_column(f"Swing Target{target_arrow}", justify="right", no_wrap=True)
    tbl.add_column(f"Long Target{target_arrow}",  justify="right", no_wrap=True)
    tbl.add_column("Best Play (affordable)", style=best_style, no_wrap=True)
    tbl.add_column(why_header, no_wrap=True)
    tbl.add_column("Today vs 20d", justify="right", no_wrap=True)
    tbl.add_column("20d vs 60d",   justify="right", no_wrap=True)
    tbl.add_column("Day ★",        justify="center", no_wrap=True)
    tbl.add_column("What This Tells You", no_wrap=True)
    tbl.add_column("Why", no_wrap=True)   # option-level reasoning

    for p in picks:
        t  = p.get("targets", {})
        bo = p.get("best_option", {})
        ts = p.get("trade_styles", {})

        if bo.get("available"):
            tf_short = "swing" if "Swing" in bo.get("timeframe", "") else "med"
            try:
                exp_short = datetime.strptime(bo["expiry"], "%Y-%m-%d").strftime("%m/%d")
            except (ValueError, KeyError, TypeError):
                exp_short = bo.get("expiry", "")
            play = (
                f"{bo['side']} ${bo['strike']:.0f} ({tf_short})  "
                f"${bo['contract_cost']:.0f}/c  BE ${bo['break_even']:.2f}  "
                f"exp {exp_short}"
                f"{p.get('earnings_flag', '')}"   # appended ' ⚠ EPS MM/DD' if relevant
            )
            option_why = bo.get("reason", "—")
        else:
            play = "—"
            option_why = "—"

        # Position-sizing recommendation appended to "Why" column
        pos_size = p.get("position_size", "")
        if pos_size and option_why and option_why != "—":
            option_why = f"{option_why}  |  {pos_size}"
        elif pos_size:
            option_why = pos_size

        why_raw   = p.get("why_ranked", "—")
        why_parts = [w.strip() for w in why_raw.split(",") if w.strip()]
        why       = ", ".join(why_parts[:3]) + ("…" if len(why_parts) > 3 else "")

        tbl.add_row(
            str(p.get("rank", "")),
            p["ticker"],
            f"${p.get('last_price', 0):.2f}",
            f"${t.get('day_target', 0):.2f}"   if t else "—",
            f"${t.get('swing_target', 0):.2f}" if t else "—",
            f"${t.get('long_target', 0):.2f}"  if t else "—",
            play,
            why,
            f"{ts.get('vol_ratio_day', 1.0):.2f}×",
            f"{ts.get('vol_ratio_long', 1.0):.2f}×",
            ts.get("day_stars", "—"),
            p.get("volume_insight", "—"),
            option_why,
        )

    return tbl


def _build_stock_intraday_table(picks: list[dict]) -> Table:
    """
    13-column stock-only intraday table. No options columns.
    # | Ticker | Current | Day Target ↑ | Day Stop ↓ | ATR% | RSI |
    Today vs 20d | Day ★ | 5d Return | Sector | What This Tells You | Why It's a Day-Trade
    """
    tbl = Table(
        title="Top 3 Intraday Stocks  [dim](buy and sell same day — no options)[/dim]",
        show_lines=True,
        highlight=True,
        header_style="bold green",
    )
    tbl.add_column("#",                    style="bold cyan",   justify="center", no_wrap=True)
    tbl.add_column("Ticker",               style="bold yellow", no_wrap=True)
    tbl.add_column("Current",              justify="right",     no_wrap=True)
    tbl.add_column("Day Target ↑",         justify="right",     no_wrap=True)
    tbl.add_column("Day Stop ↓",           justify="right",     no_wrap=True)
    tbl.add_column("ATR%",                 justify="right",     no_wrap=True)
    tbl.add_column("RSI",                  justify="right",     no_wrap=True)
    tbl.add_column("Today vs 20d",         justify="right",     no_wrap=True)
    tbl.add_column("Day ★",                justify="center",    no_wrap=True)
    tbl.add_column("5d Return",            justify="right",     no_wrap=True)
    tbl.add_column("Sector",               no_wrap=True)
    tbl.add_column("What This Tells You",  no_wrap=True)
    tbl.add_column("Why It's a Day-Trade", no_wrap=True)

    for p in picks:
        t    = p.get("targets", {})
        ts   = p.get("trade_styles", {})
        fund = p.get("fundamentals", {})

        atr_pct   = (t.get("atr_pct") or p.get("atr_pct") or 0)
        rsi       = p.get("rsi", 0)
        ret5d     = p.get("ret5d_pct", 0)
        sector    = (fund.get("sector") or "—")[:20]
        day_stop  = t.get("day_stop", 0)
        ret_color = "green" if ret5d >= 0 else "red"

        why_raw   = p.get("why_ranked", "—")
        why_parts = [w.strip() for w in why_raw.split(",") if w.strip()]
        why       = ", ".join(why_parts[:3]) + ("…" if len(why_parts) > 3 else "")

        tbl.add_row(
            str(p.get("rank", "")),
            p["ticker"],
            f"${p.get('last_price', 0):.2f}",
            f"${t.get('day_target', 0):.2f}" if t else "—",
            f"${day_stop:.2f}"               if day_stop else "—",
            f"{atr_pct:.1f}%",
            f"{rsi:.0f}",
            f"{ts.get('vol_ratio_day', 1.0):.2f}×",
            ts.get("day_stars", "—"),
            f"[{ret_color}]{ret5d:+.1f}%[/{ret_color}]",
            sector,
            p.get("volume_insight", "—"),
            why,
        )

    return tbl


def _build_stock_swinglong_table(picks: list[dict]) -> Table:
    """
    13-column stock-only swing/long table. No options columns.
    # | Ticker | Current | Swing Target | Long Target | Swing Stop |
    Analyst Target | Today vs 20d | 20d vs 60d | Swing ★ | Sector | P/E | Why It Ranked High
    """
    tbl = Table(
        title="Top 5 Swing / Long Stocks  [dim](hold days to months — no options)[/dim]",
        show_lines=True,
        highlight=True,
        header_style="bold blue",
    )
    tbl.add_column("#",               style="bold cyan",   justify="center", no_wrap=True)
    tbl.add_column("Ticker",          style="bold yellow", no_wrap=True)
    tbl.add_column("Current",         justify="right",     no_wrap=True)
    tbl.add_column("Swing Target",    justify="right",     no_wrap=True)
    tbl.add_column("Long Target",     justify="right",     no_wrap=True)
    tbl.add_column("Swing Stop",      justify="right",     no_wrap=True)
    tbl.add_column("Analyst Target",  justify="right",     no_wrap=True)
    tbl.add_column("Today vs 20d",    justify="right",     no_wrap=True)
    tbl.add_column("20d vs 60d",      justify="right",     no_wrap=True)
    tbl.add_column("Swing ★",         justify="center",    no_wrap=True)
    tbl.add_column("Sector",          no_wrap=True)
    tbl.add_column("P/E",             justify="right",     no_wrap=True)
    tbl.add_column("Why It Ranked High", no_wrap=True)

    for p in picks:
        t    = p.get("targets", {})
        ts   = p.get("trade_styles", {})
        fund = p.get("fundamentals", {})

        pe         = fund.get("forwardPE") or fund.get("trailingPE")
        pe_s       = f"{float(pe):.1f}" if pe else "—"
        sector     = (fund.get("sector") or "—")[:22]
        analyst    = t.get("analyst_mean")
        analyst_s  = f"${analyst}" if analyst else "—"
        swing_stop = t.get("swing_stop", 0)

        why_raw   = p.get("why_ranked", "—")
        why_parts = [w.strip() for w in why_raw.split(",") if w.strip()]
        why       = ", ".join(why_parts[:3]) + ("…" if len(why_parts) > 3 else "")

        earnings_tag = p.get("earnings_flag", "")
        ticker_cell  = f"{p['ticker']}{earnings_tag}" if earnings_tag else p["ticker"]

        tbl.add_row(
            str(p.get("rank", "")),
            ticker_cell,
            f"${p.get('last_price', 0):.2f}",
            f"${t.get('swing_target', 0):.2f}" if t else "—",
            f"${t.get('long_target', 0):.2f}"  if t else "—",
            f"${swing_stop:.2f}"               if swing_stop else "—",
            analyst_s,
            f"{ts.get('vol_ratio_day', 1.0):.2f}×",
            f"{ts.get('vol_ratio_long', 1.0):.2f}×",
            ts.get("swing_stars", "—"),
            sector,
            pe_s,
            why,
        )

    return tbl


def _build_stock_longterm_table(picks: list[dict]) -> Table:
    """
    13-column stock-only long-term table (1 year+ hold). No options columns.
    # | Ticker | Current | 1Y Target | Upside | Rev Growth | Fwd P/E |
    Profit Margin | vs 200 SMA | Accum 20d/60d | Sector | Long ★ | Why Long Term
    """
    tbl = Table(
        title="Top 5 Long-Term Stocks  [dim](hold 1 year+ — fundamentals-first, trend-confirmed)[/dim]",
        show_lines=True,
        highlight=True,
        header_style="bold cyan",
    )
    tbl.add_column("#",             style="bold cyan",   justify="center", no_wrap=True)
    tbl.add_column("Ticker",        style="bold yellow", no_wrap=True)
    tbl.add_column("Current",       justify="right",     no_wrap=True)
    tbl.add_column("1Y Target",     justify="right",     no_wrap=True)
    tbl.add_column("Target Src",    no_wrap=True)
    tbl.add_column("Upside",        justify="right",     no_wrap=True)
    tbl.add_column("Rev Growth",    justify="right",     no_wrap=True)
    tbl.add_column("Fwd P/E",       justify="right",     no_wrap=True)
    tbl.add_column("Profit Margin", justify="right",     no_wrap=True)
    tbl.add_column("vs 200 SMA",    justify="right",     no_wrap=True)
    tbl.add_column("Accum 20d/60d", justify="right",     no_wrap=True)
    tbl.add_column("Sector",        no_wrap=True)
    tbl.add_column("Long ★",        justify="center",    no_wrap=True)
    tbl.add_column("Why Long Term", no_wrap=True)

    for p in picks:
        t    = p.get("targets", {})
        ts   = p.get("trade_styles", {})
        fund = p.get("fundamentals", {})

        # Price targets
        long_target     = t.get("long_target", 0)
        long_target_pct = t.get("long_target_pct", 0)
        upside_color    = "green" if long_target_pct >= 0 else "red"

        # Fundamentals
        rev_growth  = fund.get("revenueGrowth")
        rev_s       = f"{float(rev_growth)*100:+.0f}%" if rev_growth else "—"
        rev_color   = "green" if rev_growth and float(rev_growth) > 0 else "red"

        pe          = fund.get("forwardPE") or fund.get("trailingPE")
        pe_s        = f"{float(pe):.1f}" if pe else "—"

        margin      = fund.get("profitMargins")
        margin_s    = f"{float(margin)*100:.1f}%" if margin else "—"
        margin_color = "green" if margin and float(margin) > 0.10 else ("yellow" if margin and float(margin) > 0 else "red")

        # vs 200 SMA
        sma200_pct  = p.get("sma200_pct", 0.0)
        sma_color   = "green" if sma200_pct >= 0 else "red"
        sma_s       = f"{sma200_pct:+.1f}%"

        # Accumulation
        vol_l       = ts.get("vol_ratio_long", 1.0)
        accum_color = "green" if vol_l >= 1.10 else ("dim" if vol_l >= 0.90 else "red")

        sector      = (fund.get("sector") or "—")[:22]

        # Why — use long_note for the most relevant long-term signal
        long_note   = ts.get("long_note", "")
        if not long_note:
            why_raw   = p.get("why_ranked", "—")
            why_parts = [w.strip() for w in why_raw.split(",") if w.strip()]
            long_note = ", ".join(why_parts[:3]) + ("…" if len(why_parts) > 3 else "")

        earnings_tag = p.get("earnings_flag", "")
        ticker_cell  = f"{p['ticker']}{earnings_tag}" if earnings_tag else p["ticker"]

        tbl.add_row(
            str(p.get("rank", "")),
            ticker_cell,
            f"${p.get('last_price', 0):.2f}",
            f"${long_target:.2f}"                          if long_target else "—",
            _target_src_cell(t.get("long_target_basis", "")),
            f"[{upside_color}]{long_target_pct:+.1f}%[/{upside_color}]",
            f"[{rev_color}]{rev_s}[/{rev_color}]",
            pe_s,
            f"[{margin_color}]{margin_s}[/{margin_color}]",
            f"[{sma_color}]{sma_s}[/{sma_color}]",
            f"[{accum_color}]{vol_l:.2f}×[/{accum_color}]",
            sector,
            ts.get("long_stars", "—"),
            long_note[:80] + ("…" if len(long_note) > 80 else ""),
        )

    return tbl


def _target_src_cell(basis: str) -> str:
    """Green for an analyst-backed target; yellow when the tool projected it."""
    if not basis:
        return "—"
    return f"[green]{basis}[/green]" if basis == "Analyst" else f"[yellow]{basis}[/yellow]"


def _print_stock_panel(p: dict) -> None:
    ticker  = p["ticker"]
    price   = p.get("last_price", 0)
    t       = p.get("targets", {})
    st      = p.get("trade_styles", {})
    bo      = p.get("best_option", {})
    fund    = p.get("fundamentals", {})
    notes   = p.get("notes", "")

    lines: list[str] = []

    # Fundamentals header
    sector  = fund.get("sector") or "N/A"
    rec     = (fund.get("recommendationKey") or "").replace("_", " ").title()
    pe      = fund.get("forwardPE") or fund.get("trailingPE")
    pe_s    = f"P/E {pe:.1f}" if pe else "P/E N/A"
    lines.append(f"Sector: {sector}  |  Analyst: {rec or 'N/A'}  |  {pe_s}")

    # Notes (plain English)
    if notes:
        lines += ["", "Notes:", *[f"  {s.strip()}" for s in notes.split("  ") if s.strip()]]

    # Trade style suitability
    if st:
        lines += [
            "",
            "Trade Style Suitability:",
            f"  Day Trade  {st.get('day_stars','—')}  {st.get('day_label','—'):10s}  {st.get('day_note','')}",
            f"  Swing      {st.get('swing_stars','—')}  {st.get('swing_label','—'):10s}  {st.get('swing_note','')}",
            f"  Long Hold  {st.get('long_stars','—')}  {st.get('long_label','—'):10s}  {st.get('long_note','')}",
        ]

    # Price targets
    if t:
        analyst_s = ""
        if t.get("analyst_mean"):
            analyst_s = (
                f"  [Analyst range ${t.get('analyst_low','?')} – "
                f"${t.get('analyst_high','?')}, mean ${t.get('analyst_mean')}]"
            )
        lines += [
            "",
            f"Price Targets  (ATR = ${t.get('atr',0):.2f} = {t.get('atr_pct',0):.1f}% daily avg move):",
            f"  Current      ${price:.2f}",
            f"  Day    ↑     ${t.get('day_target',0):.2f}  "
            f"({t.get('day_target_pct',0):+.1f}%)   "
            f"Stop ↓ ${t.get('day_stop',0):.2f}  R:R {t.get('day_rr','1:1.5')}",
            f"  Swing  ↑     ${t.get('swing_target',0):.2f}  "
            f"({t.get('swing_target_pct',0):+.1f}%)   "
            f"Stop ↓ ${t.get('swing_stop',0):.2f}",
            f"  Long   ↑     ${t.get('long_target',0):.2f}  "
            f"({t.get('long_target_pct',0):+.1f}%)"
            f"  [{t.get('long_target_src','')}]{analyst_s}",
        ]

    # Best options play — ONE recommendation per stock
    lines += ["", "─── Best Options Play ───"]
    if bo.get("available"):
        itm_tag = "[ITM]" if bo.get("in_the_money") else "[OTM]"
        lines += [
            f"  {bo['side']} ${bo['strike']:.0f} {itm_tag}  "
            f"{bo['timeframe']}  |  Expiry {bo['expiry']} ({bo['dte']} days)",
            f"  Premium  ${bo['premium']:.2f}/share  →  ${bo['contract_cost']:.0f} per contract (= 100 shares)",
            f"  Bid/Ask  ${bo['bid']:.2f} / ${bo['ask']:.2f}   "
            f"IV {bo['iv_pct']:.0f}%  ({bo['iv_label']})   "
            f"Vol {bo.get('volume',0):,}  OI {bo.get('open_interest',0):,}",
            f"  Break-even  ${bo['break_even']:.2f}  ({bo['pct_to_breakeven']:+.1f}% from current price)",
            f"  Target      ${bo['target_price']:.2f}  →  Est. gain {bo.get('est_profit_pct',0):+.0f}% on premium",
            f"  {bo['payoff_note']}",
            f"  Why this play: {bo['reason']}.",
        ]
    else:
        lines.append(f"  Not available: {bo.get('reason','')}")

    body = "\n".join(lines)
    console.print(Panel(
        body,
        title=f"[bold yellow]{ticker}[/bold yellow]  ${price:.2f}  "
              f"| Score {p.get('combined_score',0):.1f} "
              f"| Rank #{p.get('rank','')}",
        border_style="blue",
        padding=(0, 1),
    ))


# ── File writers ──────────────────────────────────────────────────────────────

def _save_json(
    picks: list[dict],
    market: dict,
    today: str,
    day_bullish: list[dict] | None = None,
    day_bearish: list[dict] | None = None,
    momentum_continuation: list[dict] | None = None,
    long_term: list[dict] | None = None,
    macro: dict | None = None,
    etf_benchmarks: dict | None = None,
) -> Path:
    path = OUTPUT_DIR / f"top5_picks_{today}.json"
    path.write_text(
        json.dumps(
            {
                "run_date":              today,
                "market_sentiment":      market,
                "macro":                 macro or {},
                "etf_benchmarks":        etf_benchmarks or {},
                "swing_long_picks":      picks,
                "day_trade_bullish":     day_bullish or [],
                "day_trade_bearish":     day_bearish or [],
                "momentum_continuation": momentum_continuation or [],
                "long_term_picks":       long_term or [],
            },
            indent=2, default=str,
        )
    )
    return path


def _save_csv_stock_intraday(picks: list[dict], today: str) -> Path:
    """13-column stock-only CSV for the intraday table."""
    path = OUTPUT_DIR / f"stock_intraday_{today}.csv"
    rows = []
    for p in picks:
        t    = p.get("targets", {})
        ts   = p.get("trade_styles", {})
        fund = p.get("fundamentals", {})
        atr_pct = t.get("atr_pct") or p.get("atr_pct") or 0
        rows.append({
            "#":                    p.get("rank", ""),
            "Ticker":               p["ticker"],
            "Current ($)":          p.get("last_price", ""),
            "Day Target ($)":       t.get("day_target", ""),
            "Day Stop ($)":         t.get("day_stop", ""),
            "ATR%":                 f"{atr_pct:.1f}",
            "RSI":                  f"{p.get('rsi', 0):.0f}",
            "Today vs 20d":         ts.get("vol_ratio_day", ""),
            "Day Stars":            ts.get("day_stars", ""),
            "5d Return (%)":        f"{p.get('ret5d_pct', 0):+.1f}",
            "Sector":               fund.get("sector") or "",
            "What This Tells You":  p.get("volume_insight", ""),
            "Why It's a Day-Trade": p.get("why_ranked", ""),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _save_csv_stock_swinglong(picks: list[dict], today: str) -> Path:
    """13-column stock-only CSV for the swing/long table."""
    path = OUTPUT_DIR / f"stock_swinglong_{today}.csv"
    rows = []
    for p in picks:
        t    = p.get("targets", {})
        ts   = p.get("trade_styles", {})
        fund = p.get("fundamentals", {})
        pe   = fund.get("forwardPE") or fund.get("trailingPE")
        rows.append({
            "#":                  p.get("rank", ""),
            "Ticker":             p["ticker"],
            "Current ($)":        p.get("last_price", ""),
            "Swing Target ($)":   t.get("swing_target", ""),
            "Long Target ($)":    t.get("long_target", ""),
            "Swing Stop ($)":     t.get("swing_stop", ""),
            "Analyst Target ($)": t.get("analyst_mean", ""),
            "Today vs 20d":       ts.get("vol_ratio_day", ""),
            "20d vs 60d":         ts.get("vol_ratio_long", ""),
            "Swing Stars":        ts.get("swing_stars", ""),
            "Sector":             fund.get("sector") or "",
            "P/E":                f"{float(pe):.1f}" if pe else "",
            "Why It Ranked High": p.get("why_ranked", ""),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _save_csv_stock_longterm(picks: list[dict], today: str) -> Path:
    """13-column stock-only CSV for the long-term table."""
    path = OUTPUT_DIR / f"stock_longterm_{today}.csv"
    rows = []
    for p in picks:
        t    = p.get("targets", {})
        ts   = p.get("trade_styles", {})
        fund = p.get("fundamentals", {})
        pe      = fund.get("forwardPE") or fund.get("trailingPE")
        margin  = fund.get("profitMargins")
        rev_gr  = fund.get("revenueGrowth")
        rows.append({
            "#":                  p.get("rank", ""),
            "Ticker":             p["ticker"],
            "Current ($)":        p.get("last_price", ""),
            "1Y Target ($)":      t.get("long_target", ""),
            "Target Src":         t.get("long_target_basis", ""),
            "Upside (%)":         f"{t.get('long_target_pct', 0):+.1f}",
            "Rev Growth (%)":     f"{float(rev_gr)*100:+.0f}" if rev_gr else "",
            "Fwd P/E":            f"{float(pe):.1f}" if pe else "",
            "Profit Margin (%)":  f"{float(margin)*100:.1f}" if margin else "",
            "vs 200 SMA (%)":     f"{p.get('sma200_pct', 0):+.1f}",
            "Accum 20d/60d":      f"{ts.get('vol_ratio_long', 1.0):.2f}",
            "Sector":             fund.get("sector") or "",
            "Long Stars":         ts.get("long_stars", ""),
            "Why Long Term":      ts.get("long_note", p.get("why_ranked", "")),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _save_csv_picks(
    picks: list[dict],
    today: str,
    name: str,
    why_label: str = "Why It Ranked High",
    best_label: str = "Best Play",
    target_arrow: str = "",
) -> Path:
    """Generic 12-column CSV writer used for day-trade lists."""
    path = OUTPUT_DIR / f"{name}_{today}.csv"
    rows = []
    for i, p in enumerate(picks):
        t  = p.get("targets", {})
        bo = p.get("best_option", {})
        ts = p.get("trade_styles", {})
        best_play = (
            f"{bo['side']} ${bo['strike']:.0f} exp {bo['expiry']} "
            f"${bo['contract_cost']:.0f}/contract break-even ${bo['break_even']:.2f}"
            if bo.get("available") else ""
        )
        rows.append({
            "#":                       p.get("rank", i + 1),
            "Ticker":                  p["ticker"],
            "Current ($)":             p.get("last_price", ""),
            f"Day Target{target_arrow} ($)":   t.get("day_target", ""),
            f"Swing Target{target_arrow} ($)": t.get("swing_target", ""),
            f"Long Target{target_arrow} ($)":  t.get("long_target", ""),
            best_label:                best_play,
            why_label:                 p.get("why_ranked", ""),
            "Today vs 20d":            ts.get("vol_ratio_day", ""),
            "20d vs 60d":              ts.get("vol_ratio_long", ""),
            "Day Stars":               ts.get("day_stars", ""),
            "What This Tells You":     p.get("volume_insight", ""),
            "Why":                     bo.get("reason", "") if bo.get("available") else "",
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _save_csv_top5(picks: list[dict], today: str) -> Path:
    path = OUTPUT_DIR / f"top5_picks_{today}.csv"
    rows = []
    for i, p in enumerate(picks):
        t  = p.get("targets", {})
        bo = p.get("best_option", {})
        ts = p.get("trade_styles", {})
        best_play = (
            f"{bo['side']} ${bo['strike']:.0f} exp {bo['expiry']} "
            f"${bo['contract_cost']:.0f}/contract break-even ${bo['break_even']:.2f}"
            if bo.get("available") else ""
        )
        rows.append({
            "#":                  p.get("rank", i + 1),
            "Ticker":             p["ticker"],
            "Current ($)":        p.get("last_price", ""),
            "Day Target ($)":     t.get("day_target", ""),
            "Swing Target ($)":   t.get("swing_target", ""),
            "Long Target ($)":    t.get("long_target", ""),
            "Best Play":          best_play,
            "Why It Ranked High":  p.get("why_ranked", ""),
            "Today vs 20d":        ts.get("vol_ratio_day", ""),
            "20d vs 60d":          ts.get("vol_ratio_long", ""),
            "Day Stars":           ts.get("day_stars", ""),
            "What This Tells You": p.get("volume_insight", ""),
            "Why":                 bo.get("reason", "") if bo.get("available") else "",
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _save_csv_full(ranked: list[dict], today: str) -> Path:
    path = OUTPUT_DIR / f"full_rankings_{today}.csv"
    drop = ["sample_headlines", "targets", "trade_styles", "options",
            "fundamentals", "best_option", "notes"]
    df = pd.DataFrame(ranked)
    if not df.empty:
        df.insert(0, "Rank", range(1, len(df) + 1))
        df = df.drop(columns=drop, errors="ignore")
    df.to_csv(path, index=False)
    return path


def _save_text_report(picks: list[dict], market: dict, today: str) -> Path:
    path = OUTPUT_DIR / f"report_{today}.txt"
    W = 72
    lines = [
        LEGEND,
        "",
        "=" * W,
        f"  STOCK RESEARCH REPORT  —  {today}",
        "=" * W,
        "",
        f"  Market Mood  : {market.get('market_mood','N/A')}",
        f"  Sentiment    : {market.get('market_sentiment',50.0):.1f}/100  "
        f"({market.get('article_count',0)} articles)",
        "",
        "  TOP 5 PAPER TRADE PICKS",
        "  " + "-" * (W - 2),
    ]

    for p in picks:
        t   = p.get("targets", {})
        st  = p.get("trade_styles", {})
        bo  = p.get("best_option", {})
        f   = p.get("fundamentals", {})
        nts = p.get("notes", "")
        px  = p.get("last_price", 0)

        lines += [
            "",
            f"  #{p.get('rank','')}  {p['ticker']}  —  ${px:.2f}  "
            f"[{f.get('sector','N/A')}]",
            f"      Score : {p.get('combined_score',0):.1f}  |  "
            f"Technical {p.get('technical_score',0):.1f}  |  "
            f"Sentiment {p.get('sentiment_score',0):.1f}",
        ]

        # Notes
        if nts:
            lines += ["", "      Notes:"]
            for sentence in nts.split("  "):
                if sentence.strip():
                    lines.append(f"        {sentence.strip()}")

        # Trade styles
        if st:
            lines += [
                "",
                "      Trade Style Suitability:",
                f"        Day Trade  {st.get('day_stars','—')}  "
                f"{st.get('day_label','')}: {st.get('day_note','')}",
                f"        Swing      {st.get('swing_stars','—')}  "
                f"{st.get('swing_label','')}: {st.get('swing_note','')}",
                f"        Long Hold  {st.get('long_stars','—')}  "
                f"{st.get('long_label','')}: {st.get('long_note','')}",
            ]

        # Targets
        if t:
            an_s = ""
            if t.get("analyst_mean"):
                an_s = f"  [analyst mean ${t['analyst_mean']}]"
            lines += [
                "",
                f"      Price Targets  (ATR ${t.get('atr',0):.2f} = {t.get('atr_pct',0):.1f}%):",
                f"        Day   ↑ ${t.get('day_target',0):.2f} "
                f"({t.get('day_target_pct',0):+.1f}%)  "
                f"Stop ${t.get('day_stop',0):.2f}  R:R {t.get('day_rr','1:1.5')}",
                f"        Swing ↑ ${t.get('swing_target',0):.2f} "
                f"({t.get('swing_target_pct',0):+.1f}%)  "
                f"Stop ${t.get('swing_stop',0):.2f}",
                f"        Long  ↑ ${t.get('long_target',0):.2f} "
                f"({t.get('long_target_pct',0):+.1f}%){an_s}",
            ]

        # Best options play
        lines += ["", "      Best Options Play:"]
        if bo.get("available"):
            itm = "[ITM]" if bo.get("in_the_money") else "[OTM]"
            lines += [
                f"        {bo['side']} ${bo['strike']:.0f} {itm}  "
                f"{bo['timeframe']}  Expiry {bo['expiry']} ({bo['dte']}d)",
                f"        Premium ${bo['premium']:.2f}/share = "
                f"${bo['contract_cost']:.0f}/contract  "
                f"IV={bo['iv_pct']:.0f}% ({bo['iv_label']})",
                f"        Break-even ${bo['break_even']:.2f} "
                f"({bo['pct_to_breakeven']:+.1f}%)  "
                f"Est gain {bo.get('est_profit_pct',0):+.0f}% at target",
                f"        {bo['payoff_note']}",
                f"        Why: {bo['reason']}.",
            ]
        else:
            lines.append(f"        Not available — {bo.get('reason','')}")

        lines.append("  " + "-" * (W - 2))

    lines += [
        "",
        "=" * W,
        f"  Generated by stock-research  |  {today}",
        "  For academic / paper trading use only. Not financial advice.",
        "",
    ]

    path.write_text("\n".join(lines))
    return path
