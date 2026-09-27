"""
main.py — CLI entry point for the stock research pipeline.

Usage:
  python main.py run                                 Run the full pipeline (6 tables, cached)
  python main.py run --no-cache                      Force re-download all data
  python main.py run --watchlist ai_ecosystem        Add watchlist to universe
  python main.py run --watchlist ai_ecosystem --watchlist-only
                                                     Analyze ONLY watchlist tickers
  python main.py run --stocks-only                   Render ONLY 3 stock tables
                                                     (intraday + swing/long + long-term)
                                                     Suppresses options/momentum/bearish

  RECOMMENDED DAILY:
  python main.py run --no-cache --watchlist ai_ecosystem --watchlist-only --stocks-only

  python main.py evolve --watchlist ai_ecosystem     Self-improving loop: try weight changes,
                                                     keep only holdout-tested ones (--apply)
  python main.py journal backfill|grade|report       Grade the sector scan's predictions
  python main.py experiment                          Test structural score changes (H1–H3)
  python main.py sectors                             Sector rotation scan (runs before
                                                     every run; skip with --skip-sector-scan)
  python main.py check-watchlist cyber               Health-check a watchlist (runs
                                                     automatically before --watchlist
                                                     runs; skip with --skip-watchlist-check)
  python main.py schedule                            Start nightly scheduler (9 PM ET)
  python main.py monitor                             Watch top picks intraday and alert
  python main.py monitor --interval 60               Custom poll interval (seconds)
  python main.py status                              Show the most recent report
  python main.py track buy  AAPL --price 182         Log a paper buy
  python main.py track sell AAPL --price 195         Log a paper sell
  python main.py positions                           Show open paper positions
  python main.py history                             Show all past recommendations
  python main.py backtest                            Validate past picks vs SPY (5d forward)
  python main.py backtest --days 10 --since 60       Custom backtest windows
"""
import argparse
import logging
import sys
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from config import OUTPUT_DIR, TOP_CANDIDATES

# ── File-descriptor headroom (macOS Python soft limit can be as low as 256) ──
# yfinance opens many short-lived HTTP connections during batch downloads.
# Raise the soft limit so we don't hit "Too many open files" on long runs.
try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _target = min(8192, _hard if _hard != resource.RLIM_INFINITY else 8192)
    if _soft < _target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_target, _hard))
except (ImportError, ValueError, OSError):
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger  = logging.getLogger(__name__)
console = Console()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    force_refresh: bool = False,
    watchlist: str | None = None,
    watchlist_only: bool = False,
    stocks_only: bool = False,
    check_watchlist: bool = True,
    sector_scan: bool = True,
) -> list[dict]:
    """
    Full pipeline: ingest → technicals → sentiment → score → report → save.
    Returns the top 5 picks.

    watchlist:       name of file in watchlists/ to add to the universe
    watchlist_only:  if True, analyze ONLY the watchlist (skip the full universe)
    check_watchlist: if True and a watchlist is given, run the advisory
                     pre-run health check (watchlist_check.py) first
    sector_scan:     if True, run the sector rotation scan (sector_scan.py)
                     first, in every mode
    """
    import ingest
    import macro as macro_mod
    import options as opts
    import report
    import scorer
    import sentiment
    import targets as tgts
    import technicals
    import tracker

    if force_refresh:
        _clear_caches()

    console.rule("[bold blue]Stock Research Pipeline")

    # 0a. Sector rotation scan — where is the money going before we pick stocks?
    if sector_scan:
        import sector_scan as sector_scan_mod
        sector_scan_mod.run_scan()

    # 0b. Advisory watchlist health check — flags stale, broken or event-risk
    #    tickers and trending names the list is missing. Never blocks the run.
    if watchlist and check_watchlist:
        import watchlist_check
        watchlist_check.run_check(watchlist)

    # 1. Fetch ticker universe (with optional watchlist injection)
    console.print("[bold][1/5][/bold] Fetching ticker universe …")
    if watchlist_only and watchlist:
        tickers = ingest.load_watchlist(watchlist)
        console.print(f"      [yellow]Watchlist-only mode:[/yellow] {len(tickers)} tickers from '{watchlist}'")
        # Watchlist-only changes price-cache shape, so force a fresh fetch
        if not force_refresh:
            _clear_caches()
            console.print("      [dim]Cleared cache to fetch fresh data for watchlist[/dim]")
    else:
        tickers = ingest.get_ticker_universe()
        if watchlist:
            wl = ingest.load_watchlist(watchlist)
            added = [t for t in wl if t not in tickers]
            tickers = tickers + added
            console.print(f"      Watchlist '{watchlist}' added {len(added)} new tickers (total {len(tickers):,})")
        else:
            console.print(f"      {len(tickers):,} US common stocks found")

    # 2. Download price data (batched, cached)
    console.print("[bold][2/5][/bold] Downloading price data …")
    console.print("      (First run: 15–25 min. Subsequent runs use cache.)")
    price_data = ingest.fetch_price_data(tickers)
    console.print(f"      {len(price_data):,} stocks passed quality filters")

    # 3. Technical analysis on all stocks
    console.print("[bold][3/5][/bold] Calculating technical indicators …")
    tech_scores = technicals.score_all_technicals(price_data)
    console.print(f"      Scored {len(tech_scores):,} stocks")

    # 4. Narrow to top candidates, fetch news, score sentiment
    console.print(f"[bold][4/5][/bold] Fetching news for top {TOP_CANDIDATES} candidates …")
    candidates  = scorer.get_top_candidates(tech_scores, n=TOP_CANDIDATES)
    ticker_news = ingest.fetch_news_for_tickers(candidates)
    sent_scores = sentiment.score_sentiment_for_tickers(ticker_news)
    market_sent = sentiment.fetch_market_rss_sentiment()
    # yfinance / feedparser leave urllib3 pools alive — force a sweep before
    # the heavy enrichment phase that opens more connections per pick.
    import gc as _gc
    _gc.collect()
    console.print(f"      Market mood: [bold]{market_sent.get('market_mood', 'N/A')}[/bold]")

    # 5. Final scoring and ranking
    console.print("[bold][5/5][/bold] Scoring and ranking …")
    tech_candidates = {t: tech_scores[t] for t in candidates if t in tech_scores}
    all_ranked      = scorer.final_ranking(tech_candidates, sent_scores)

    # 6. Four bullish pick lists from the same ranking pass + bearish later:
    #    - Top 5 bullish swing/long      (combined_score, max 2 per sector)
    #    - Top 3 bullish day-trade       (60% day_score + 30% tech + 10% sent)
    #    - Top 3 momentum continuation   (RSI > 70 + above all SMAs)
    #    - Top 3 bearish day-trade       (separate bearish ranking — built below)
    #
    # Sector concentration cap: enforces max-2-per-sector on the swing/long top 5.
    # Lazily fetches sector info as needed (typically 5–10 fundamental fetches).
    # A single-sector watchlist (`# sector_cap: off`) would only ever yield
    # 2 picks under the cap, so it is lifted there — the theme is the choice.
    def _sector_for(t):
        f = ingest.fetch_fundamentals(t)
        return f.get("sector") or "Unknown"

    sector_cap: int | None = 2
    if watchlist_only and watchlist and \
            (ingest.load_watchlist_header(watchlist, "sector_cap") or "").lower() == "off":
        sector_cap = None
        console.print(f"      [yellow]Sector cap off for '{watchlist}'[/yellow] — "
                      "single-theme list: all picks are one factor bet, size accordingly")

    swing_long_picks = scorer.select_top_picks(
        all_ranked, max_per_sector=sector_cap, sector_fetcher=_sector_for,
    )
    day_bullish_picks = scorer.select_top_day_bullish(all_ranked, n=3)
    momentum_continuation_picks = scorer.select_momentum_continuation_picks(all_ranked, n=3)
    long_term_picks = scorer.select_long_term_picks(
        all_ranked, max_per_sector=sector_cap, sector_fetcher=_sector_for,
        fundamentals_fetcher=ingest.fetch_fundamentals,
    )

    # ── Bearish ranking pass for the bearish day-trade list ─────────────────
    console.print("[bold][6/6][/bold] Bearish ranking + day-trade selection …")
    bear_candidates = scorer.get_top_bearish_candidates(tech_scores, n=20)
    bear_news       = ingest.fetch_news_for_tickers(bear_candidates)
    bear_sent_raw   = sentiment.score_sentiment_for_tickers(bear_news)
    bear_tech_dict  = {t: tech_scores[t] for t in bear_candidates if t in tech_scores}
    bear_ranked     = scorer.bearish_final_ranking(bear_tech_dict, bear_sent_raw)
    day_bearish_picks = scorer.select_top_day_bearish(bear_ranked, n=3)

    # 7. Enrichment — fundamentals/targets/options/notes per pick.
    #    Bullish dedup: a stock can appear in BOTH swing/long and day lists;
    #    enrich once, attach to both.
    console.print("[bold][+][/bold]  Deep analysis: targets, options, notes …")
    bullish_tickers = (
        {p["ticker"] for p in swing_long_picks}
        | {p["ticker"] for p in day_bullish_picks}
        | {p["ticker"] for p in momentum_continuation_picks}
        | {p["ticker"] for p in long_term_picks}
    )
    bullish_enriched: dict[str, dict] = {}

    for ticker in bullish_tickers:
        df = price_data.get(ticker)
        if df is None:
            continue
        tech              = tech_scores.get(ticker, {})
        last_price        = tech.get("last_price", 0)
        fundamentals      = ingest.fetch_fundamentals(ticker)
        next_earnings     = ingest.fetch_next_earnings_date(ticker)
        targets           = tgts.calculate_price_targets(df, fundamentals)
        trade_styles      = tgts.score_trade_styles(df, tech, fundamentals)
        options           = opts.get_options_recommendations(
            ticker, last_price, targets["swing_target"], targets["long_target"],
        )
        best_option       = opts.pick_best_option(options, direction="bullish")
        sma200     = tech.get("sma200", 0.0)
        sma200_pct = round((last_price / sma200 - 1) * 100, 1) if sma200 else 0.0

        # ── Catalyst intelligence ─────────────────────────────────────────────
        # EPS beat/miss: did the most recent quarter beat or miss estimates?
        earnings_beat = ingest.fetch_earnings_beat_miss(ticker)

        # Analyst target velocity: did consensus targets move up/down recently?
        # Query BEFORE storing so we compare against prior history, not today.
        current_target = fundamentals.get("targetMeanPrice")
        analyst_velocity = tracker.get_analyst_target_velocity(ticker, current_target)
        tracker.store_analyst_target(ticker, current_target)

        bullish_enriched[ticker] = {
            "fundamentals":        fundamentals,
            "next_earnings_date":  next_earnings,
            "targets":             targets,
            "trade_styles":        trade_styles,
            "options":             options,
            "best_option":         best_option,
            "sma200":              round(sma200, 2),
            "sma200_pct":          sma200_pct,   # +N% = above, -N% = below
            "earnings_beat":       earnings_beat,
            "analyst_velocity":    analyst_velocity,
        }

    def _attach_bullish(picks: list[dict]) -> list[dict]:
        for p in picks:
            data = bullish_enriched.get(p["ticker"])
            if not data:
                continue
            p.update(data)
            p["why_ranked"]     = _why_ranked(p)
            p["notes"]          = _generate_notes(p)
            p["volume_insight"] = _volume_insight(p, direction="bullish")
            p["position_size"]  = _suggest_position_size(p)
            p["earnings_flag"]  = _earnings_warning(p)
        return picks

    swing_long_picks            = _attach_bullish(swing_long_picks)
    day_bullish_picks           = _attach_bullish(day_bullish_picks)
    momentum_continuation_picks = _attach_bullish(momentum_continuation_picks)
    long_term_picks             = _attach_bullish(long_term_picks)

    for pick in swing_long_picks:
        bo = pick.get("best_option", {})
        console.print(
            f"      [swing/long] {pick['ticker']}: swing ${pick.get('targets',{}).get('swing_target',0)} | "
            f"best play: {bo.get('side','—')} ${bo.get('strike','—')}"
        )
    for pick in day_bullish_picks:
        ts = pick.get("trade_styles", {})
        console.print(
            f"      [day-bull]   {pick['ticker']}: day_score {ts.get('day_score',0):.0f} "
            f"({ts.get('day_stars','—')})  vol_today {ts.get('vol_ratio_day',0):.2f}×"
        )
    for pick in momentum_continuation_picks:
        console.print(
            f"      [momentum]   {pick['ticker']}: RSI {pick.get('rsi',0):.0f}  "
            f"5d {pick.get('ret5d_pct',0):+.1f}%  trend {pick.get('trend_score',0):.0f}"
        )
    for pick in long_term_picks:
        ts = pick.get("trade_styles", {})
        console.print(
            f"      [long-term]  {pick['ticker']}: long_score {ts.get('long_score',0):.0f} "
            f"({ts.get('long_stars','—')})  trend {pick.get('trend_score',0):.0f}  "
            f"sma200 {pick.get('sma200_pct',0):+.1f}%"
        )

    # Bearish day-trade enrichment
    if day_bearish_picks:
        for pick in day_bearish_picks:
            ticker = pick["ticker"]
            df     = price_data.get(ticker)
            if df is None:
                continue
            tech = tech_scores.get(ticker, {})

            fundamentals              = ingest.fetch_fundamentals(ticker)
            next_earnings             = ingest.fetch_next_earnings_date(ticker)
            pick["fundamentals"]      = fundamentals
            pick["next_earnings_date"] = next_earnings
            pick["targets"]           = tgts.calculate_bearish_levels(df)
            pick["trade_styles"]      = tgts.score_trade_styles(df, tech, fundamentals)
            pick["options"]           = opts.get_options_recommendations(
                ticker, pick.get("last_price", 0),
                pick.get("last_price", 0) * 1.05,
                pick.get("last_price", 0) * 1.10,
            )
            pick["best_option"]    = opts.pick_best_option(pick["options"], direction="bearish")
            pick["notes"]          = _generate_bearish_notes(pick)
            pick["why_ranked"]     = _why_ranked_bearish(pick)
            pick["volume_insight"] = _volume_insight(pick, direction="bearish")
            pick["position_size"]  = _suggest_position_size(pick)
            pick["earnings_flag"]  = _earnings_warning(pick)
            ts = pick.get("trade_styles", {})
            console.print(
                f"      [day-bear]   {ticker}: day_score {ts.get('day_score',0):.0f} "
                f"({ts.get('day_stars','—')})  vol_today {ts.get('vol_ratio_day',0):.2f}×"
            )
    else:
        console.print("      [dim]No qualifying bearish day-trade setups today.[/dim]")

    # Macro context — SPY/VIX/DXY regime + thematic ETF benchmarks
    console.print("[bold][+][/bold]  Fetching macro snapshot + thematic ETF benchmarks …")
    macro_data     = macro_mod.get_macro_snapshot()
    from watchlist_check import load_benchmarks
    benchmarks     = load_benchmarks(watchlist) if watchlist else []
    etf_benchmarks = macro_mod.get_thematic_etf_returns(benchmarks or None)
    if macro_data.get("available"):
        console.print(
            f"      Regime: {macro_data['regime']}  |  "
            f"SPY ${macro_data['spy_close']} ({macro_data['spy_5d']:+.2f}% 5d)  |  "
            f"VIX {macro_data['vix']:.1f}"
        )

    # Save results and recommendations
    paths = report.save_results(
        swing_long_picks, all_ranked, market_sent,
        day_bullish=day_bullish_picks,
        day_bearish=day_bearish_picks,
        momentum_continuation=momentum_continuation_picks,
        long_term=long_term_picks,
        macro=macro_data,
        etf_benchmarks=etf_benchmarks,
        tag=watchlist,
    )
    tracker.save_recommendations(swing_long_picks)

    report.print_summary(
        swing_long_picks, market_sent,
        day_bullish=day_bullish_picks,
        day_bearish=day_bearish_picks,
        momentum_continuation=momentum_continuation_picks,
        long_term=long_term_picks,
        macro=macro_data,
        etf_benchmarks=etf_benchmarks,
        stocks_only=stocks_only,
    )

    console.print("\n[bold green]Files saved:[/bold green]")
    for path in paths.values():
        console.print(f"  [dim]{path}[/dim]")

    return swing_long_picks


def _generate_notes(pick: dict) -> str:
    """
    Senior-desk read on a single pick — 2–4 sentences in trader voice.
    Frames: setup quality, institutional footprint, key levels, risk flags.
    """
    ticker    = pick["ticker"]
    ret5d     = pick.get("ret5d_pct", 0)
    vol_ratio = pick.get("vol_ratio", 1.0)
    rsi       = pick.get("rsi", 50)
    tgts_d    = pick.get("targets", {})
    style     = pick.get("trade_styles", {})
    fund      = pick.get("fundamentals", {})
    best_opt  = pick.get("best_option", {})

    parts: list[str] = []

    # ── Setup quality + price action ─────────────────────────────────────────
    if ret5d >= 15:
        parts.append(f"{ticker} ripping +{ret5d:.1f}% on the week — late-stage momentum, not an early entry.")
    elif ret5d >= 5:
        parts.append(f"{ticker} +{ret5d:.1f}% this week with the trend intact; setup is constructive.")
    else:
        parts.append(f"{ticker} basing with a clean technical setup — patience play, not a breakout chase.")

    # ── Institutional footprint ──────────────────────────────────────────────
    if vol_ratio >= 2.0:
        parts.append(f"Volume {vol_ratio:.1f}× normal — institutions are showing their hand.")
    elif vol_ratio >= 1.5:
        parts.append(f"Volume {vol_ratio:.1f}× normal — real participation, not retail noise.")

    # ── Analyst frame ────────────────────────────────────────────────────────
    analyst_mean = tgts_d.get("analyst_mean")
    long_pct     = tgts_d.get("long_target_pct", 0)
    if analyst_mean:
        if long_pct >= 5:
            parts.append(f"Street consensus ${analyst_mean} ({long_pct:+.0f}%) — analysts agree there's room.")
        elif long_pct <= -5:
            parts.append(f"Street consensus ${analyst_mean} ({long_pct:+.0f}%) — price is ahead of coverage. Caution.")
        else:
            parts.append(f"Street consensus ${analyst_mean} — fair value zone.")

    # ── EPS catalyst ─────────────────────────────────────────────────────────
    earnings_beat = pick.get("earnings_beat", {})
    if earnings_beat.get("beat") and earnings_beat.get("beat_pct", 0) >= 5:
        beat_pct = earnings_beat["beat_pct"]
        if beat_pct >= 15:
            parts.append(
                f"EPS beat by +{beat_pct:.0f}% last quarter — institutions likely still re-rating; "
                f"this is a catalyst trade, not just a technical setup."
            )
        else:
            parts.append(f"EPS beat +{beat_pct:.0f}% — clean catalyst backing the tape.")
    elif earnings_beat.get("beat") is False and earnings_beat.get("beat_pct", 0) <= -5:
        parts.append(
            f"EPS miss {earnings_beat['beat_pct']:.0f}% last quarter — "
            f"watch for overhead supply from trapped longs."
        )

    # ── Analyst target velocity ──────────────────────────────────────────────
    vel = pick.get("analyst_velocity") or {}
    if vel.get("pct_change", 0) >= 5:
        parts.append(
            f"Analyst targets raised +{vel['pct_change']:.0f}% over {vel['days_ago']} days "
            f"(from ${vel['from_target']}) — coverage agrees with the move."
        )
    elif vel.get("pct_change", 0) <= -5:
        parts.append(
            f"Analyst targets cut {vel['pct_change']:.0f}% over {vel['days_ago']} days "
            f"(from ${vel['from_target']}) — street is ahead of the price. Caution on swing length."
        )

    # ── Best timeframe ───────────────────────────────────────────────────────
    scores = {
        "intraday":  style.get("day_score", 0),
        "swing":     style.get("swing_score", 0),
        "long-term": style.get("long_score", 0),
    }
    best_style, best_sc = max(scores.items(), key=lambda x: x[1])
    if best_sc >= 60:
        parts.append(f"Trade it as a {best_style} setup.")

    # ── Risk flags ──────────────────────────────────────────────────────────
    if rsi > 75:
        parts.append(f"RSI {rsi:.0f} — overbought. Wait for a pullback or scale in small.")
    elif rsi < 40:
        parts.append(f"RSI {rsi:.0f} — weak. Need a bounce confirmation before stepping in.")

    sector = fund.get("sector", "")
    if sector:
        parts.append(f"Sector: {sector}.")

    # ── Options frame ───────────────────────────────────────────────────────
    if best_opt.get("available"):
        parts.append(
            f"Options view: {best_opt['timeframe']} {best_opt['side']} "
            f"${best_opt['strike']:.0f} exp {best_opt['expiry']} "
            f"— ${best_opt['contract_cost']:.0f}/contract, BE ${best_opt['break_even']:.2f} "
            f"({best_opt['pct_to_breakeven']:+.1f}%). {best_opt['reason']}."
        )

    return "  ".join(parts)


def _why_ranked(pick: dict) -> str:
    """
    One-line reason this stock made the top 5 — for the summary table column.
    Combines the 2–3 strongest signals into a short, comma-separated string.
    """
    parts: list[str] = []
    ret5d     = pick.get("ret5d_pct", 0)
    vol_ratio = pick.get("vol_ratio", 1.0)
    up_pct    = pick.get("vol_up_pct", 0.5)
    rsi       = pick.get("rsi", 50)
    macd_h    = pick.get("macd_hist", 0)
    w52       = pick.get("week52_pos", 0)
    fund      = pick.get("fundamentals", {})
    tgts_d    = pick.get("targets", {})

    # Momentum
    if ret5d >= 15:
        parts.append(f"+{ret5d:.0f}% surge this week")
    elif ret5d >= 5:
        parts.append(f"+{ret5d:.0f}% this week")

    # Catalyst signals first — these are higher-signal than MACD/52-wk position
    # and must survive the parts[:6] truncation below.
    earnings_beat = pick.get("earnings_beat", {})
    if earnings_beat.get("beat") and earnings_beat.get("beat_pct", 0) >= 5:
        parts.append(f"EPS beat +{earnings_beat['beat_pct']:.0f}%")
    elif earnings_beat.get("beat") is False and earnings_beat.get("beat_pct", 0) <= -5:
        parts.append(f"EPS miss {earnings_beat['beat_pct']:.0f}%")

    vel = pick.get("analyst_velocity") or {}
    if vel.get("pct_change", 0) >= 5:
        parts.append(f"analyst target raised +{vel['pct_change']:.0f}% ({vel['days_ago']}d)")
    elif vel.get("pct_change", 0) <= -5:
        parts.append(f"analyst target cut {vel['pct_change']:.0f}% ({vel['days_ago']}d)")

    # Volume — call out direction skew if strong
    if vol_ratio >= 1.5:
        if up_pct >= 0.65:
            parts.append(f"{vol_ratio:.1f}× volume mostly on up days ({up_pct*100:.0f}%)")
        else:
            parts.append(f"{vol_ratio:.1f}× avg volume")

    # RSI
    if 50 <= rsi <= 68:
        parts.append(f"RSI {rsi:.0f} (momentum zone)")
    elif rsi > 75:
        parts.append(f"RSI {rsi:.0f} (overbought — caution)")

    # MACD
    if macd_h > 0:
        parts.append("MACD bullish crossover")

    # 52-week position
    if w52 >= 0.90:
        parts.append("near 52-wk high")

    # Fundamentals
    rev_gr = fund.get("revenueGrowth")
    if rev_gr and float(rev_gr) > 0.15:
        parts.append(f"{float(rev_gr)*100:.0f}% revenue growth")

    pe = fund.get("forwardPE") or fund.get("trailingPE")
    if pe and float(pe) < 20:
        parts.append(f"P/E {float(pe):.1f}")

    analyst = tgts_d.get("analyst_mean")
    price   = tgts_d.get("current_price")
    if analyst and price:
        # Measure against the analyst target itself, not long_target_pct -
        # long_target falls back to a 52-wk/15% projection when analysts
        # sit below the current price.
        pct = (float(analyst) / float(price) - 1) * 100
        parts.append(f"analyst target ${analyst} ({pct:+.0f}%)")

    return ", ".join(parts[:6]) if parts else "strong overall technical score"


def _suggest_position_size(pick: dict) -> str:
    """
    ATR-based position sizing — the veteran's rule. Half the trade is the
    pick, the other half is sizing. Without sizing, even good picks blow
    up the account.

    Lower volatility → larger position; higher → smaller. Hard cap 5%.
    """
    ts = pick.get("trade_styles", {})
    atr_pct = ts.get("atr_pct", pick.get("atr_pct", 0)) or 0
    if atr_pct == 0:
        return "size: 2% (default — no vol read)"
    if atr_pct < 2:    return "size: 4–5% (low vol — can lean in)"
    if atr_pct < 3:    return "size: 3% (normal vol)"
    if atr_pct < 5:    return "size: 2% (elevated vol — tighten stop)"
    return "size: 1% (high vol — treat as a trade, not a position)"


def _earnings_warning(pick: dict) -> str:
    """
    If a confirmed earnings date falls between today and option expiry,
    flag it. IV crush after earnings will wreck even a directionally-correct
    options trade.
    """
    next_earnings = pick.get("next_earnings_date")
    bo = pick.get("best_option", {})
    if not next_earnings or not bo.get("available"):
        return ""
    expiry = bo.get("expiry", "")
    try:
        # Normalise both sides to plain `date` to avoid datetime/date/Timestamp mix-ups
        ne_date = next_earnings.date() if hasattr(next_earnings, "date") else next_earnings
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        today_date = datetime.today().date()
        if today_date <= ne_date <= exp_date:
            return f" ⚠ EPS {ne_date.strftime('%m/%d')}"
    except Exception:
        return ""
    return ""


def _volume_insight(pick: dict, direction: str = "bullish") -> str:
    """
    Desk read on volume — what the tape is telling you. Translates the raw
    20d/60d ratios into trader language: distribution, accumulation, capitulation,
    quiet, institutional footprint.
    """
    ts      = pick.get("trade_styles", {})
    vd      = ts.get("vol_ratio_day", 1.0)
    vl      = ts.get("vol_ratio_long", 1.0)
    avg_vol = ts.get("avg_volume", 0)

    flow = "selling" if direction == "bearish" else "buying"

    if vd >= 2.5:
        if avg_vol >= 1_000_000:
            return f"Heavy {flow} ({vd:.1f}× normal) — institutions showing their hand"
        return f"Volume spike ({vd:.1f}×) on thin liquidity — hard to trade cleanly"

    if vd >= 1.5:
        return f"Above-avg {flow} today ({vd:.1f}×) — real participation"

    if direction == "bullish" and vl >= 1.15:
        return f"Sustained accumulation ({vl:.2f}× 60d avg) — building position"
    if direction == "bearish" and vl <= 0.85:
        return f"Interest bleeding out ({vl:.2f}× 60d avg) — supports the short thesis"
    if direction == "bullish" and vl <= 0.85:
        return "Surge without long-term accumulation — chase risk, not a base"

    if vd < 0.7:
        return f"Quiet tape ({vd:.2f}× normal) — wait for confirmation"

    return "Normal tape — no unusual footprint"


def _generate_bearish_notes(pick: dict) -> str:
    """Senior desk read on a bearish setup — 3–4 sentences in trader voice."""
    ticker    = pick["ticker"]
    ret5d     = pick.get("ret5d_pct", 0)
    rsi       = pick.get("rsi", 50)
    macd_h    = pick.get("macd_hist", 0)
    w52       = pick.get("week52_pos", 0)
    sma200    = pick.get("sma200", 0)
    price     = pick.get("last_price", 0)
    fund      = pick.get("fundamentals", {})
    best_opt  = pick.get("best_option", {})

    parts: list[str] = []

    if ret5d <= -10:
        parts.append(f"{ticker} down {ret5d:.1f}% on the week — heavy supply, sellers in control.")
    elif ret5d <= -3:
        parts.append(f"{ticker} {ret5d:.1f}% this week — downtrend confirmed, not a one-day flush.")
    elif w52 < 0.30:
        parts.append(f"{ticker} sitting at {w52*100:.0f}% of 52-week range — basing near lows, weakness persistent.")
    else:
        parts.append(f"{ticker} bearish setup on the daily despite a flat tape — watch for the break.")

    if sma200 and price < sma200:
        parts.append("Below the 200-day — long-term structure broken.")
    if macd_h < 0:
        parts.append("MACD histogram negative — momentum confirms the bear case.")
    if rsi > 75:
        parts.append(f"RSI {rsi:.0f} — overbought into resistance, classic put setup.")

    sector = fund.get("sector", "")
    if sector:
        parts.append(f"Sector: {sector}.")

    if best_opt.get("available"):
        parts.append(
            f"Options view: {best_opt['timeframe']} {best_opt['side']} "
            f"${best_opt['strike']:.0f} exp {best_opt['expiry']} — "
            f"${best_opt['contract_cost']:.0f}/contract, BE ${best_opt['break_even']:.2f}. "
            f"{best_opt['reason']}."
        )

    parts.append(
        "Read the caveat: puts decay daily and a sharp counter-rally rips premium fast. "
        "Size small, define invalidation, and respect the stop."
    )

    return "  ".join(parts)


def _why_ranked_bearish(pick: dict) -> str:
    """One-line reason this stock made the bearish watch list."""
    parts: list[str] = []
    ret5d     = pick.get("ret5d_pct", 0)
    rsi       = pick.get("rsi", 50)
    macd_h    = pick.get("macd_hist", 0)
    w52       = pick.get("week52_pos", 0.5)
    price     = pick.get("last_price", 0)
    sma200    = pick.get("sma200", 0)
    sent      = pick.get("sentiment_score", 50)
    vol_ratio = pick.get("vol_ratio", 1.0)
    down_pct  = pick.get("vol_down_pct", 0.5)

    if ret5d <= -10:
        parts.append(f"{ret5d:.0f}% weekly decline")
    elif ret5d <= -3:
        parts.append(f"{ret5d:.0f}% this week")

    # Selling-pressure volume confirmation
    if vol_ratio >= 1.5 and down_pct >= 0.65:
        parts.append(f"{vol_ratio:.1f}× volume mostly on down days ({down_pct*100:.0f}%)")

    if sma200 and price < sma200:
        parts.append("below 200 SMA")

    if macd_h < 0:
        parts.append("MACD bearish")

    if rsi > 75:
        parts.append(f"RSI {rsi:.0f} (topping)")
    elif rsi <= 45:
        parts.append(f"RSI {rsi:.0f} (weak)")

    if w52 < 0.25:
        parts.append("near 52-wk low")

    if sent < 35:
        parts.append(f"negative news ({sent:.0f}/100)")

    return ", ".join(parts[:5]) if parts else "active downtrend setup"


def _clear_caches() -> None:
    from config import DATA_DIR
    for f in DATA_DIR.glob("*.pkl"):
        f.unlink()
    for f in DATA_DIR.glob("tickers.json"):
        f.unlink()
    console.print("[yellow]Cache cleared — fresh data will be downloaded[/yellow]")


# ── Sub-command handlers ──────────────────────────────────────────────────────

def _cmd_status() -> None:
    # Newest by mtime: file names carry an optional watchlist tag after the
    # date, so a name sort no longer means newest first.
    reports = sorted(OUTPUT_DIR.glob("report_*.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not reports:
        console.print("[yellow]No results yet. Run: python main.py run[/yellow]")
        return
    latest = reports[0]
    console.print(f"[dim]{latest}[/dim]\n")
    console.print(latest.read_text())


def _cmd_track(args: argparse.Namespace) -> None:
    import tracker
    tracker.log_trade(
        ticker=args.ticker.upper(),
        action=args.action,
        price=args.price,
        shares=getattr(args, "shares", 1),
        notes=getattr(args, "notes", ""),
    )
    console.print(
        f"[green]Logged:[/green] {args.action.upper()} "
        f"{getattr(args, 'shares', 1)}× {args.ticker.upper()} @ ${args.price:.2f}"
    )


def _cmd_positions() -> None:
    import tracker
    positions = tracker.get_open_positions()
    if not positions:
        console.print("[yellow]No open positions.[/yellow]")
        return
    tbl = Table(title="Open Paper Positions", show_lines=True)
    tbl.add_column("Ticker",          style="bold yellow")
    tbl.add_column("Shares",          justify="right")
    tbl.add_column("Avg Buy Price",   justify="right")
    for p in positions:
        tbl.add_row(p["ticker"], str(p["net_shares"]), f"${p['avg_buy_price']:.2f}")
    console.print(tbl)


def _cmd_history() -> None:
    import tracker
    history = tracker.get_recommendation_history()
    if not history:
        console.print("[yellow]No history yet.[/yellow]")
        return
    tbl = Table(title="Recommendation History", show_lines=True)
    for col in ("run_date", "rank", "ticker", "price_at_rec", "combined_score"):
        tbl.add_column(col.replace("_", " ").title())
    for row in history[:50]:   # cap at 50 rows for readability
        tbl.add_row(
            row["run_date"],
            str(row["rank"]),
            row["ticker"],
            f"${row['price_at_rec']:.2f}" if row["price_at_rec"] else "—",
            f"{row['combined_score']:.1f}" if row["combined_score"] else "—",
        )
    console.print(tbl)


# ── CLI Setup ─────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stock-research",
        description="Academic stock research pipeline — paper trading recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # run
    run_p = sub.add_parser("run", help="Run the full analysis pipeline now")
    run_p.add_argument("--no-cache", action="store_true", help="Force re-download all data")
    run_p.add_argument("--watchlist", type=str, default=None,
                       help="Watchlist name (file in watchlists/) — added to universe")
    run_p.add_argument("--watchlist-only", action="store_true",
                       help="With --watchlist: analyze ONLY watchlist tickers")
    run_p.add_argument("--stocks-only", action="store_true",
                       help="Render ONLY the 3 stock tables (intraday, swing/long, long-term) "
                            "— suppress options, momentum, and bearish tables")
    run_p.add_argument("--skip-watchlist-check", action="store_true",
                       help="With --watchlist: skip the pre-run watchlist health check")
    run_p.add_argument("--skip-sector-scan", action="store_true",
                       help="Skip the pre-run sector rotation scan")

    # sectors
    sub.add_parser("sectors", help="Sector rotation scan: trending, emerging and breaking themes")

    # evolve
    evo_p = sub.add_parser("evolve", help="Self-improving loop: propose technical-weight changes, "
                                          "keep only those that pass a holdout test")
    evo_p.add_argument("--watchlist",  default="ai_ecosystem",
                       help="Watchlist to replay, or 'sp500' for the index (default ai_ecosystem)")
    evo_p.add_argument("--period",     default="3y", help="Price history to replay, e.g. 3y or 5y (default 3y)")
    evo_p.add_argument("--horizon",    type=int, default=10, help="Forward window in trading days (default 10)")
    evo_p.add_argument("--step",       type=int, default=5,  help="Days between replay dates (default 5)")
    evo_p.add_argument("--iterations", type=int, default=40, help="Changes to try (default 40)")
    evo_p.add_argument("--seed",       type=int, default=0,  help="Random seed, for reproducible runs")
    evo_p.add_argument("--apply", action="store_true",
                       help="Write params/scoring.json if, and only if, the verdict is PASS")

    # experiment
    ex_p = sub.add_parser("experiment", help="Test pre-registered structural changes to the score "
                                             "(holdout-gated, must hold on 3 of 4 watchlists)")
    ex_p.add_argument("--watchlists", default="ai_ecosystem,cyber,biotech,energy",
                      help="Comma-separated watchlists, or 'sp500' (default: the 4 main lists)")
    ex_p.add_argument("--horizon", type=int, default=10, help="Forward window in trading days (default 10)")
    ex_p.add_argument("--period", default="3y", help="Price history to replay, e.g. 3y or 5y (default 3y)")

    # journal
    jr_p = sub.add_parser("journal", help="Research journal: grade the sector scan's predictions")
    jr_p.add_argument("action", choices=["backfill", "grade", "report"])
    jr_p.add_argument("--weeks", type=int, default=104, help="backfill: weeks of history (default 104)")

    # check-watchlist
    chk_p = sub.add_parser("check-watchlist",
                           help="Health-check a watchlist (market, theme, technicals, earnings, news, trending)")
    chk_p.add_argument("name", help="Watchlist name (file in watchlists/)")

    # schedule
    sub.add_parser("schedule", help="Start the nightly automated scheduler")

    # monitor
    mon_p = sub.add_parser("monitor", help="Watch top picks (or a full watchlist) intraday and alert on big moves")
    mon_p.add_argument("--interval",  type=int, default=300, help="Poll interval in seconds (default 300)")
    mon_p.add_argument("--watchlist", type=str, default=None,
                       help="Watchlist name to monitor (e.g. ai_ecosystem). Omit to watch today's picks only.")

    # status
    sub.add_parser("status", help="Show the most recent saved report")

    # track
    track_p = sub.add_parser("track", help="Log a paper trade")
    track_p.add_argument("action", choices=["buy", "sell"])
    track_p.add_argument("ticker")
    track_p.add_argument("--price",  type=float, required=True)
    track_p.add_argument("--shares", type=int,   default=1)
    track_p.add_argument("--notes",  type=str,   default="")

    # positions
    sub.add_parser("positions", help="Show open paper trade positions")

    # history
    sub.add_parser("history", help="Show all past recommendations")

    # backtest
    bt_p = sub.add_parser("backtest", help="Replay saved recommendations against actual prices")
    bt_p.add_argument("--days",  type=int, default=5,  help="Days-forward window (default 5)")
    bt_p.add_argument("--since", type=int, default=30, help="Look at recs from last N days (default 30)")

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "run":
        run_pipeline(
            force_refresh=args.no_cache,
            watchlist=args.watchlist,
            watchlist_only=args.watchlist_only,
            stocks_only=args.stocks_only,
            check_watchlist=not args.skip_watchlist_check,
            sector_scan=not args.skip_sector_scan,
        )
    elif args.command == "sectors":
        import sector_scan
        sector_scan.run_scan()
    elif args.command == "evolve":
        import evolve
        evolve.run(args.watchlist, horizon=args.horizon, step=args.step,
                   iterations=args.iterations, seed=args.seed, do_apply=args.apply,
                   period=args.period)
    elif args.command == "experiment":
        import experiments
        experiments.run([w.strip() for w in args.watchlists.split(",") if w.strip()],
                        horizon=args.horizon, period=args.period)
    elif args.command == "journal":
        import journal
        if args.action == "backfill":
            console.print(f"Backfilling {args.weeks} weeks of sector-scan calls …")
            journal.backfill(weeks=args.weeks)
        elif args.action == "grade":
            console.print(f"Graded {journal.grade()} predictions.")
        journal.render_report()
    elif args.command == "check-watchlist":
        import watchlist_check
        watchlist_check.run_check(args.name)
    elif args.command == "schedule":
        from scheduler import start_scheduler
        start_scheduler(run_pipeline)
    elif args.command == "monitor":
        if args.watchlist:
            from ingest import load_watchlist
            from monitor import monitor_watchlist
            tickers = load_watchlist(args.watchlist)
            monitor_watchlist(tickers, poll_interval=args.interval)
        else:
            from monitor import monitor_top_picks
            monitor_top_picks(poll_interval=args.interval)
    elif args.command == "status":
        _cmd_status()
    elif args.command == "track":
        _cmd_track(args)
    elif args.command == "positions":
        _cmd_positions()
    elif args.command == "history":
        _cmd_history()
    elif args.command == "backtest":
        from backtest import run_backtest
        run_backtest(days_forward=args.days, since_days=args.since)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
