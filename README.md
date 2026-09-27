# Stock Research Pipeline

[![CI](https://github.com/gh325/stock-research/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gh325/stock-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A senior-analyst-grade morning scan. Sweeps a watchlist (or the full S&P 500),
scores every name on technicals + sentiment + fundamentals, and surfaces the
**three ranked stock tables** a trader actually uses: intraday setups,
swing/long candidates, and long-term holds. Options tables available on demand.

Built for paper-trading and learning the craft — the kind of pre-market read
you'd get sitting on a senior trading desk.

> ⚠️ **Not financial advice.** This is a quantitative screener with veteran-voice
> commentary, not a recommendation engine. The numbers find the setups; you and
> your risk management decide what to do with them.

---

## Why it exists

Retail screeners show you a list of tickers. They don't tell you whether the
theme is still leading, whether the 1-year target is a real analyst number or a
projection, whether earnings land inside your hold, or how big the position
should be. This tool answers those questions in one pass, for learning and
paper trading.

## Requirements

- Python 3.10 or newer. CI tests 3.11 and 3.12. numpy is pinned below 2.0, so
  Python 3.13+ has to build numpy from source.
- Internet access. Price, news and fundamentals come from Yahoo Finance through
  [yfinance](https://github.com/ranaroussi/yfinance). No API key is needed.
- Optional: a [NewsAPI](https://newsapi.org) key in `.env` (see `.env.example`).

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone git@github.com:gh325/stock-research.git
cd stock-research
uv sync
```

Or with plain pip, using the setup script. It creates `.venv` and copies
`.env.example` to `.env`:

```bash
bash setup.sh
source .venv/bin/activate
```

## Quick Start

```bash
cd stock-research
source .venv/bin/activate          # or prefix commands with `uv run`

# Recommended daily run — fresh data, AI ecosystem, all 3 stock tables
python main.py run --no-cache --watchlist ai_ecosystem --watchlist-only --stocks-only
```

That single command always produces the **3 stock tables** described below.
Before the tables it prints a sector rotation scan and a watchlist health check.

---

## The 3 Stock Tables (always rendered)

Every `--stocks-only` run produces these three tables in order:

### 1. ⚡ Top 3 Intraday Stocks (same-day, no options)

**Time horizon:** open → close (one trading day)
**Use case:** quick scalp / day-trade based on volatility + volume.

| Column | What it means |
|--------|---------------|
| `#` | Rank |
| `Ticker` | Symbol (+ `⚠ EPS MM/DD` if earnings within option expiry) |
| `Current` | Live price at run time |
| `Day Target ↑` | 1.5× ATR profit target |
| `Day Stop ↓` | 1× ATR stop loss |
| `ATR%` | Average true range as % of price — daily swing magnitude |
| `RSI` | 14-period RSI. 30=oversold, 70=overbought, 50–68=momentum zone |
| `Today vs 20d` | Today's volume vs 20-day average (1.5×+ = unusual interest) |
| `Day ★` | 1–5 star intraday suitability rating |
| `5d Return` | Past week price change |
| `Sector` | GICS sector |
| `What This Tells You` | Plain-English volume interpretation |
| `Why It's a Day-Trade` | The 2–3 strongest technical signals |

### 2. 📊 Top 5 Swing/Long Stocks (3 weeks – 3 months, no options)

**Time horizon:** weeks to a few months.
**Use case:** ride a trend with defined targets, exit before earnings risk.

| Column | What it means |
|--------|---------------|
| `Swing Target` | 3-week target based on ATR + analyst data |
| `Long Target` | 3-month target |
| `Swing Stop` | 2× ATR stop loss |
| `Analyst Target` | Wall Street consensus from yfinance |
| `20d vs 60d` | 20-day/60-day volume ratio (>1 = accumulation) |
| `Swing ★` | Star rating for swing suitability |
| `P/E` | Forward (preferred) or trailing P/E |
| Earnings flag in `Ticker` | `⚠ EPS MM/DD` warning |

Sector cap: max 2 picks per GICS sector — prevents 5 picks all in semis.

### 3. 📈 Top 5 Long-Term Stocks (1 year+, no options)

**Time horizon:** 1 year or longer.
**Use case:** identify trends backed by fundamentals (revenue growth + margins + above 200 SMA).

| Column | What it means |
|--------|---------------|
| `1Y Target` | Analyst consensus when it is above the price. Otherwise the higher of the 52-week high and price +15% |
| `Target Src` | `Analyst` (green) or a tool projection (yellow): `52w high` / `+15% rule`, tagged `(analysts below)` or `(no analysts)`. Treat yellow upside with suspicion |
| `Upside` | % gain to 1Y target |
| `Rev Growth` | YoY revenue growth |
| `Fwd P/E` | Forward price-to-earnings |
| `Profit Margin` | Net profit margin |
| `vs 200 SMA` | % above/below the 200-day SMA |
| `Accum 20d/60d` | Volume accumulation ratio |
| `Long ★` | Long-term star rating |
| `Why Long Term` | Combined signal summary |

Selection logic: 50% technical_score + 35% trend_score + 15% volume accumulation. Trend score < 25 (below all SMAs) is filtered out.

---

## Command Reference (with examples)

### Daily recommended run

```bash
# Always show 3 stock tables, fresh data, 90-ticker AI ecosystem
python main.py run --no-cache --watchlist ai_ecosystem --watchlist-only --stocks-only
```

### All output modes

```bash
# Default: full pipeline with options tables
python main.py run                                    # ~500 S&P 500 stocks, cached
python main.py run --no-cache                         # Force fresh download

# Watchlist-focused (faster, themed)
python main.py run --watchlist ai_ecosystem           # Add 90 AI tickers to universe
python main.py run --watchlist ai_ecosystem --watchlist-only
                                                       # Run ONLY the 90 tickers
python main.py run --watchlist ai_chips --watchlist-only
                                                       # 38-ticker AI chips watchlist

# 3-tables-only mode (no options noise)
python main.py run --stocks-only                      # Full universe, 3 stock tables only
python main.py run --watchlist ai_ecosystem --watchlist-only --stocks-only
                                                       # 90-ticker focus, 3 tables only
```

### Backtesting & history

```bash
python main.py backtest                               # Replay last 30d picks vs SPY
python main.py backtest --days 10 --since 60          # Custom window
python main.py history                                # All past recommendations
python main.py status                                 # Most recent saved report
```

### Paper trading

```bash
python main.py track buy AAPL --price 182 --shares 10
python main.py track sell AAPL --price 195
python main.py positions                              # Open paper positions
```

### Automation

```bash
python main.py schedule                               # Start 9PM ET nightly scheduler

# Picks monitor — watches today's 5-10 pipeline picks only
python main.py monitor                                # Poll every 5 min
python main.py monitor --interval 60                  # Poll every 60s

# Watchlist monitor — watches every ticker in a watchlist
python main.py monitor --watchlist ai_ecosystem       # 90 tickers, 5-min poll
python main.py monitor --watchlist ai_ecosystem --interval 60   # 90 tickers, 1-min
python main.py monitor --watchlist ai_chips --interval 120      # 38 chips tickers
```

---

## Self-Improving Loop

Two tools check the screener against what actually happened, and change it
only when the evidence holds up. This follows the "autoresearch" pattern:
make one change, measure it, then keep it or revert it.

### Research journal: do the sector scan's calls come true?

```bash
python main.py journal backfill   # grade ~2 years of scans from price history
python main.py journal grade      # grade live calls whose 20-day window has passed
python main.py journal report     # hit rate per bucket, with 95% ranges
```

Every sector scan logs its calls: Trending and Emerging themes should beat SPY
over the next 20 trading days, and Breaking themes should lag it. Calls on the
same ETF never overlap in time, so each one is independent evidence. A bucket
is marked "works" only when its whole 95% range clears 50%.

### Evolve: tune the technical weights, but only on evidence

```bash
python main.py evolve --watchlist ai_ecosystem          # propose, test, report
python main.py evolve --watchlist ai_ecosystem --apply  # write the weights only on PASS
```

The loop works in five steps:

1. It replays ~2 years of history on one shared calendar. It scores every ticker
   as of each date, using only data up to that date.
2. It moves 0.05 of weight between two components, one change at a time. It keeps
   a change only if the top-5 picks improve on the training window.
3. It gates once on a held-out recent window. The new weights must beat the
   current ones there by t ≥ 2. On pure noise, this gate falsely passes about
   2.5% of the time; `tests/test_evolve.py` checks that.
4. `--apply` writes `params/scoring.json` and appends to
   `params/evolution_log.jsonl`. Commit that on a branch and review it like code.
5. The loop never edits code. Weights stay between 0.05 and 0.50.

Limits:

- It tunes technical weights only, because sentiment and fundamentals have no
  history to replay.
- The replay uses today's watchlist, so it has survivorship bias. It also models
  no trading costs.

## Example Output (truncated)

```
🌍 Risk-On | SPY $742.31 (+10.4% vs 200-SMA) | VIX 17.9 | News mood: Slightly Bullish

★ Top 3 Intraday Stocks
┃ # ┃ Ticker ┃ Current ┃ Day Target ↑ ┃ Day Stop ↓ ┃ ATR% ┃ RSI ┃ Day ★ ┃ ...
│ 1 │ QUBT   │  $11.03 │       $12.48 │     $10.06 │ 8.8% │  63 │ ★★★★★ │
│ 2 │ AAOI   │ $223.10 │      $258.45 │    $199.53 │10.6% │  69 │ ★★★★★ │
│ 3 │ HPE    │  $32.07 │       $33.81 │     $30.91 │ 3.6% │  71 │ ★★★★★ │

Top 5 Swing/Long Stocks  (hold days to months)
┃ # ┃ Ticker ┃ Current ┃ Swing Target ┃ Long Target ┃ Analyst Target ┃ Swing ★ ┃ P/E ┃ ...
│ 1 │ QUBT   │  $11.03 │       $13.93 │      $17.83 │         $17.83 │ ★★★★★   │ -39 │
│ 2 │ MTSI   │ $381.55 │      $463.66 │     $389.43 │        $389.43 │ ★★★☆☆   │  56 │
│ ...

📈 Top 5 Long-Term Stocks  (hold 1 year+)
┃ # ┃ Ticker ┃ Current ┃ 1Y Target ┃ Upside ┃ Rev Growth ┃ Fwd P/E ┃ Margin ┃ Long ★
│ 1 │ QUBT   │  $11.03 │    $17.83 │ +61.6% │     +9364% │   -38.7 │   —    │ ★★★★☆
│ 2 │ AAOI   │ $223.10 │   $256.57 │ +15.0% │       +51% │    46.8 │  -8.5% │ ★★☆☆☆
│ ...
```

---

## How It Decides (scoring architecture)

```
All tickers (watchlist or S&P 500)
  ↓
[quality filter: price > $5, avg volume > 500k]
  ↓
technicals.score_all_technicals()
  → technical_score, bearish_score, day_score, trend_score
  ↓
top 50 by technical_score
  ↓
ingest.fetch_news_for_tickers() → sentiment.score_sentiment_for_tickers()
  → sentiment_score
  ↓
scorer.final_ranking()
  combined_score = 0.70 × technical + 0.30 × sentiment
  ↓
  ├── select_top_picks(n=5, max_per_sector=2)         → swing/long picks
  ├── select_top_day_bullish(n=3, min_day_score=60)   → intraday picks
  ├── select_long_term_picks(n=5, max_per_sector=2)   → long-term picks
  ├── select_momentum_continuation_picks(rsi≥70)      → bonus list (options mode)
  └── [bearish pass] → day_bearish_picks              → PUTs list (options mode)
  ↓
enrichment: fundamentals + targets + options + earnings + position size
  ↓
render 3–6 tables → save CSVs + JSON + text report
```

Key constants in `config.py`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `WEIGHT_TECHNICAL` | 0.70 | Technical score weight |
| `WEIGHT_SENTIMENT` | 0.30 | Sentiment score weight |
| `TOP_CANDIDATES` | 50 | Stocks scored for sentiment |
| `TOP_PICKS` | 5 | Swing/long & long-term list size |
| `MIN_PRICE` | $5.00 | Penny stock filter |
| `MIN_AVG_VOLUME` | 500,000 | Liquidity filter |

---

## Watchlists

Located in `watchlists/`:

| File | Tickers | Theme |
|------|---------|-------|
| `ai_chips.txt` | 38 | AI chips, memory, servers, power |
| `ai_ecosystem.txt` | 90 | Full AI + Robotics + Quantum + Space |
| `cyber.txt` | 19 | Cybersecurity: platforms, identity, SecOps, data security |
| `biotech.txt` | 33 | Large-cap biotech, obesity, pipeline names, genomics/diagnostics |
| `energy.txt` | 37 | Majors, E&P, gas/LNG, refiners, services, midstream, tankers |
| `custom.txt` | any | Your own list — edit freely |

For the full S&P 500, omit `--watchlist-only`. The universe comes from the
NASDAQ trader listing, with Wikipedia's S&P 500 list as a fallback.

Add tickers with `# comment` explaining why. One per line.

A `# benchmark: CIBR, HACK, BUG` header names the theme ETFs. The first ETF is
the theme benchmark for the watchlist check. All of them seed the
"trending but missing" suggestions.

### Sector rotation scan (runs before every run)

```bash
python main.py sectors                                # scan only
python main.py run --stocks-only --skip-sector-scan   # skip it
```

The scan covers about 60 ETFs: GICS sectors, sub-themes, commodities and countries. It sorts them into three buckets:

1. **Trending**: beats SPY over 3mo and 6mo, and trades above the 200 SMA.
2. **Emerging**: beats SPY this month, the lead is accelerating, and the price is rising, but it is not yet a 6-month leader.
3. **Breaking**: below the 200 SMA and behind SPY over 3mo and 6mo.

The scan shows what is moving, not why. For the why, read the news yourself, or use an AI assistant with web search (see `AGENTS.md`).

A `# sector_cap: off` header lifts the 2-per-sector cap for single-theme lists.

### Watchlist check (runs before every `--watchlist` run)

```bash
python main.py check-watchlist cyber                  # check only
python main.py run --watchlist cyber --watchlist-only --stocks-only --skip-watchlist-check
```

The check is advisory and never edits the file. It reports:

1. **Market**: the regime. If it is not Risk-On, halve position sizes.
2. **Theme**: the benchmark ETF vs SPY, rated Leading, Emerging, Fading, Lagging or Broken.
3. **Tickers**: each ticker gets a status.
   - `REMOVE`: no price data, or the ticker fails the price or volume filter.
   - `BENCH`: the ticker is below its 50 and 200 SMA and trails the theme ETF by more than 10% over 3 months.
   - `WATCH`: RSI is above 75, the price is more than 50% above the 200 SMA, earnings are within 7 days, or the news is negative.
   - `KEEP`: none of the above.
4. **Trending but missing**: benchmark ETF holdings that are not on the list and are beating the theme.

The check saves results to `output/watchlist_check_<name>_YYYY-MM-DD.csv`.

---

## Output Files

All saved to `output/YYYY-MM-DD/` (named by run date):

| File | Contents |
|------|----------|
| `stock_intraday_YYYY-MM-DD.csv` | 13-col intraday picks |
| `stock_swinglong_YYYY-MM-DD.csv` | 13-col swing/long picks |
| `stock_longterm_YYYY-MM-DD.csv` | 13-col long-term picks |
| `top5_picks_YYYY-MM-DD.json` | Full machine-readable output |
| `full_rankings_YYYY-MM-DD.csv` | All 50 candidates ranked |
| `day_bullish_YYYY-MM-DD.csv` | Day-trade CALL options picks |
| `day_bearish_YYYY-MM-DD.csv` | Bearish PUT options picks |
| `momentum_continuation_YYYY-MM-DD.csv` | RSI>70 trending picks |
| `report_YYYY-MM-DD.txt` | Human-readable text report |

SQLite tracker: `data/tracker.db` (paper trades + recommendation history).

---

## Manual Chart Workflow

The pipeline finds setups; you confirm them on a chart before entering. Recommended free tools:

1. **TradingView** (free) — set up daily/15min/5min layout with RSI(14), MACD(12,26,9), SMA(20/50/200), Volume, VWAP
2. **ThinkOrSwim paperMoney** — more pro features, requires Schwab login
3. **Finviz** — screening + heatmaps

For each pick:
- Daily chart: confirm RSI < 80, MACD histogram positive, price above 20 SMA
- 15m: look for breakout above premarket high OR pullback to VWAP
- 5m: enter on green candle with above-average volume
- Stop: pipeline's `Day Stop ↓` (1× ATR)
- Target: pipeline's `Day Target ↑` (1.5× ATR)

---

## Hard Rules (don't break)

1. **Options direction match**: `direction="bullish"` → CALLs only, `direction="bearish"` → PUTs only. Mixing caused a paper loss on DDOG (May 2026).
2. **Column lock**: 13 columns per table (Long-Term: 14, including the user-approved `Target Src`). Don't add/remove/rename without explicit approval.
3. **No global pip installs**: always use `.venv`.
4. **`--watchlist-only` clears cache** to prevent contamination.
5. **Earnings flag normalisation**: always `.date()` both sides before comparison.

---

## How To Read This Tool — Senior Analyst Framing

Think of this pipeline as a **junior quant on your desk**. It scans the tape every
morning and flags the setups. Your job — the senior trader — is to:

1. **Read the macro first.** Regime, VIX, sector benchmarks. If the tape is wrong,
   no setup matters.
2. **Walk the 3 tables with skepticism.** Look for:
   - Setups that line up with the macro (long bias in risk-on, short bias in risk-off)
   - Pipeline-vs-analyst target disagreements (the model leaning bullish past
     where coverage sits = caution)
   - Stretched-momentum names: RSI > 75 + price > 50% above 200 SMA = patience
   - Earnings flags ⚠ inside option expiry = IV crush risk
3. **Size before you enter.** ATR-based — the "Why" column tells you 1–5% of
   account. Veterans size first.
4. **Define invalidation BEFORE you click buy.** Day Stop / Swing Stop are the
   lines. Below = thesis broken, exit without ego.
5. **Track every paper trade.** The tracker DB is for honest self-review.
   "I won because I was right" vs "I won because the market was generous"
   are different things.

### What this tool is

- A deterministic quantitative screener combining yfinance OHLCV + ta indicators +
  VADER sentiment + yfinance fundamentals
- A consistent way to surface setups you can validate manually
- A teaching tool for paper traders learning the craft

### What it isn't

- Not financial advice
- Not an oracle — RSI > 70 is "caution", star ratings are quantitative summaries
- Not a replacement for chart reading, position management, or risk discipline

### The trader's mindset

> "The screener finds the names. The chart confirms the setup. Position sizing
> protects the account. Discipline keeps you in the game long enough to learn."

If you don't understand a signal, look it up before risking capital (paper or
real). Every output of this tool is an opportunity to evaluate your judgement
against the data.

---

## File Map

| File | Role |
|------|------|
| `main.py` | CLI entry, pipeline orchestrator, enrichment helpers |
| `ingest.py` | Fetch tickers, price data, news, fundamentals |
| `technicals.py` | Per-stock indicator scoring (RSI, MACD, SMA, ATR, volume) |
| `sentiment.py` | VADER sentiment on news headlines + market RSS |
| `scorer.py` | Ranking, dedup, sector cap, pick selection (all 3 stock lists) |
| `targets.py` | Price targets, stop losses, trade-style ratings (★) |
| `options.py` | Options chain fetch + best-play selection |
| `report.py` | All table renderers + CSV/JSON/text file writers |
| `macro.py` | SPY/VIX/DXY regime panel + thematic ETF benchmarks |
| `sector_scan.py` | Pre-run sector rotation scan (trending / emerging / breaking themes) |
| `journal.py` | Research journal: logs the scan's calls and grades them against real returns |
| `evolve.py` | Self-improving loop for the technical weights, with a holdout gate |
| `params/scoring.json` | Technical sub-weights (loaded by `config.py`, changed only by `evolve --apply`) |
| `watchlist_check.py` | Pre-run watchlist health check (market, theme, technicals, earnings, news, trending) |
| `backtest.py` | Replay saved recs vs actual prices + SPY benchmark |
| `tracker.py` | SQLite paper-trade log + recommendation history |
| `config.py` | All tuneable constants |
| `watchlists/*.txt` | Themed ticker lists |
| `AGENTS.md` | Onboarding guide for AI assistants |
| `tests/` | Unit tests (no network) — run with `pytest` |
| `pyproject.toml`, `uv.lock` | Dependencies (uv). `requirements.txt` is kept for `setup.sh` |

---

## Daily Workflow

```bash
# 1. Run pipeline (3 stock tables, fresh data)
cd stock-research && source .venv/bin/activate
python main.py run --no-cache --watchlist ai_ecosystem --watchlist-only --stocks-only

# 2. Open TradingView, validate each pick visually

# 3. Paper-trade your top conviction picks
python main.py track buy <TICKER> --price <ENTRY> --shares <N>

# 4. End of day, log exits
python main.py track sell <TICKER> --price <EXIT>

# 5. Weekly: backtest to see what worked
python main.py backtest --days 5 --since 30
```

---

## Data Sources and Disclaimer

- Market data, news and fundamentals come from Yahoo Finance through the
  unofficial `yfinance` library. Yahoo's terms allow personal use only. Check
  them before you use this tool for anything else. Data can be delayed,
  incomplete or wrong. The watchlist check exists partly to catch bad tickers.
- The tool writes your paper trades to `data/paper_trades.db` and your picks to
  `output/`. Both are git-ignored. Don't commit them.
- **This is not financial advice.** It is an educational screener. Nothing it
  prints is a recommendation to buy or sell. You are responsible for your own
  trades. See the no-warranty terms in [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `uv run pytest` and
`uv run ruff check .` before you open a PR.

## License

[MIT](LICENSE)
