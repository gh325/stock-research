# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Regime-switch hypotheses and 5-year replay** (`experiment --period 5y`):
  - Registered before any 5-year result was seen:
    - R1 (breadth switch): uses H1 when fewer than 50% of the list trade above their 200 SMA.
    - R2 (factor-momentum switch): inverts the ranking when the current score lagged over its last 12 known replay dates. It is lagged 3 dates, and a test checks for no look-ahead.
  - `compare` now scores the full history before splitting, so regime features can cross the train/holdout boundary.
  - First 5-year run (2026-09-27; replay dates Oct 2022–Sep 2026, holdout from Jul 2025): all five hypotheses were REJECTED.
    - H3 (fade) lost badly on `ai_ecosystem` over the longer holdout (−3.81%, t = −2.8), so the 3-year "fade wins" result was the last 7 months only.
    - R2 was active on 20–66% of holdout dates and still added nothing.
    - R1 rarely triggered after mid-2025, except on cyber.
  - Conclusion: no tested structural change beats the current score robustly.
- **Structural experiments** (`experiments.py`, `python main.py experiment`):
  - Three pre-registered hypotheses, each compared with the current score on the evolve replay:
    - H1: a stretch penalty above +25% over the 200 SMA.
    - H2: momentum rebuilt from the 52-week position only.
    - H3: the ranking inverted, as a control.
  - PASS on a list needs both a train gain and a holdout gain with t ≥ 2. ADOPT needs a PASS on ≥ 3 of 4 lists.
  - The replay now also stores `sma200_pct`, `ret5d_pct` and `week52_pos`. `evolve.evaluate` and `evolve.per_date_excess` accept a precomputed `score`.
  - First run (2026-09-27): all three were REJECTED. H1 and H3 lost on 2023–Feb 2026 but won on Feb–Sep 2026, in every list for H3. That points to a momentum-to-mean-reversion regime flip rather than a stable structural edge.
- **Self-improving loop** (autoresearch pattern, built natively rather than on Hermes Agent):
  - `journal.py` and `python main.py journal backfill|grade|report`:
    - Every sector scan logs its calls. Trending and Emerging themes should beat SPY over 20 trading days, and Breaking themes should lag it.
    - The journal grades the calls and reports hit rates with 95% Wilson intervals.
    - Calls on the same ETF never overlap in time, so each one is independent evidence.
  - `evolve.py` and `python main.py evolve [--apply]`:
    - Replays about 2 years of technical sub-scores on one shared calendar and proposes one weight change at a time.
    - Keeps a change only if it improves train, then gates once on a recent holdout window (t ≥ 2, which falsely passes about 2.5% on noise and is tested).
    - `--apply` writes `params/scoring.json` and appends `params/evolution_log.jsonl`. It never edits code.
  - `params/scoring.json`: the technical sub-weights moved out of `config.py`. The values are unchanged.
- First real results (2026-09-27):
  - The scan's buckets show no proven edge: 52% / 51% / 48% hit rates, all within the coin-flip range.
  - On `ai_ecosystem`, the current top-5 technical picks lagged the watchlist by 2.66% per 10 days over Feb–Sep 2026.
  - `evolve` rejected the weights it found as overfit (holdout t = 0.48).
- **Open-source readiness:**
  - Added an MIT `LICENSE`, `CONTRIBUTING.md` and `.gitignore`. The ignore file keeps `data/`, `output/`, `logs/`, `.env` and archives out of git.
  - Added `pyproject.toml` and `uv.lock`, with a uv dev group for pytest and ruff.
  - Added CI in `.github/workflows/ci.yml`: ruff, compile check, unit tests and a CLI smoke test on Python 3.11 and 3.12, with actions pinned to SHAs.
  - Added 27 network-free unit tests in `tests/` that cover rank isolation, the profit gate, the analyst %, the real 200 SMA, target-source labels, the watchlist check and the sector-scan buckets.
  - README: added install steps, requirements, and a data-source and disclaimer section. Removed personal paths.
- **`Target Src` column in the Long-Term table** (user-approved column change): shows `Analyst` when the 1Y target is the consensus, or `52w high` / `+15% rule` with `(analysts below)` / `(no analysts)` when the tool projected it. Before this change the projection looked the same as an analyst target.
- **Sector rotation scan** (`sector_scan.py`, `python main.py sectors`): runs before every `run` in every mode; skip it with `--skip-sector-scan`. It ranks about 60 sector, theme and country ETFs by relative strength vs SPY, sorts them into Trending, Emerging and Breaking, and adds a headline-sentiment score. The results go to `output/sector_scan_YYYY-MM-DD.csv`.
- **`# sector_cap: off` watchlist header**: lifts the 2-per-sector cap for single-theme lists. Set on `cyber`, `biotech` and `energy`.
- **Sector watchlists**: `cyber.txt` (19), `biotech.txt` (33) and `energy.txt` (37), each with a `# benchmark:` ETF header. Added benchmark headers to `ai_ecosystem.txt` and `ai_chips.txt`.
- **Pre-run watchlist check** (`watchlist_check.py`, `python main.py check-watchlist <name>`):
  - Runs automatically before every `--watchlist` run; skip it with `--skip-watchlist-check`. The check is advisory and never blocks the run.
  - Market regime, theme ETF status vs SPY (Leading / Emerging / Fading / Lagging / Broken)
  - Per-ticker status: REMOVE (no data, fails liquidity), BENCH (broken trend and lagging the theme), WATCH (overbought, stretched, earnings within 7 days, negative news), KEEP
  - "Trending but missing" suggestions from the benchmark ETFs' top holdings
  - Uses 1 year of price data, so its 200 SMA is real. The main pipeline downloads 90 days and cannot compute one.
- **EPS beat/miss detection** (`ingest.fetch_earnings_beat_miss`):
  - Fetches most recent quarterly EPS actual vs estimate via yfinance `earnings_history`
  - Surfaces in `_why_ranked` when beat ≥ +5% (e.g. "EPS beat +10%") or miss ≤ −5%
  - Surfaces in `_generate_notes` with trader framing: catalyst backing the tape, or overhead supply warning
  - Addresses gap where VADER missed MRVL/HPE earnings-driven moves until the candles were already extended
- **Analyst target change velocity** (`tracker.get_analyst_target_velocity` / `store_analyst_target`):
  - New `analyst_targets` SQLite table stores consensus target per ticker per run date
  - Each run: query prior history first (velocity), then store current — avoids comparing today to today
  - Velocity appears in `_why_ranked` ("analyst target raised +12% (14d)") and `_generate_notes`
  - Noise floor: changes < 3% suppressed; significant raises/cuts ≥ ±5% are flagged
  - Builds value over time: the longer the pipeline runs, the richer the velocity signal

- `README.md` — full user-facing guide with daily workflow examples
- `CHANGELOG.md` — this file
- **Watchlist monitor mode** (`python main.py monitor --watchlist <name>`):
  - Monitors every ticker in a watchlist (e.g. all 92 `ai_ecosystem` names)
  - Computes ATR-based intraday day target and stop at startup from 20d daily data
  - New alert types: `BREAKOUT` (price crosses yesterday's high on >1.5× vol) and
    `DAY STOP` (price hits ATR stop level)
  - Compact heartbeat: shows top 5 gainers / top 5 losers per poll — not 92 lines
  - macOS notification + Ping sound on every alert

### Changed
- **Voice shift to senior analyst (20+ years on the desk)** across:
  - `AGENTS.md` — new "Voice — Senior Analyst" section with trader vocabulary,
    cycle-awareness, anti-narrative discipline, and hard rules retained
  - `~/.claude/skills/stock-research/skill.md` — voice instructions for the slash
    command match the senior-desk persona
  - `README.md` — "Honest Framing" replaced with senior-analyst framing
    (5-step desk workflow, trader mindset quote)
  - `main.py` — natural-language strings rewritten in trader voice:
    - `_generate_notes()` — frames setup quality, institutional footprint,
      analyst alignment, risk flags
    - `_volume_insight()` — distribution/accumulation/capitulation language
    - `_suggest_position_size()` — "treat as a trade, not a position" tone
    - `_generate_bearish_notes()` — "puts decay daily, respect the stop"
- Updated `AGENTS.md` to reflect the 3-stock-table architecture and `--stocks-only` flag
- Updated `main.py` docstring to highlight the recommended daily command

### Changed
- **Watchlist cleanup from the pre-run check**:
  - Removed CYBR (acquired), CTRA, ABB and BRKS (no price data), and ARGX and MDGL (volume under 500k). Each is commented out in its file with the reason.
  - Added PSNL (biotech), and CRGY and CLMT (energy): trending benchmark holdings.

### Fixed
- **`backtest` never found any picks**: it read `data/tracker.db`, but the tracker writes `config.DB_PATH` (`data/paper_trades.db`). It now reads `config.DB_PATH`, and a test guards it.
- Removed dead code that ruff flagged: 6 unused imports and 2 unused variables.
- **Long-term profitability gate**: a name now also needs `profitMargins > 0`. A positive forward P/E alone let NET (-8% margin), SAIL and TXG into the 1-year list.
- **200-day SMA was really the 50-day SMA**: `LOOKBACK_DAYS` was 90 calendar days (about 62 trading days), so `technicals` fell back to the 50 SMA for every "200 SMA" and computed the 52-week range over 62 days. It is now 400 calendar days (about 275 trading days). The price cache file name now includes the lookback (`prices_400d.pkl`), so a short-history cache is never reused. This changes trend scores, the long-term gate and all rankings.
- **Output files overwritten across watchlists**: when a run uses `--watchlist`, file names now carry the watchlist name after the date (`stock_swinglong_2026-09-26_cyber.csv`). `status` and the picks monitor now pick the newest file by modification time.
- **Themed watchlists returned only 2 swing and long-term picks** because of the 2-per-sector cap. Watchlists with `# sector_cap: off` now skip the cap.
- **ETF benchmark panel was always the AI ETFs**: the panel now uses the watchlist's `# benchmark:` ETFs plus SPY.
- **Analyst upside % in "Why" text** (`main._why_ranked`): now computed from the analyst mean target vs current price. It previously reused `long_target_pct`, which falls back to a 52-wk/15% projection when analysts sit below price, so CRWD showed "$235.67 (+15%)" while trading at $252.
- **Duplicate ranks across tables** (`scorer.select_top_picks`, `scorer.select_long_term_picks`): picks are now copied before `rank` is set. A ticker in both the swing and long-term tables shared one dict, so the long-term rank overwrote the swing rank (two #2 rows, no #1).
- **Long-term list polluted by unprofitable momentum micro-caps**
  (`scorer.select_long_term_picks` + `_has_real_fundamentals`):
  - The long-term "fundamentals-first" table only gated on trend (above 200 SMA),
    so pre-revenue micro-caps riding a news spike (IPWR, QTTB, AGEN, VEEE) landed
    in the year+ hold list with negative or missing P/E and technical-projection
    "targets" masquerading as analyst consensus
  - Added a positive-earnings gate: `select_long_term_picks` now accepts a
    `fundamentals_fetcher` and drops any candidate without a positive trailing OR
    forward P/E; `main.py` wires in `ingest.fetch_fundamentals`
  - Momentum micro-caps still surface in the swing table where they belong
- **Full-universe ticker fetch silently degrading to 503 names**
  (`ingest._fetch_nasdaq_tickers`):
  - The NASDAQ HTTPS mirror (`ftp.nasdaqtrader.com:443`) is blocked/timed-out on
    some networks, causing a silent fallback to the 503-name S&P 500 from
    Wikipedia — so "full universe" runs were really S&P-500-only runs
  - Added a plain-`ftp://` fallback (stdlib `urllib`) before the Wikipedia
    fallback; recovers the full ~6,900-name common-stock universe when HTTPS is
    blocked but FTP is reachable
  - Factored parsing into `_parse_nasdaq_traded()` shared by both paths

### Retained (anti-pump guardrails)
- No fabricated narratives ("smart money rotation", "seasonal pattern")
- No "buy this / sell this" — these are **reads**, not recommendations
- Always flag earnings risk, overbought RSI, stretched 200 SMA
- "Not financial advice" caveat preserved everywhere

## [1.2.0] - 2026-05-13

### Added
- **`--stocks-only` flag** on `run` subcommand. Renders only the 3 stock tables
  (intraday, swing/long, long-term) and suppresses the options/momentum/bearish
  tables. Recommended for daily workflow.
- **Long-Term Stocks table** — 5 picks ranked by trend + accumulation + fundamentals
  for 1-year+ holds. New scorer: `select_long_term_picks()`.
- New CSV writer: `stock_longterm_YYYY-MM-DD.csv`.

### Changed
- `print_summary()` now accepts a `stocks_only` boolean; default mode unchanged
  (renders all 6 tables).

## [1.1.0] - 2026-05-10

### Added
- **Two new stock-only tables**: intraday and swing/long, each with 13 columns
  matching the locked layout. Renders alongside the existing 3 options tables.
- `_save_csv_stock_intraday()` and `_save_csv_stock_swinglong()` writers.

### Changed
- `print_summary()` order is now: Intraday Stocks → Day-Trade CALLs →
  Swing/Long Stocks → Swing/Long CALLs → Momentum → Bearish PUTs.

## [1.0.0] - 2026-05-07

### Added
- Initial pipeline release.
- 5 tables: intraday CALLs, swing/long CALLs, bearish PUTs, momentum continuation.
- Scoring architecture: technicals + sentiment, sector cap, dedup, deep enrichment.
- Watchlists: `ai_chips.txt` (38), `ai_ecosystem.txt` (92), `sp500.txt`,
  `morning.txt`.
- Tracker DB for paper trades and recommendation history.
- Macro panel (SPY/VIX/DXY + thematic ETFs).
- Backtest replay with 5-day forward windows vs SPY.

### Fixed
- DDOG paper-loss incident (2026-05-07): added direction-matching to
  `pick_best_option()` so bullish picks get CALLs and bearish picks get PUTs.
