"""
sector_scan.py — Pre-run sector rotation scan.

Before picking stocks, a veteran asks where the money is going. This scan
ranks ~60 sector, theme and country ETFs by relative strength vs SPY and
sorts them into three buckets:

  Trending   sustained leadership: beats SPY over 3mo AND 6mo, above 200 SMA
  Emerging   an early turn: beats SPY this month, accelerating, price rising,
             but not yet a 6-month leader
  Breaking   below 200 SMA and lagging SPY over 3mo AND 6mo — avoid

Each listed theme also gets a headline-sentiment read (VADER on the ETF's
yfinance news). The scan explains *what* is moving; the "why" (news,
policy, macro) is left to a human or to the /stock-research skill's web
research step.

Runs before every `python main.py run` (skip with --skip-sector-scan) or
alone with `python main.py sectors`. Never blocks the run.
"""
import logging
from datetime import date

import pandas as pd
import yfinance as yf
from rich import box
from rich.console import Console
from rich.table import Table

from config import OUTPUT_DIR

logger  = logging.getLogger(__name__)
console = Console(width=160)   # wide enough that tables never wrap

# ETF → theme. Broad enough to catch rotation outside the AI watchlists.
THEME_ETFS: dict[str, str] = {
    # GICS sectors
    "XLK": "Tech", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLI": "Industrials", "XLY": "Consumer Disc", "XLP": "Consumer Staples",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Comm Services",
    # Tech sub-themes
    "SMH": "Semis", "IGV": "Software", "CIBR": "Cybersecurity", "SKYY": "Cloud",
    "QTUM": "Quantum", "BOTZ": "Robotics/AI",
    # Health care sub-themes
    "XBI": "Biotech (small/mid)", "IBB": "Biotech (large)", "ARKG": "Genomics",
    "IHI": "Med devices", "PPH": "Pharma",
    # Industrial / defense / infrastructure
    "ITA": "Aerospace/Defense", "UFO": "Space", "PAVE": "US infrastructure",
    "GRID": "Grid infra", "IYT": "Transports", "JETS": "Airlines",
    # Energy & power
    "XOP": "Oil E&P", "OIH": "Oil services", "URA": "Uranium", "NLR": "Nuclear power",
    "TAN": "Solar", "ICLN": "Clean energy",
    # Materials & commodities
    "GDX": "Gold miners", "SIL": "Silver miners", "COPX": "Copper miners",
    "XME": "Metals & mining", "REMX": "Rare earths", "LIT": "Lithium/battery",
    "MOO": "Agribusiness", "GLD": "Gold", "SLV": "Silver", "USO": "Crude oil",
    # Financial sub-themes
    "KRE": "Regional banks", "KBE": "Banks", "IAI": "Brokers", "IPAY": "Payments",
    "IBIT": "Bitcoin", "BLOK": "Blockchain equities",
    # Consumer / rates-sensitive
    "ITB": "Homebuilders", "XRT": "Retail", "IWM": "Small caps", "TLT": "Long bonds",
    # Countries
    "EWJ": "Japan", "EWT": "Taiwan", "EWY": "South Korea", "INDA": "India",
    "EWZ": "Brazil", "KWEB": "China internet", "EEM": "Emerging markets",
}

TOP_N         = 6   # rows shown per bucket
MIN_HEADLINES = 3   # fewer headlines = sentiment too thin to trust


# ── Metrics (pure — no network) ──────────────────────────────────────────────

def _ret(s: pd.Series, n: int) -> float | None:
    if len(s) <= n:
        return None
    return (float(s.iloc[-1]) / float(s.iloc[-1 - n]) - 1) * 100


def etf_metrics(close: pd.Series, volume: pd.Series, spy: pd.Series) -> dict | None:
    """Relative-strength and trend numbers for one ETF. None if too little history."""
    close = close.dropna()
    if len(close) < 200:
        return None
    rs    = close / spy.reindex(close.index).ffill()
    vol60 = float(volume.tail(60).mean())
    m = {
        "ret1m": _ret(close, 21), "ret3m": _ret(close, 63), "ret6m": _ret(close, 126),
        "rs1m":  _ret(rs, 21),    "rs3m":  _ret(rs, 63),    "rs6m":  _ret(rs, 126),
        "vs200": (float(close.iloc[-1]) / float(close.rolling(200).mean().iloc[-1]) - 1) * 100,
        "off_high": (float(close.iloc[-1]) / float(close.tail(252).max()) - 1) * 100,
        "vol20_60": float(volume.tail(20).mean()) / vol60 if vol60 > 0 else None,
    }
    if any(m[k] is None for k in ("ret1m", "rs1m", "rs3m", "rs6m")):
        return None
    # Acceleration: this month's RS vs the 3-month average monthly pace
    m["accel"] = m["rs1m"] - m["rs3m"] / 3
    return m


def classify(m: dict) -> str | None:
    """Trending / Emerging / Breaking, or None for the unremarkable middle."""
    if m["rs3m"] > 0 and m["rs6m"] > 0 and m["vs200"] > 0:
        return "Trending"
    if m["rs1m"] > 0 and m["accel"] > 0 and m["ret1m"] > 0 and m["rs6m"] <= 0:
        # ret1m > 0 matters: heavy volume on a falling price is distribution,
        # not an early turn.
        return "Emerging"
    if m["vs200"] < 0 and m["rs3m"] < 0 and m["rs6m"] < 0:
        return "Breaking"
    return None


def _flags(m: dict) -> str:
    notes = []
    if m["vs200"] > 50:
        notes.append("stretched (>50% over 200 SMA)")
    if m["off_high"] < -25:
        notes.append(f"{m['off_high']:.0f}% off high")
    if (m["vol20_60"] or 0) >= 1.3:
        notes.append(f"volume {m['vol20_60']:.1f}× 60d")
    return ", ".join(notes)


# ── Network ──────────────────────────────────────────────────────────────────

def _headline_sentiment(etfs: list[str]) -> dict[str, float | None]:
    """VADER score (0–100) of each ETF's recent yfinance headlines."""
    import ingest
    import sentiment

    scored = sentiment.score_sentiment_for_tickers(ingest.fetch_news_for_tickers(etfs))
    return {
        t: (s["sentiment_score"] if s.get("headline_count", 0) >= MIN_HEADLINES else None)
        for t, s in scored.items()
    }


def scan() -> dict:
    """Download, score and bucket every theme ETF."""
    tickers = list(THEME_ETFS) + ["SPY"]
    raw = yf.download(" ".join(tickers), period="400d", interval="1d",
                      group_by="ticker", auto_adjust=True, threads=True, progress=False)
    spy = raw["SPY"]["Close"].dropna()

    rows = []
    for t, theme in THEME_ETFS.items():
        try:
            m = etf_metrics(raw[t]["Close"], raw[t]["Volume"], spy)
        except KeyError:
            m = None
        if m is None:
            continue
        rows.append({"etf": t, "theme": theme, "bucket": classify(m), **m})
    df = pd.DataFrame(rows)

    buckets = {
        "Trending": df[df.bucket == "Trending"].assign(_k=lambda d: d.rs3m + d.rs6m)
                     .sort_values("_k", ascending=False).head(TOP_N),
        "Emerging": df[df.bucket == "Emerging"].sort_values("accel", ascending=False).head(TOP_N),
        "Breaking": df[df.bucket == "Breaking"].assign(_k=lambda d: d.rs3m + d.rs6m)
                     .sort_values("_k").head(TOP_N),
    }
    shown = [t for b in buckets.values() for t in b.etf]
    sent  = _headline_sentiment(shown)
    for name, b in buckets.items():
        buckets[name] = b.assign(news=b.etf.map(sent), flags=b.apply(_flags, axis=1))

    return {"all": df, "buckets": buckets,
            "spy": {"ret1m": _ret(spy, 21), "ret3m": _ret(spy, 63), "ret6m": _ret(spy, 126)}}


# ── Output ───────────────────────────────────────────────────────────────────

def render(result: dict) -> None:
    spy = result["spy"]
    console.rule("[bold]Sector Rotation Scan")
    console.print(f"  SPY 1mo {spy['ret1m']:+.1f}%  3mo {spy['ret3m']:+.1f}%  6mo {spy['ret6m']:+.1f}%"
                  "   (RS = ETF return minus SPY return)")
    colors = {"Trending": "green", "Emerging": "cyan", "Breaking": "red"}
    blurbs = {
        "Trending": "sustained leadership — buy pullbacks, not extensions",
        "Emerging": "early turn — watch for confirmation, small size",
        "Breaking": "below 200 SMA and lagging — avoid until a base forms",
    }
    for name, b in result["buckets"].items():
        if b.empty:
            continue
        tbl = Table(title=f"[{colors[name]}]{name}[/] — {blurbs[name]}",
                    box=box.SIMPLE_HEAVY, show_edge=False, title_justify="left")
        for col in ("ETF", "Theme", "1mo", "RS 1mo", "RS 3mo", "RS 6mo", "vs 200", "News", "Notes"):
            tbl.add_column(col, justify="left" if col in ("ETF", "Theme", "Notes") else "right")
        for r in b.itertuples():
            tbl.add_row(r.etf, r.theme, f"{r.ret1m:+.1f}%", f"{r.rs1m:+.1f}%", f"{r.rs3m:+.1f}%",
                        f"{r.rs6m:+.1f}%", f"{r.vs200:+.0f}%",
                        "—" if pd.isna(r.news) else f"{r.news:.0f}", r.flags)
        console.print(tbl)
    console.print("  [dim]News = VADER score of ETF headlines (0–100, 50 neutral). "
                  "Numbers show what is moving, not why.[/dim]\n")


def save(result: dict) -> str:
    path = OUTPUT_DIR / f"sector_scan_{date.today().isoformat()}.csv"
    result["all"].round(2).to_csv(path, index=False)
    return str(path)


def run_scan() -> dict | None:
    """scan + render + save. Never raises — a failed scan must not block a run."""
    try:
        result = scan()
        render(result)
        save(result)
        return result
    except Exception as exc:
        logger.warning(f"Sector scan failed: {exc}")
        return None
