"""
macro.py — Market-regime snapshot and thematic ETF benchmarks.

Produces context that sits above the picks tables:
  - SPY trend, distance to 200-SMA, recent returns (1d / 5d / 1mo / 3mo / YTD)
  - VIX level (fear gauge)
  - DXY (dollar strength)
  - Thematic ETFs for AI/chip/robotics context: SOXX, SMH, ROBO, BOTZ, AIQ
  - Regime classification: Risk-On / Caution / Risk-Off

A 20-year-veteran investor would never look at picks without first checking:
"What environment am I in?" — bull/bear, low-vol/high-vol, dollar regime.
"""
import logging
from datetime import datetime

import yfinance as yf

logger = logging.getLogger(__name__)

THEMATIC_ETFS = ["SOXX", "SMH", "ROBO", "BOTZ", "AIQ"]   # default (AI) panel

# Labels for the ETF panel. Watchlists pick their own ETFs via the
# `# benchmark:` header; unknown tickers render with a blank label.
ETF_THEMES = {
    "SPY":  "S&P 500 (the bar to beat)",
    "SOXX": "Semiconductors (broad)",
    "SMH":  "Semiconductors (cap-weighted)",
    "ROBO": "Robotics & automation",
    "BOTZ": "Robotics & AI",
    "AIQ":  "AI & technology",
    "CIBR": "Cybersecurity",
    "HACK": "Cybersecurity",
    "BUG":  "Cybersecurity (pure-play)",
    "XBI":  "Biotech (equal-weight, small/mid)",
    "IBB":  "Biotech (cap-weighted, large)",
    "ARKG": "Genomics",
    "XLE":  "Energy (majors)",
    "XOP":  "Oil & gas E&P / refiners",
    "OIH":  "Oil services",
}


def get_macro_snapshot() -> dict:
    """Fetch SPY/VIX/DXY indicators and classify the current regime."""
    try:
        # SPY: 1 year for 200-SMA + YTD computation
        spy = yf.Ticker("SPY").history(period="1y")
        if spy.empty:
            return {"available": False, "reason": "SPY data unavailable"}

        spy_close   = float(spy["Close"].iloc[-1])
        spy_sma200  = float(spy["Close"].rolling(200).mean().iloc[-1]) if len(spy) >= 200 else None
        spy_above   = spy_sma200 is None or spy_close > spy_sma200
        spy_dist200 = ((spy_close / spy_sma200 - 1) * 100) if spy_sma200 else None

        spy_1d  = ((spy_close / spy["Close"].iloc[-2] - 1) * 100) if len(spy) >= 2  else 0
        spy_5d  = ((spy_close / spy["Close"].iloc[-6] - 1) * 100) if len(spy) >= 6  else 0
        spy_1mo = ((spy_close / spy["Close"].iloc[-22] - 1) * 100) if len(spy) >= 22 else 0
        spy_3mo = ((spy_close / spy["Close"].iloc[-66] - 1) * 100) if len(spy) >= 66 else 0

        # YTD: from first trading day of current year
        current_year  = datetime.now().year
        ytd_slice     = spy[spy.index.year == current_year]
        spy_ytd       = (
            (spy_close / ytd_slice["Close"].iloc[0] - 1) * 100
            if not ytd_slice.empty else 0
        )

        # VIX (fear gauge)
        vix = yf.Ticker("^VIX").history(period="20d")
        vix_now    = float(vix["Close"].iloc[-1]) if not vix.empty else None
        vix_5d_avg = float(vix["Close"].tail(5).mean()) if len(vix) >= 5 else vix_now

        # DXY (dollar)
        dxy = yf.Ticker("DX-Y.NYB").history(period="20d")
        dxy_now = float(dxy["Close"].iloc[-1]) if not dxy.empty else None
        dxy_5d  = (
            (dxy_now / dxy["Close"].iloc[-6] - 1) * 100
            if dxy_now is not None and len(dxy) >= 6 else 0
        )

        # ── Regime classification ─────────────────────────────────────────
        # Risk-On  = SPY above 200-SMA AND VIX < 20
        # Risk-Off = SPY below 200-SMA OR VIX > 30
        # Caution  = anything in between
        if spy_above and (vix_now is None or vix_now < 20):
            regime, regime_color = "Risk-On",  "green"
        elif (vix_now and vix_now > 30) or (spy_sma200 and not spy_above):
            regime, regime_color = "Risk-Off", "red"
        else:
            regime, regime_color = "Caution",  "yellow"

        return {
            "available":      True,
            "regime":         regime,
            "regime_color":   regime_color,
            "spy_close":      round(spy_close, 2),
            "spy_sma200":     round(spy_sma200, 2) if spy_sma200 else None,
            "spy_above_200":  spy_above,
            "spy_dist_200_pct": round(spy_dist200, 2) if spy_dist200 is not None else None,
            "spy_1d":         round(spy_1d, 2),
            "spy_5d":         round(spy_5d, 2),
            "spy_1mo":        round(spy_1mo, 2),
            "spy_3mo":        round(spy_3mo, 2),
            "spy_ytd":        round(spy_ytd, 2),
            "vix":            round(vix_now, 2) if vix_now is not None else None,
            "vix_5d_avg":     round(vix_5d_avg, 2) if vix_5d_avg is not None else None,
            "dxy":            round(dxy_now, 2) if dxy_now is not None else None,
            "dxy_5d_pct":     round(dxy_5d, 2),
        }
    except Exception as exc:
        logger.warning(f"Macro snapshot fetch failed: {exc}")
        return {"available": False, "reason": str(exc)}


def get_thematic_etf_returns(etfs: list[str] | None = None) -> dict:
    """
    Recent returns for the thematic ETFs you'd actually buy for the theme
    if you didn't want to pick stocks. The benchmark every picks list
    should be honest about: 'are my picks beating these ETFs?'

    etfs: the watchlist's `# benchmark:` ETFs. SPY is always appended so
          the panel shows whether the theme itself beats the market.
          Defaults to the AI panel (THEMATIC_ETFS).
    """
    tickers = list(dict.fromkeys((etfs or THEMATIC_ETFS) + ["SPY"]))
    results: dict[str, dict] = {}
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(period="3mo")
            if df.empty or len(df) < 2:
                continue
            close = float(df["Close"].iloc[-1])
            ret_5d  = ((close / df["Close"].iloc[-6]  - 1) * 100) if len(df) >= 6  else 0
            ret_1mo = ((close / df["Close"].iloc[-22] - 1) * 100) if len(df) >= 22 else 0
            ret_3mo = ((close / df["Close"].iloc[0]   - 1) * 100)
            results[ticker] = {
                "close":   round(close, 2),
                "ret_5d":  round(ret_5d, 2),
                "ret_1mo": round(ret_1mo, 2),
                "ret_3mo": round(ret_3mo, 2),
            }
        except Exception as exc:
            logger.debug(f"{ticker} ETF fetch error: {exc}")
    return results
