"""Ranking, gates and target logic. Pure functions only — no network."""
import numpy as np

import main
import scorer
import targets
import technicals
from conftest import make_ohlcv


def _candidates() -> list[dict]:
    # Swing order follows combined_score (A first); long-term order follows
    # technical_score (D first), so the two tables rank differently.
    return [
        {"ticker": t, "sector": t, "combined_score": 90 - i,
         "technical_score": 50 + i * 10, "trend_score": 75, "vol_ratio_long": 1.0}
        for i, t in enumerate(["A", "B", "C", "D"])
    ]


# ── Rank isolation across tables ─────────────────────────────────────────────

def test_ranks_are_independent_across_tables() -> None:
    # Regression: a ticker in both tables shared one dict, so the long-term
    # rank overwrote the swing rank (two #2 rows, no #1).
    cands = _candidates()
    swing = scorer.select_top_picks(cands, n=4, max_per_sector=2)
    long_ = scorer.select_long_term_picks(cands, n=4, max_per_sector=2)
    assert [p["rank"] for p in swing] == [1, 2, 3, 4]
    assert [p["ticker"] for p in swing] == ["A", "B", "C", "D"]
    assert long_[0]["ticker"] == "D" and long_[0]["rank"] == 1


def test_sector_cap_none_returns_full_list() -> None:
    cands = [dict(c, sector="Tech") for c in _candidates()]
    assert len(scorer.select_top_picks(cands, n=4, max_per_sector=2)) == 2
    assert len(scorer.select_top_picks(cands, n=4, max_per_sector=None)) == 4


# ── Long-term profitability gate ─────────────────────────────────────────────

def test_profit_gate_rejects_negative_margin_even_with_forward_pe() -> None:
    assert scorer._has_real_fundamentals({"forwardPE": 208, "profitMargins": -0.082}) is False


def test_profit_gate_rejects_zero_margin() -> None:
    assert scorer._has_real_fundamentals({"forwardPE": 20, "profitMargins": 0.0}) is False


def test_profit_gate_accepts_profitable() -> None:
    assert scorer._has_real_fundamentals({"forwardPE": 17, "profitMargins": 0.23}) is True


def test_profit_gate_missing_margin_falls_back_to_pe() -> None:
    assert scorer._has_real_fundamentals({"trailingPE": 12}) is True
    assert scorer._has_real_fundamentals({"forwardPE": -8}) is False
    assert scorer._has_real_fundamentals({}) is False


# ── "Why" text ───────────────────────────────────────────────────────────────

def test_why_ranked_uses_analyst_vs_price_not_long_target() -> None:
    # Regression: CRWD showed "analyst target $235.67 (+15%)" at $252.
    why = main._why_ranked({"targets": {
        "analyst_mean": 235.67, "current_price": 252.13, "long_target_pct": 15.0,
    }})
    assert "analyst target $235.67 (-7%)" in why


# ── 200-day SMA ──────────────────────────────────────────────────────────────

def test_sma200_is_real_with_enough_history(uptrend) -> None:
    tech = technicals._analyze_one(uptrend)
    expected = float(uptrend["Close"].rolling(200).mean().iloc[-1])
    assert abs(tech["sma200"] - expected) < 0.01


def test_sma200_differs_from_sma50_on_a_trend(uptrend) -> None:
    # Regression: with 90 days of data, "sma200" silently equalled sma50.
    tech  = technicals._analyze_one(uptrend)
    sma50 = float(uptrend["Close"].rolling(50).mean().iloc[-1])
    assert abs(tech["sma200"] - sma50) > 1


def test_lookback_covers_200_trading_days() -> None:
    from config import LOOKBACK_DAYS
    assert LOOKBACK_DAYS * 252 / 365 >= 200


# ── Long-term target source label ────────────────────────────────────────────

def test_target_basis_analyst_when_consensus_above_price(uptrend) -> None:
    t = targets.calculate_price_targets(uptrend, {"targetMeanPrice": 250})
    assert t["long_target"] == 250 and t["long_target_basis"] == "Analyst"


def test_target_basis_flags_analysts_below_price(uptrend) -> None:
    t = targets.calculate_price_targets(uptrend, {"targetMeanPrice": 150})
    assert t["long_target_basis"].endswith("(analysts below)")
    assert t["long_target_basis"].startswith(("52w high", "+15% rule"))


def test_target_basis_flags_no_coverage() -> None:
    df = make_ohlcv(np.full(275, 100.0))
    t  = targets.calculate_price_targets(df, {})
    assert t["long_target_basis"] == "+15% rule (no analysts)"
    assert t["long_target"] == 115.0
