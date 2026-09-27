"""Structural hypotheses: scoring variants, the per-list gate and the cross-list rule. No network."""
import numpy as np
import pandas as pd

import config
import evolve
import experiments

COMPONENTS = evolve.COMPONENTS


def _features(effect: str | None, n_dates: int = 120, n_tickers: int = 30, seed: int = 1) -> pd.DataFrame:
    """
    Synthetic replay rows with raw fields. `effect`:
      "stretch"  → names far above their 200 SMA underperform (H1 should win)
      "chase"    → big 5-day gainers underperform (H2 should win)
      None       → forward returns are pure noise
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2024-01-05", periods=n_dates, freq="W-FRI"):
        for t in range(n_tickers):
            sma200_pct = rng.uniform(-30, 120)
            ret5d      = rng.uniform(-10, 10)
            pos52      = rng.uniform(0, 1)
            scores     = {f"{c}_score": rng.uniform(0, 100) for c in COMPONENTS}
            scores["momentum_score"] = max(0, min(100, pos52 * 60 + min(max(ret5d * 4, -20), 40)))
            fwd = rng.normal(0, 2)
            if effect == "stretch":
                fwd -= max(0, sma200_pct - 25) / 10
            if effect == "chase":
                fwd -= ret5d / 2
            rows.append({"date": d, "ticker": f"T{t}", **scores, "sma200_pct": sma200_pct,
                         "ret5d_pct": ret5d, "week52_pos": pos52, "fwd_ret": fwd})
    return pd.DataFrame(rows)


W = config.DEFAULT_TECH_WEIGHTS


# ── Scoring variants ─────────────────────────────────────────────────────────

def test_every_hypothesis_scores_every_row() -> None:
    feats = _features(None, n_dates=5)
    for name, h in experiments.HYPOTHESES.items():
        s = h["score"](feats, W)
        assert len(s) == len(feats) and s.notna().all(), name


def test_stretch_penalty_only_hits_names_beyond_the_threshold() -> None:
    feats = _features(None, n_dates=3)
    base  = evolve._score(feats, W)
    h1    = experiments.HYPOTHESES["H1_stretch_penalty"]["score"](feats, W)
    calm  = feats.sma200_pct <= experiments.STRETCH_FROM
    assert (h1[calm] == base[calm]).all()
    assert (h1[~calm] < base[~calm]).all()


def test_mean_reversion_control_inverts_the_ranking() -> None:
    feats = _features(None, n_dates=3)
    h3 = experiments.HYPOTHESES["H3_mean_reversion"]["score"](feats, W)
    assert (h3 == -evolve._score(feats, W)).all()


# ── Per-list comparison ──────────────────────────────────────────────────────

def test_compare_passes_the_hypothesis_matching_the_effect() -> None:
    res = experiments.compare(_features("stretch"), "H1_stretch_penalty", W)
    assert res["verdict"] == "PASS" and res["holdout_gain"] > 0


def test_compare_fails_on_noise() -> None:
    res = experiments.compare(_features(None, seed=5), "H1_stretch_penalty", W)
    assert res["verdict"] != "PASS"


def test_no_chase_wins_when_short_term_gainers_fade() -> None:
    res = experiments.compare(_features("chase"), "H2_no_short_term_chase", W)
    assert res["verdict"] == "PASS"


# ── Cross-list rule ──────────────────────────────────────────────────────────

def test_adoption_needs_three_of_four_lists() -> None:
    def results(n_pass: int) -> list[dict]:
        return [{"verdict": "PASS" if i < n_pass else "FAIL"} for i in range(4)]
    assert experiments.overall(results(3)) == "ADOPT_CANDIDATE"
    assert experiments.overall(results(2)) == "REJECT"
    assert experiments.overall([{"verdict": "PASS"}]) == "REJECT"   # one list is never enough


# ── Regime switches (R1, R2) ─────────────────────────────────────────────────

def test_breadth_switch_uses_penalty_only_on_narrow_dates() -> None:
    feats = _features(None, n_dates=4)
    feats.loc[feats.date == feats.date.min(), "sma200_pct"] = -10.0   # date 1: every name below 200 SMA
    feats.loc[feats.date != feats.date.min(), "sma200_pct"] = 80.0    # others: broad and stretched
    r1   = experiments.HYPOTHESES["R1_breadth_switch"]["score"](feats, W)
    base = evolve._score(feats, W)
    h1   = experiments.HYPOTHESES["H1_stretch_penalty"]["score"](feats, W)
    narrow = feats.date == feats.date.min()
    assert (r1[narrow] == h1[narrow]).all()       # narrow → penalty (no-op here, all below +25%)
    assert (r1[~narrow] == base[~narrow]).all()   # broad → current score


def test_factor_momentum_switch_has_no_look_ahead() -> None:
    # Changing FUTURE forward returns must never change a PAST date's score.
    feats = _features("stretch", n_dates=40)
    fn    = experiments.HYPOTHESES["R2_factor_momentum_switch"]["score"]
    before = fn(feats, W)
    cut    = sorted(feats.date.unique())[25]
    shocked = feats.copy()
    shocked.loc[shocked.date >= cut, "fwd_ret"] *= -5
    after = fn(shocked, W)
    past = feats.date < cut
    assert (before[past] == after[past]).all()


def test_factor_momentum_switch_flips_after_the_score_has_been_losing() -> None:
    # Make the current score's top picks lose on every early date: once that
    # track record is known (lagged), the switch must invert the ranking.
    feats = _features(None, n_dates=40)
    base  = evolve._score(feats, W)
    feats["fwd_ret"] = -(base - base.mean()) / 10
    r2 = experiments.HYPOTHESES["R2_factor_momentum_switch"]["score"](feats, W)
    late = feats.date > sorted(feats.date.unique())[experiments.FACTOR_LOOKBACK + experiments.FACTOR_LAG + 1]
    assert (r2[late] == -base[late]).all()


def test_compare_scores_on_the_full_history_before_splitting() -> None:
    # Regime features need history that crosses the train/holdout boundary.
    res = experiments.compare(_features("stretch", n_dates=120), "R2_factor_momentum_switch", W)
    assert res["n_holdout"] > 0 and "holdout_gain" in res


def test_result_file_name_identifies_the_lists() -> None:
    # Regression: an sp500 run overwrote the 4-list run's CSV (same date + period).
    a = experiments.result_path(["ai_ecosystem", "cyber", "biotech", "energy"], "5y")
    b = experiments.result_path(["sp500"], "5y")
    assert a != b and "sp500" in b.name and b.name.endswith("_5y.csv")
