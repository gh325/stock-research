"""
experiments.py — Structural hypotheses for the technical score.

evolve.py tunes the weights of the existing score. This module tests changes
to the score's *shape*: a penalty, a changed component, an inverted ranking.
They are pre-registered, and their settings are fixed in this file before any
result is seen. Tuning a setting after seeing the numbers is the overfitting
the gate exists to catch.

Each hypothesis is compared with the current score on the same replay
(evolve.build_features):

  per list  PASS when it beats the current score on train AND on the recent
            holdout, with a paired holdout t >= evolve.MIN_HOLDOUT_T
  overall   ADOPT_CANDIDATE only if it passes on at least 3 of 4 lists
            (MIN_PASS_FRAC), which guards against a fix that fits one list

A candidate is not switched on automatically. Adopting it means a reviewed
code change that puts it behind a flag in params/scoring.json.

  python main.py experiment                      # all hypotheses, 4 watchlists
  python main.py experiment --watchlists cyber,energy
"""
import logging
from datetime import date

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

import evolve
from config import OUTPUT_DIR

logger  = logging.getLogger(__name__)
console = Console(width=160)

DEFAULT_WATCHLISTS = ["ai_ecosystem", "cyber", "biotech", "energy"]
MIN_PASS_FRAC      = 0.75   # pass on >= 3 of 4 lists
MIN_LISTS          = 3      # never adopt on fewer lists than this

# Pre-registered settings. Do not tune these to the results.
STRETCH_FROM    = 25.0      # H1: penalty starts at +25% above the 200 SMA
STRETCH_PER_PCT = 0.5       # H1: points removed per % beyond that


# ── Hypotheses (pure) ────────────────────────────────────────────────────────

def _h1_stretch_penalty(f: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    """Overbought names mean-revert: penalise distance above the 200 SMA."""
    excess = (f["sma200_pct"] - STRETCH_FROM).clip(lower=0)
    return evolve._score(f, w) - STRETCH_PER_PCT * excess


def _h2_no_short_term_chase(f: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    """Rebuild the momentum component from the 52-week position only, with no 5-day-return bonus."""
    return evolve._score(f.assign(momentum_score=f["week52_pos"] * 100), w)


def _h3_mean_reversion(f: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    """Control: the current ranking inverted. If this wins, the momentum premise is wrong here."""
    return -evolve._score(f, w)


HYPOTHESES = {
    "H1_stretch_penalty":     {"desc": f"−{STRETCH_PER_PCT} pt per % above +{STRETCH_FROM:.0f}% over 200 SMA",
                               "score": _h1_stretch_penalty},
    "H2_no_short_term_chase": {"desc": "momentum = 52-week position only (drop 5-day return)",
                               "score": _h2_no_short_term_chase},
    "H3_mean_reversion":      {"desc": "control: invert the current ranking",
                               "score": _h3_mean_reversion},
}


# ── Comparison (pure) ────────────────────────────────────────────────────────

def compare(features: pd.DataFrame, name: str, weights: dict[str, float]) -> dict:
    """One hypothesis versus the current score on one replay table."""
    train, hold = evolve.split(features)
    fn = HYPOTHESES[name]["score"]
    res: dict = {"hypothesis": name, "n_train": train.date.nunique(), "n_holdout": hold.date.nunique()}
    if res["n_train"] < evolve.MIN_TRAIN_DATES or res["n_holdout"] < evolve.MIN_HOLDOUT_DATES:
        return {**res, "verdict": "TOO_LITTLE_DATA", "train_gain": 0.0, "holdout_gain": 0.0,
                "holdout_t": 0.0, "holdout_excess": 0.0, "baseline_holdout_excess": 0.0}

    def ex(part: pd.DataFrame, hyp: bool) -> pd.Series:
        return evolve.per_date_excess(part, weights, score=fn(part, weights) if hyp else None)

    train_gain = float((ex(train, True) - ex(train, False)).mean())
    diff = (ex(hold, True) - ex(hold, False)).dropna()
    sd   = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    t    = float(diff.mean() / (sd / np.sqrt(len(diff)))) if sd > 0 else 0.0
    passed = train_gain > 0 and diff.mean() > 0 and t >= evolve.MIN_HOLDOUT_T
    return {**res, "train_gain": train_gain, "holdout_gain": float(diff.mean()), "holdout_t": t,
            "holdout_excess": float(ex(hold, True).mean()),
            "baseline_holdout_excess": float(ex(hold, False).mean()),
            "verdict": "PASS" if passed else "FAIL"}


def overall(per_list: list[dict]) -> str:
    """Cross-list rule: a hypothesis must hold on most lists, and on at least MIN_LISTS."""
    n_pass = sum(r["verdict"] == "PASS" for r in per_list)
    if len(per_list) >= MIN_LISTS and n_pass >= max(MIN_LISTS, int(np.ceil(MIN_PASS_FRAC * len(per_list)))):
        return "ADOPT_CANDIDATE"
    return "REJECT"


# ── Runner (network) ─────────────────────────────────────────────────────────

def run(watchlists: list[str] | None = None, horizon: int = 10, step: int = 5) -> pd.DataFrame:
    import ingest
    from config import load_tech_weights

    watchlists = watchlists or DEFAULT_WATCHLISTS
    weights    = load_tech_weights()
    rows = []
    for wl in watchlists:
        console.print(f"[bold]Replaying {wl}[/bold] …")
        feats = evolve.build_features(ingest.load_watchlist(wl), horizon=horizon, step=step)
        for name in HYPOTHESES:
            rows.append({"watchlist": wl, **compare(feats, name, weights)})
    df = pd.DataFrame(rows)
    render(df, horizon)
    path = OUTPUT_DIR / f"experiment_{date.today().isoformat()}.csv"
    df.round(4).to_csv(path, index=False)
    console.print(f"  [dim]Saved {path.name}[/dim]\n")
    return df


def render(df: pd.DataFrame, horizon: int) -> None:
    console.rule(f"[bold]Structural experiments vs current score ({horizon}d forward)")
    lists = list(dict.fromkeys(df.watchlist))
    tbl = Table(box=box.SIMPLE_HEAVY, show_edge=False)
    tbl.add_column("Hypothesis")
    tbl.add_column("Change")
    for wl in lists:
        tbl.add_column(f"{wl}\nholdout gain (t)", justify="right")
    tbl.add_column("Overall")
    for name, h in HYPOTHESES.items():
        sub  = df[df.hypothesis == name]
        cells = []
        for wl in lists:
            r = sub[sub.watchlist == wl].iloc[0]
            color = "green" if r.verdict == "PASS" else "red"
            cells.append(f"[{color}]{r.holdout_gain:+.2f}% ({r.holdout_t:+.1f})[/{color}]")
        verdict = overall(sub.to_dict("records"))
        vcol = "green" if verdict == "ADOPT_CANDIDATE" else "red"
        tbl.add_row(name, h["desc"], *cells, f"[{vcol}]{verdict}[/{vcol}]")
    console.print(tbl)
    console.print(f"  [dim]Gain = the hypothesis's top-5 excess minus the current score's, per {horizon}-day "
                  f"period, on the recent holdout. PASS needs train gain > 0 and holdout t ≥ "
                  f"{evolve.MIN_HOLDOUT_T}. ADOPT needs a PASS on ≥ 3 of 4 lists.[/dim]")
