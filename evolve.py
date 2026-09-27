"""
evolve.py — Self-improving loop for the technical scoring weights.

The "autoresearch" pattern (Karpathy; ATLAS applies it to trading), adapted
to a deterministic screener:

  1. Replay history: for past dates, compute each ticker's technical
     sub-scores using only data up to that date, plus its forward return.
  2. Propose ONE targeted change: move STEP of weight from one component to
     another.
  3. Keep it if the training window improves; otherwise revert it.
  4. Gate: the final weights must beat the current weights on a held-out
     window the loop never saw, by a margin unlikely to be noise.
  5. Only `--apply` writes params/scoring.json, and it logs the change to
     params/evolution_log.jsonl. Commit it on a branch and review it like any
     other change. The loop never edits code.

What it does NOT learn: sentiment and fundamentals, which have no history
to replay. Known biases, so read the numbers conservatively:
  - Survivorship: the replay uses today's watchlist members.
  - Overlap: with step < horizon, forward windows overlap, which inflates t-stats.
  - No costs: it models no slippage or fees.
"""
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from config import BASE_DIR, MIN_AVG_VOLUME, MIN_PRICE, PARAMS_PATH

logger  = logging.getLogger(__name__)
console = Console(width=160)

LOG_PATH = BASE_DIR / "params" / "evolution_log.jsonl"

COMPONENTS        = ["rsi", "macd", "trend", "volume", "momentum"]
TOP_N             = 5      # picks per replay date, same as the swing table
STEP              = 0.05   # weight moved per proposed change
MIN_WEIGHT        = 0.05   # no component is ever switched off entirely
MAX_WEIGHT        = 0.50   # no single component may dominate
MIN_GAIN          = 0.05   # % excess return a change must add on train
MIN_HOLDOUT_T     = 2.0    # holdout gain ≥ 2 std errors: ~2.5% false-pass on pure noise (see tests)
TRAIN_FRAC        = 0.70
MIN_TRAIN_DATES   = 20
MIN_HOLDOUT_DATES = 10
WARMUP_ROWS       = 260    # history needed before the first replay date
MIN_TICKERS       = 2 * TOP_N   # a "top 5" of 5 tickers is the whole universe


# ── Evaluation (pure) ────────────────────────────────────────────────────────

def _score(features: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(features[f"{c}_score"] * weights[c] for c in COMPONENTS)


def per_date_excess(features: pd.DataFrame, weights: dict[str, float],
                    top_n: int = TOP_N) -> pd.Series:
    """
    For each date: mean forward return of the top-N picks minus the universe
    mean. Dates with fewer than MIN_TICKERS tickers are skipped, because their
    top N is most of the universe and the excess would be ~0 by construction.
    """
    counts = features.groupby("date")["ticker"].transform("nunique")
    df = features[counts >= max(MIN_TICKERS, 2 * top_n)]
    df = df.assign(_s=_score(df, weights))
    top  = (df.sort_values("_s", ascending=False).groupby("date").head(top_n)
              .groupby("date")["fwd_ret"].mean())
    univ = df.groupby("date")["fwd_ret"].mean()
    return (top - univ).dropna()


def evaluate(features: pd.DataFrame, weights: dict[str, float], top_n: int = TOP_N) -> dict:
    """Mean excess return (% per horizon), hit rate and t-stat over replay dates."""
    ex = per_date_excess(features, weights, top_n)
    n  = len(ex)
    sd = float(ex.std(ddof=1)) if n > 1 else 0.0
    return {
        "excess":   float(ex.mean()) if n else 0.0,
        "hit_rate": float((ex > 0).mean()) if n else 0.0,
        "t_stat":   float(ex.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0,
        "n_dates":  n,
    }


def split(features: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the holdout is always the most recent dates."""
    dates = np.sort(features["date"].unique())
    cut   = int(round(len(dates) * train_frac))
    return features[features.date.isin(dates[:cut])], features[features.date.isin(dates[cut:])]


# ── Mutation (pure) ──────────────────────────────────────────────────────────

def mutate(weights: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    """One targeted change: move STEP from one component to another, within bounds."""
    for _ in range(100):
        give, take = rng.choice(COMPONENTS, size=2, replace=False)
        if weights[give] - STEP >= MIN_WEIGHT - 1e-9 and weights[take] + STEP <= MAX_WEIGHT + 1e-9:
            cand = dict(weights)
            cand[give] = round(cand[give] - STEP, 4)
            cand[take] = round(cand[take] + STEP, 4)
            return cand
    return dict(weights)


# ── The loop ─────────────────────────────────────────────────────────────────

def run_evolution(features: pd.DataFrame, baseline: dict[str, float],
                  iterations: int = 40, seed: int = 0) -> dict:
    """Hill-climb on the training window, then gate once on the holdout."""
    train, hold = split(features)
    n_train, n_hold = train.date.nunique(), hold.date.nunique()
    result: dict = {"baseline": dict(baseline), "weights": dict(baseline), "attempts": [],
                    "n_train_dates": n_train, "n_holdout_dates": n_hold}
    if n_train < MIN_TRAIN_DATES or n_hold < MIN_HOLDOUT_DATES:
        result["verdict"] = "TOO_LITTLE_DATA"
        return result

    rng     = np.random.default_rng(seed)
    current = dict(baseline)
    cur     = evaluate(train, current)["excess"]
    for i in range(1, iterations + 1):
        cand  = mutate(current, rng)
        score = evaluate(train, cand)["excess"]
        kept  = score > cur + MIN_GAIN
        moved = {k: round(cand[k] - current[k], 4) for k in COMPONENTS if cand[k] != current[k]}
        result["attempts"].append({"i": i, "change": moved, "train_excess": score, "kept": kept})
        if kept:
            current, cur = cand, score

    # Holdout gate: paired per-date improvement of the new weights over the baseline
    diff = (per_date_excess(hold, current) - per_date_excess(hold, baseline)).dropna()
    sd   = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    t    = float(diff.mean() / (sd / np.sqrt(len(diff)))) if sd > 0 else 0.0
    result.update({
        "weights":          current,
        "train":            evaluate(train, current),
        "baseline_train":   evaluate(train, baseline),
        "holdout":          evaluate(hold, current),
        "baseline_holdout": evaluate(hold, baseline),
        "holdout_gain":     float(diff.mean()) if len(diff) else 0.0,
        "holdout_gain_t":   t,
    })
    if current == baseline:
        result["verdict"] = "NO_CHANGE"
    elif result["holdout_gain"] > 0 and t >= MIN_HOLDOUT_T:
        result["verdict"] = "PASS"
    else:
        result["verdict"] = "FAIL"      # improved on train only: overfitting
    return result


# ── Replay data (network) ────────────────────────────────────────────────────

def build_features(tickers: list[str], horizon: int = 10, step: int = 5,
                   period: str = "3y") -> pd.DataFrame:
    """
    Replay table: one row per (date, ticker) with the 5 technical sub-scores
    as of that date and the forward return (%) over `horizon` trading days.
    Each score uses only rows up to its date, so there is no look-ahead.
    """
    import yfinance as yf

    raw = yf.download(" ".join(tickers), period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    frames = {}
    for t in tickers:
        try:
            frames[t] = (raw.copy() if len(tickers) == 1 else raw[t]).dropna(subset=["Close"])
        except (KeyError, TypeError):
            continue
    return features_from_frames(frames, horizon=horizon, step=step)


def features_from_frames(frames: dict[str, pd.DataFrame], horizon: int = 10,
                         step: int = 5) -> pd.DataFrame:
    """
    Pure part of build_features. Every ticker is replayed on ONE shared
    calendar (the union of all trading days), so each replay date ranks the
    whole universe against each other. A ticker joins a date only once it has
    WARMUP_ROWS of its own history and `horizon` rows of future.
    """
    from technicals import _analyze_one

    calendar = sorted(set().union(*(df.index for df in frames.values()))) if frames else []
    dates    = calendar[WARMUP_ROWS:len(calendar) - horizon:step]
    rows = []
    for t, df in frames.items():
        close = df["Close"]
        for d in dates:
            if d not in df.index:
                continue
            pos = df.index.get_loc(d)
            if pos < WARMUP_ROWS or pos + horizon >= len(df):
                continue
            hist = df.iloc[max(0, pos - 300):pos + 1]
            # Same quality filter the live pipeline applies on the day
            if float(hist["Close"].iloc[-1]) < MIN_PRICE or \
               float(hist["Volume"].tail(20).mean()) < MIN_AVG_VOLUME:
                continue
            try:
                tech = _analyze_one(hist)
            except Exception as exc:
                logger.debug(f"{t} replay {df.index[pos]:%Y-%m-%d}: {exc}")
                continue
            rows.append({
                "date":   d,
                "ticker": t,
                **{f"{c}_score": tech[f"{c}_score"] for c in COMPONENTS},
                "fwd_ret": (float(close.iloc[pos + horizon]) / float(close.iloc[pos]) - 1) * 100,
            })
    return pd.DataFrame(rows)


# ── Apply + output ───────────────────────────────────────────────────────────

def apply(result: dict, context: dict, params_path=PARAMS_PATH, log_path=LOG_PATH) -> None:
    """Write the new weights and append an audit line. Refuses anything but PASS."""
    if result.get("verdict") != "PASS":
        raise ValueError(f"refusing to apply a {result.get('verdict')} result")
    data = json.loads(params_path.read_text()) if params_path.exists() else {}
    data["tech_weights"] = result["weights"]
    params_path.write_text(json.dumps(data, indent=2) + "\n")
    entry = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "context":          context,
        "old_weights":      result["baseline"],
        "new_weights":      result["weights"],
        "train":            result["train"],
        "holdout":          result["holdout"],
        "baseline_holdout": result["baseline_holdout"],
    }
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def render(result: dict, context: dict) -> None:
    console.rule(f"[bold]Evolve: technical weights ({context.get('watchlist')}, "
                 f"{context.get('horizon')}d forward)")
    console.print(f"  Replay dates: {result['n_train_dates']} train, {result['n_holdout_dates']} holdout")
    if result["verdict"] == "TOO_LITTLE_DATA":
        console.print("  [yellow]Too little history to learn from. Nothing changed.[/yellow]\n")
        return

    kept = [a for a in result["attempts"] if a["kept"]]
    console.print(f"  Attempts: {len(result['attempts'])}  kept {len(kept)}  "
                  f"reverted {len(result['attempts']) - len(kept)}")
    for a in kept:
        change = ", ".join(f"{k} {v:+.2f}" for k, v in a["change"].items())
        console.print(f"    #{a['i']:<3} kept   {change:<32} train excess {a['train_excess']:+.2f}%")

    tbl = Table(box=box.SIMPLE_HEAVY, show_edge=False)
    for col in ("Weights", *COMPONENTS, "Train excess", "Holdout excess", "Holdout hit rate"):
        tbl.add_column(col, justify="left" if col == "Weights" else "right")
    for label, w, tr, ho in (("current", result["baseline"], result["baseline_train"], result["baseline_holdout"]),
                             ("proposed", result["weights"], result["train"], result["holdout"])):
        tbl.add_row(label, *(f"{w[c]:.2f}" for c in COMPONENTS), f"{tr['excess']:+.2f}%",
                    f"{ho['excess']:+.2f}%", f"{ho['hit_rate']*100:.0f}%")
    console.print(tbl)
    console.print(f"  Holdout gain {result['holdout_gain']:+.2f}% per pick-period, "
                  f"t = {result['holdout_gain_t']:.2f} (needs ≥ {MIN_HOLDOUT_T})")

    verdicts = {
        "PASS":      "[green]PASS[/green]: better on unseen data. Re-run with --apply, then commit on a branch.",
        "FAIL":      "[red]FAIL[/red]: better on train only, which is overfitting. Current weights stay.",
        "NO_CHANGE": "[yellow]NO_CHANGE[/yellow]: no change beat the current weights on train.",
    }
    console.print(f"  Verdict: {verdicts[result['verdict']]}")
    console.print("  [dim]Excess = top-5 forward return minus universe average. Replay uses today's "
                  "watchlist (survivorship bias) and no trading costs.[/dim]\n")


def run(watchlist: str, horizon: int = 10, step: int = 5, iterations: int = 40,
        seed: int = 0, do_apply: bool = False) -> dict:
    """CLI entry: replay, evolve, report, and apply only on PASS + --apply."""
    import ingest
    from config import load_tech_weights

    context  = {"watchlist": watchlist, "horizon": horizon, "step": step,
                "iterations": iterations, "seed": seed}
    console.print(f"[bold]Replaying {watchlist}[/bold] ({horizon}d forward, every {step} days) …")
    features = build_features(ingest.load_watchlist(watchlist), horizon=horizon, step=step)
    result   = run_evolution(features, load_tech_weights(), iterations=iterations, seed=seed)
    render(result, context)
    if do_apply:
        if result["verdict"] == "PASS":
            apply(result, context)
            console.print(f"  [green]Applied[/green] to {PARAMS_PATH.name} and logged to {LOG_PATH.name}.")
        else:
            console.print("  [yellow]--apply ignored: only a PASS result is applied.[/yellow]")
    return result
