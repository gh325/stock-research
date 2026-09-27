"""Evolve loop: evaluation, mutation and the holdout gate. No network."""
import json

import numpy as np
import pandas as pd
import pytest

import config
import evolve

COMPONENTS = list(config.DEFAULT_TECH_WEIGHTS)


def _features(signal: str | None, n_dates: int = 80, n_tickers: int = 30,
              seed: int = 1) -> pd.DataFrame:
    """
    Synthetic replay table. If `signal` names a component, that component's
    score drives forward returns; every other component is pure noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates, freq="W-FRI")
    rows = []
    for d in dates:
        for t in range(n_tickers):
            scores = {f"{c}_score": rng.uniform(0, 100) for c in COMPONENTS}
            noise  = rng.normal(0, 2)
            edge   = (scores[f"{signal}_score"] - 50) / 10 if signal else 0.0
            rows.append({"date": d, "ticker": f"T{t}", **scores, "fwd_ret": edge + noise})
    return pd.DataFrame(rows)


# ── Evaluation ───────────────────────────────────────────────────────────────

def test_evaluate_rewards_weight_on_the_predictive_component() -> None:
    feats = _features("trend")
    heavy = {c: (0.60 if c == "trend" else 0.10) for c in COMPONENTS}
    light = {c: (0.05 if c == "trend" else 0.2375) for c in COMPONENTS}
    assert evolve.evaluate(feats, heavy)["excess"] > evolve.evaluate(feats, light)["excess"]


def test_evaluate_reports_sample_counts() -> None:
    m = evolve.evaluate(_features(None, n_dates=12), config.DEFAULT_TECH_WEIGHTS)
    assert m["n_dates"] == 12 and 0 <= m["hit_rate"] <= 1


def test_split_is_chronological_and_disjoint() -> None:
    train, hold = evolve.split(_features(None, n_dates=50))
    assert train.date.max() < hold.date.min()
    assert train.date.nunique() == 35 and hold.date.nunique() == 15


def test_evaluate_skips_dates_with_too_few_tickers() -> None:
    # With <= TOP_N tickers the "top 5" is the whole universe: excess is 0 by
    # construction and would silently drag the hit rate down.
    thin = _features("trend", n_dates=5, n_tickers=4)
    assert evolve.evaluate(thin, config.DEFAULT_TECH_WEIGHTS)["n_dates"] == 0


def test_replay_puts_every_ticker_on_one_calendar() -> None:
    # Regression: each ticker stepped from its own first row, so tickers with
    # different history lengths landed on different dates (median 2 per date).
    from conftest import make_ohlcv
    long_  = make_ohlcv(np.linspace(100, 200, 400))
    short  = make_ohlcv(np.linspace(50, 80, 333))            # IPO'd later
    frames = {"LONG": long_, "SHORT": short, "ALSO": make_ohlcv(np.linspace(80, 90, 400))}
    feats  = evolve.features_from_frames(frames, horizon=10, step=5)
    per_date = feats.groupby("date").ticker.nunique()
    late = per_date[per_date.index >= short.index[evolve.WARMUP_ROWS]]
    assert (late == 3).all()                                   # all three share each date
    assert feats.date.isin(long_.index).all()
    # forward return really is `horizon` rows ahead, on the ticker's own index
    row = feats[feats.ticker == "LONG"].iloc[0]
    pos = long_.index.get_loc(row.date)
    expected = (long_.Close.iloc[pos + 10] / long_.Close.iloc[pos] - 1) * 100
    assert abs(row.fwd_ret - expected) < 1e-9


# ── Mutation ─────────────────────────────────────────────────────────────────

def test_mutation_is_one_transfer_that_keeps_bounds_and_sum() -> None:
    rng = np.random.default_rng(0)
    base = dict(config.DEFAULT_TECH_WEIGHTS)
    for _ in range(200):
        cand = evolve.mutate(base, rng)
        assert abs(sum(cand.values()) - 1) < 1e-9
        assert all(evolve.MIN_WEIGHT - 1e-9 <= v <= evolve.MAX_WEIGHT + 1e-9 for v in cand.values())
        changed = [k for k in base if abs(cand[k] - base[k]) > 1e-9]
        assert len(changed) == 2          # exactly one targeted change
        base = cand


# ── The loop and its gate ────────────────────────────────────────────────────

def test_loop_finds_the_signal_and_passes_holdout() -> None:
    result = evolve.run_evolution(_features("momentum", n_dates=120),
                                  config.DEFAULT_TECH_WEIGHTS, iterations=60, seed=0)
    assert result["verdict"] == "PASS"
    assert result["weights"]["momentum"] > config.DEFAULT_TECH_WEIGHTS["momentum"]
    assert any(a["kept"] for a in result["attempts"])
    assert any(not a["kept"] for a in result["attempts"])   # some changes are reverted


def test_loop_rejects_when_there_is_no_signal() -> None:
    # Pure noise: anything the loop "learns" on train is overfitting, and the
    # holdout gate must refuse it.
    result = evolve.run_evolution(_features(None, n_dates=120, seed=7),
                                  config.DEFAULT_TECH_WEIGHTS, iterations=60, seed=0)
    assert result["verdict"] != "PASS"


def test_gate_false_pass_rate_on_noise_is_low() -> None:
    # Calibration guard: across many pure-noise datasets the loop always
    # "improves" train, so only the holdout gate stands between noise and a
    # weight change. Loosening MIN_HOLDOUT_T should make this test fail.
    passes = sum(
        evolve.run_evolution(_features(None, n_dates=120, seed=100 + s),
                             config.DEFAULT_TECH_WEIGHTS, iterations=60, seed=s)["verdict"] == "PASS"
        for s in range(20)
    )
    assert passes <= 2      # at most 10%


def test_loop_refuses_too_little_data() -> None:
    result = evolve.run_evolution(_features("trend", n_dates=10),
                                  config.DEFAULT_TECH_WEIGHTS, iterations=5, seed=0)
    assert result["verdict"] == "TOO_LITTLE_DATA"


# ── Applying ─────────────────────────────────────────────────────────────────

def test_apply_writes_params_and_appends_log(tmp_path) -> None:
    params = tmp_path / "scoring.json"
    log    = tmp_path / "evolution_log.jsonl"
    params.write_text(json.dumps({"tech_weights": config.DEFAULT_TECH_WEIGHTS}))
    new = {**config.DEFAULT_TECH_WEIGHTS, "rsi": 0.15, "momentum": 0.20}
    result = {"verdict": "PASS", "weights": new, "baseline": config.DEFAULT_TECH_WEIGHTS,
              "train": {}, "holdout": {}, "baseline_holdout": {}}
    evolve.apply(result, {"watchlist": "cyber"}, params_path=params, log_path=log)
    assert config.load_tech_weights(params) == new
    entry = json.loads(log.read_text().splitlines()[-1])
    assert entry["new_weights"] == new and entry["context"]["watchlist"] == "cyber"


def test_apply_refuses_a_failed_run(tmp_path) -> None:
    with pytest.raises(ValueError):
        evolve.apply({"verdict": "FAIL"}, {}, params_path=tmp_path / "p.json",
                     log_path=tmp_path / "l.jsonl")


# ── Universes and the replay cache ───────────────────────────────────────────

def test_sp500_universe_uses_the_index_list(monkeypatch) -> None:
    import ingest
    monkeypatch.setattr(ingest, "_get_sp500_tickers", lambda: [f"T{i}" for i in range(503)])
    assert len(evolve.universe_tickers("sp500")) == 503
    assert evolve.universe_tickers("cyber") == ingest.load_watchlist("cyber")


def test_sp500_universe_refuses_the_silent_sp100_fallback(monkeypatch) -> None:
    # ingest falls back to a hard-coded S&P 100 when Wikipedia fails. A
    # replay labelled "sp500" must never quietly run on 100 names.
    import ingest
    monkeypatch.setattr(ingest, "_get_sp500_tickers", lambda: [f"T{i}" for i in range(100)])
    with pytest.raises(RuntimeError):
        evolve.universe_tickers("sp500")


def test_replay_cache_builds_once_per_day(monkeypatch, tmp_path) -> None:
    calls = []
    def fake_build(tickers, horizon, step, period):
        calls.append(period)
        return _features(None, n_dates=3)
    monkeypatch.setattr(evolve, "build_features", fake_build)
    monkeypatch.setattr(evolve, "universe_tickers", lambda name: ["A", "B"])
    monkeypatch.setattr(evolve, "CACHE_DIR", tmp_path)
    a = evolve.load_features("cyber", horizon=10, step=5, period="5y")
    b = evolve.load_features("cyber", horizon=10, step=5, period="5y")
    evolve.load_features("cyber", horizon=10, step=5, period="3y")   # different key
    assert calls == ["5y", "3y"]
    pd.testing.assert_frame_equal(a, b)
