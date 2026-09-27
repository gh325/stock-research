"""
options.py — Options chain analysis and trade recommendations.

Fetches real options data from yfinance and recommends the best call/put
for two timeframes: swing (3–5 weeks) and medium-term (2–3 months).

Options learner glossary:
  Call option  — gives you the RIGHT to BUY 100 shares at the strike price.
                 You profit when the stock RISES above your break-even.
  Put option   — gives you the RIGHT to SELL 100 shares at the strike price.
                 You profit when the stock FALLS below your break-even.
  Strike price — the price at which you can buy (call) or sell (put) the stock.
  Premium      — what you pay TODAY for the option (× 100 because 1 contract = 100 shares).
  Break-even   — for calls: strike + premium. For puts: strike − premium.
  IV (Implied Volatility) — the market's expectation of future volatility baked
                 into the option price. High IV = expensive options.
  DTE          — Days To Expiration. Closer expiry = faster time decay (theta).
  In-the-money — Call is ITM if stock price > strike. Put is ITM if stock < strike.
  Delta        — how much the option moves per $1 move in the stock (~0.50 = $0.50 per $1).

Risk warning: options can expire worthless, losing 100% of premium. Start small.
"""
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Target DTE windows for each timeframe
SWING_DTE   = (14, 55)     # 2–8 weeks out
MEDIUM_DTE  = (55, 120)    # 2–4 months out


def get_options_recommendations(
    ticker: str,
    current_price: float,
    swing_target: float,
    long_target: float,
) -> dict:
    """
    Fetch the options chain and return recommended call + put for each timeframe.

    Returns a dict with keys:
      available   : bool
      swing       : { expiry, dte, call, put }
      medium      : { expiry, dte, call, put }
    """
    try:
        yft = yf.Ticker(ticker)
        expirations = yft.options
        if not expirations:
            return {"available": False, "reason": "No options data available"}

        today     = datetime.today()
        result    = {"available": True, "current_price": current_price}

        for label, target, (dmin, dmax) in [
            ("swing",  swing_target, SWING_DTE),
            ("medium", long_target,  MEDIUM_DTE),
        ]:
            exp = _pick_expiry(expirations, today, dmin, dmax)
            if not exp:
                continue

            chain = yft.option_chain(exp)
            dte   = (datetime.strptime(exp, "%Y-%m-%d") - today).days
            put_target = current_price * (0.93 if label == "swing" else 0.88)

            call = _pick_option(chain.calls, current_price, target,     "call")
            put  = _pick_option(chain.puts,  current_price, put_target, "put")

            result[label] = {
                "expiry": exp,
                "dte":    dte,
                "call":   _annotate(call, current_price, target,     "call", dte, ticker) if call else None,
                "put":    _annotate(put,  current_price, put_target, "put",  dte, ticker) if put  else None,
            }

        return result

    except Exception as exc:
        logger.warning(f"{ticker} options fetch error: {exc}")
        return {"available": False, "reason": str(exc)}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _pick_expiry(expirations: tuple, today: datetime, dmin: int, dmax: int) -> str | None:
    """Return the expiration date whose DTE is closest to the midpoint of [dmin, dmax]."""
    mid  = (dmin + dmax) / 2
    best, best_diff = None, float("inf")
    for exp in expirations:
        try:
            dte  = (datetime.strptime(exp, "%Y-%m-%d") - today).days
        except ValueError:
            continue
        if dmin <= dte <= dmax:
            d = abs(dte - mid)
            if d < best_diff:
                best_diff, best = d, exp
    return best


def _pick_option(
    chain: pd.DataFrame,
    current_price: float,
    target: float,
    option_type: str,
) -> dict | None:
    """
    Select the single best option from the chain.

    Strategy:
      Call — prefer a strike between current price and ~halfway to target.
             Aim for a delta around 0.40–0.55 (slightly OTM, good leverage).
      Put  — prefer a strike ~5% below current price.
    """
    if chain is None or chain.empty:
        return None

    chain = chain.copy()

    # Keep only liquid options (some volume or open interest)
    chain = chain[(chain["volume"].fillna(0) > 0) | (chain["openInterest"].fillna(0) > 100)]
    if chain.empty:
        return None

    if option_type == "call":
        ideal  = current_price * 1.03           # ~3% OTM — good risk/reward
        chain  = chain[chain["strike"].between(current_price * 0.97, current_price * 1.20)]
    else:
        ideal  = current_price * 0.95           # ~5% OTM put
        chain  = chain[chain["strike"].between(current_price * 0.80, current_price * 1.02)]

    if chain.empty:
        return None

    chain["_dist"] = (chain["strike"] - ideal).abs()
    return chain.sort_values("_dist").iloc[0].to_dict()


def _annotate(
    opt: dict,
    current_price: float,
    target: float,
    option_type: str,
    dte: int,
    ticker: str,
) -> dict:
    """Convert a raw options-chain row into a readable recommendation dict."""
    strike   = float(opt.get("strike", 0))
    ask      = float(opt.get("ask")  or opt.get("lastPrice") or 0)
    bid      = float(opt.get("bid")  or 0)
    iv       = float(opt.get("impliedVolatility") or 0)
    volume   = int(opt.get("volume")        or 0)
    oi       = int(opt.get("openInterest")  or 0)
    itm      = bool(opt.get("inTheMoney", False))

    premium       = ask if ask > 0 else float(opt.get("lastPrice") or 0)
    contract_cost = round(premium * 100, 2)   # 1 contract controls 100 shares

    if option_type == "call":
        breakeven         = round(strike + premium, 2)
        intrinsic_target  = max(0.0, target - strike)
        pct_to_breakeven  = round((breakeven / current_price - 1) * 100, 2)
        action_note       = f"Buy to open CALL {ticker} ${strike:.0f} exp {dte}d — bullish play"
        payoff_note       = (
            f"If {ticker} reaches ${target:.2f}: intrinsic value ≈ ${intrinsic_target:.2f}/share "
            f"(contract ≈ ${intrinsic_target*100:.0f})"
        )
    else:
        breakeven         = round(strike - premium, 2)
        intrinsic_target  = max(0.0, strike - target)
        pct_to_breakeven  = round((1 - breakeven / current_price) * 100, 2)
        action_note       = f"Buy to open PUT {ticker} ${strike:.0f} exp {dte}d — bearish / hedge play"
        payoff_note       = (
            f"If {ticker} drops to ${target:.2f}: intrinsic value ≈ ${intrinsic_target:.2f}/share "
            f"(contract ≈ ${intrinsic_target*100:.0f})"
        )

    est_profit_pct = (
        round((intrinsic_target / premium - 1) * 100)
        if premium > 0 and intrinsic_target > 0
        else 0
    )

    # IV context: < 30% = cheap, 30–60% = normal, > 60% = expensive
    iv_label = "Low (cheap)" if iv < 0.30 else ("High (expensive)" if iv > 0.60 else "Normal")

    return {
        "type":                    option_type.upper(),
        "strike":                  strike,
        "in_the_money":            itm,
        "dte":                     dte,
        "premium_per_share":       round(premium, 2),
        "contract_cost_usd":       contract_cost,
        "bid":                     round(bid, 2),
        "ask":                     round(ask, 2),
        "break_even":              breakeven,
        "pct_to_breakeven":        pct_to_breakeven,
        "iv_pct":                  round(iv * 100, 1),
        "iv_label":                iv_label,
        "volume":                  volume,
        "open_interest":           oi,
        "target_price":            round(target, 2),
        "est_intrinsic_at_target": round(intrinsic_target, 2),
        "est_profit_pct_at_target": est_profit_pct,
        "action_note":             action_note,
        "payoff_note":             payoff_note,
    }


# ── Best single trade picker ──────────────────────────────────────────────────

def pick_best_option(rec: dict, direction: str = "bullish") -> dict:
    """
    Choose the ONE best option that MATCHES the stock's direction.

      direction="bullish" → considers only CALLs
      direction="bearish" → considers only PUTs

    Mismatched recommendations (PUT on a bullish-ranked stock, or CALL on a
    bearish-ranked stock) are contradictory and confused the user during the
    DDOG paper-loss incident (2026-05-07). Always pass the direction that
    matches how the underlying stock was ranked.

    Scoring weights:
      35% — expected profit % at price target
      20% — liquidity (volume + open interest)
      15% — IV level (lower = cheaper premium for the buyer)
      30% — affordability (prefer contracts under $600 for beginners)
    """
    if not rec.get("available"):
        return {"available": False, "reason": rec.get("reason", "No options data")}

    if direction == "bullish":
        side = "call"
    elif direction == "bearish":
        side = "put"
    else:
        return {"available": False, "reason": f"Unknown direction: {direction!r}"}

    candidates: list[dict] = []

    for tf_key, tf_label in [("swing", "Swing (3–5 wk)"), ("medium", "Medium (2–3 mo)")]:
        tf = rec.get(tf_key)
        if not tf:
            continue
        opt = tf.get(side)
        if not opt or opt.get("premium_per_share", 0) == 0:
            continue

        profit  = opt.get("est_profit_pct_at_target", 0) or 0
        vol     = opt.get("volume", 0) or 0
        oi      = opt.get("open_interest", 0) or 0
        iv      = opt.get("iv_pct", 50) or 50
        cost    = opt.get("contract_cost_usd", 999) or 999

        profit_sc  = min(100, max(0, profit * 0.5))
        liquid_sc  = min(100, vol * 0.4 + oi * 0.08)
        iv_sc      = max(0, 100 - iv * 1.2)

        # Affordability: strong preference for contracts under $600.
        # < $300 → 100 pts   $300–$600 → 60–100 pts   > $600 → heavy penalty
        if cost <= 300:
            cost_sc = 100
        elif cost <= 600:
            cost_sc = 100 - (cost - 300) * 0.13     # 100 → 60
        else:
            cost_sc = max(0, 60 - (cost - 600) * 0.06)  # rapid drop-off

        total = (
            profit_sc * 0.35 +
            liquid_sc * 0.20 +
            iv_sc     * 0.15 +
            cost_sc   * 0.30
        )

        candidates.append({
            "timeframe":       tf_label,
            "side":            side,
            "expiry":          tf["expiry"],
            "dte":             tf["dte"],
            "opt":             opt,
            "_score":          total,
        })

    if not candidates:
        return {"available": False, "reason": "No scoreable options found"}

    best = max(candidates, key=lambda x: x["_score"])
    opt  = best["opt"]

    # Plain-English reason for this pick
    reasons: list[str] = []
    if best["side"] == "call":
        reasons.append("bullish momentum favours calls")
    elif best["side"] == "put":
        reasons.append("bearish setup favours puts")
    if (opt.get("iv_pct") or 50) < 30:
        reasons.append(f"cheap IV ({opt['iv_pct']:.0f}%) keeps premium low")
    if (opt.get("est_profit_pct_at_target") or 0) > 80:
        reasons.append(f"strong upside ({opt['est_profit_pct_at_target']:+.0f}%) if target hit")
    if (opt.get("volume") or 0) > 50 or (opt.get("open_interest") or 0) > 100:
        reasons.append("liquid — easy to enter/exit")
    if (opt.get("contract_cost_usd") or 999) < 400:
        reasons.append(f"affordable at ${opt['contract_cost_usd']:.0f}/contract")
    reason_str = "; ".join(reasons) if reasons else "best overall risk/reward"

    return {
        "available":        True,
        "timeframe":        best["timeframe"],
        "side":             best["side"].upper(),
        "expiry":           best["expiry"],
        "dte":              best["dte"],
        "strike":           opt["strike"],
        "premium":          opt["premium_per_share"],
        "contract_cost":    opt["contract_cost_usd"],
        "bid":              opt["bid"],
        "ask":              opt["ask"],
        "break_even":       opt["break_even"],
        "pct_to_breakeven": opt["pct_to_breakeven"],
        "iv_pct":           opt["iv_pct"],
        "iv_label":         opt["iv_label"],
        "volume":           opt.get("volume", 0),
        "open_interest":    opt.get("open_interest", 0),
        "target_price":     opt.get("target_price", 0),
        "est_profit_pct":   opt.get("est_profit_pct_at_target", 0),
        "payoff_note":      opt.get("payoff_note", ""),
        "reason":           reason_str,
    }
