"""
Today's trade plan, exactly as the original R3 bot would have built it.

Uses the original code only: the detectors' entry/stop/target, rules._build_signal
(rounding, confidence, R-multiple), the scan() ranking for the bot's top pick,
and the bot's signal-level acceptance rules (scanner/trader/preflight):
min confidence 2, min R 1.5, stop and target on the correct side of entry.
Account-level rules (buying power, max open trades, kill switch) don't apply to
a history site and are not checked.
"""

from __future__ import annotations

import pandas as pd

import rules
from pattern_history.history import _WINDOW, candidates_on_bar

MIN_CONFIDENCE = 2   # trader.MIN_CONFIDENCE / scanner.DEFAULT_MIN_CONF
MIN_R = 1.5          # trader.MIN_R / preflight.MIN_R


def _decision(sig: rules.Signal) -> tuple[str, str]:
    if sig.confidence < MIN_CONFIDENCE:
        return "SKIP", f"confidence {sig.confidence} is below {MIN_CONFIDENCE}"
    if sig.entry == sig.stop or sig.stop <= 0:
        return "SKIP", "missing stop"
    if sig.entry == sig.target or sig.target <= 0:
        return "SKIP", "missing target"
    long = sig.direction == "LONG"
    if (long and sig.stop >= sig.entry) or (not long and sig.stop <= sig.entry):
        return "SKIP", "stop is on the wrong side of entry"
    if (long and sig.target <= sig.entry) or (not long and sig.target >= sig.entry):
        return "SKIP", "target is on the wrong side of entry"
    if sig.r_multiple < MIN_R:
        return "SKIP", f"reward:risk {sig.r_multiple:.2f}× is below the bot's {MIN_R}× minimum"
    return "TAKE", f"confidence {sig.confidence}, reward:risk {sig.r_multiple:.2f}× (≥ {MIN_R}×)"


def todays_plans(symbol: str, df: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Return (context, plans) for the latest bar of df."""
    if len(df) < rules.MIN_BARS:
        return {"trend": "–", "technical": "–", "bow_tie": "–"}, []
    d = rules._add_indicators(df)
    w = d.iloc[-_WINDOW:]
    context = {
        "trend": rules._trend(w),
        "technical": rules._technical_position(w),
        "bow_tie": rules._bow_tie(w) or "none",
    }
    cands = candidates_on_bar(w)
    if not cands:
        return context, []
    # Same ranking as rules.scan(): highest confidence first, then (as coded) lower reward:risk.
    ranked = sorted(cands, key=lambda x: (
        -x["confidence"],
        abs(x["target"] - x["entry"]) / max(abs(x["entry"] - x["stop"]), 1e-9),
    ))
    plans = []
    for rank, c in enumerate(ranked):
        sig = rules._build_signal(symbol, c["direction"], c["pattern"], c["entry"], c["stop"],
                                  c["target"], c["factors"])
        decision, reason = _decision(sig)
        if rank > 0:
            # rules.scan() returns only the top-ranked setup; the bot never saw the others.
            on_own = "would pass the bot's filters on its own" if decision == "TAKE" else f"on its own: {reason}"
            decision, reason = "NOT CONSIDERED", f"the bot only acts on its top pick ({on_own})"
        plans.append({
            "pattern": sig.pattern, "direction": sig.direction, "entry": float(sig.entry),
            "stop": float(sig.stop), "target": float(sig.target), "r": float(sig.r_multiple),
            "confidence": int(sig.confidence),
            "factors": sig.factors, "decision": decision, "reason": reason, "top_pick": rank == 0,
        })
    return context, plans
