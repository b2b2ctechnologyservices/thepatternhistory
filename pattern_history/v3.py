"""
Version 3 rule set (from Ritesh's "Version 3 rules" spec), layered on top of
the ORIGINAL detectors in rules.py, which are not modified.

A signal from the original detectors also counts under V3 only if:

1. Location (spec 1.2 / 6.1 / 13.5): it happens at a meaningful place:
     - a MAJOR moving-average touch (SMA20/50/100/200, first touch from an
       extended move, using the original major/minor logic), or
     - a bow tie in the trade's direction, or
     - the setup itself is a flag or wedge ("valid continuation zone").
2. Supporting factors (spec 6.2): at least one of
     round number, narrow-body bar, inside bar, wide-range bar.
   Pattern self-labels (e.g. "inside_bar" on an inside-bar signal) and the
   trend itself no longer count as factors.
3. Weekly trend (spec 2.1 / 13.4): skip longs when the weekly trend is
   bearish and shorts when it is bullish. Weekly trend uses completed weeks
   only: bullish = weekly close > SMA20w > SMA50w, bearish = the reverse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rules

CONTINUATION_PATTERNS = {"BULL_FLAG", "BEAR_FLAG", "BULL_WEDGE", "BEAR_WEDGE"}
SUPPORTING = ("round_number", "narrow_body_bar", "inside_bar", "wide_range_bar")
LOCATION_LABEL = {"major_ma": "major moving average", "bow_tie": "bow tie", "continuation": "flag/wedge zone"}
SUPPORT_LABEL = {"round_number": "round number", "narrow_body_bar": "narrow body",
                 "inside_bar": "inside bar", "wide_range_bar": "wide-range bar"}


def weekly_trend_by_day(df: pd.DataFrame) -> pd.Series:
    """For each daily bar: weekly trend from weeks completed BEFORE that bar's week."""
    wk = df["close"].resample("W-FRI").last().dropna()
    s20, s50 = wk.rolling(20).mean(), wk.rolling(50).mean()
    trend = pd.Series("neutral", index=wk.index)
    trend[(wk > s20) & (s20 > s50)] = "bullish"
    trend[(wk < s20) & (s20 < s50)] = "bearish"
    trend[s50.isna()] = "neutral"
    prev = trend.shift(1).fillna("neutral")  # only completed weeks
    week_end = df.index.to_period("W-FRI").to_timestamp(how="end").normalize()
    return pd.Series(prev.reindex(week_end).fillna("neutral").to_numpy(), index=df.index)


def _major_ma_touch(w: pd.DataFrame) -> bool:
    r = w.iloc[-1]
    atr = r["atr14"]
    if pd.isna(atr) or atr <= 0:
        return False
    for period in (20, 50, 100, 200):
        ma = r[f"ma{period}"]
        if pd.isna(ma):
            continue
        touched = r["low"] <= ma <= r["high"] or abs(r["close"] - ma) <= 0.10 * atr
        if touched and rules._ma_significance(w, period) == "MAJOR":
            return True
    return False


def factors_on_bar(w: pd.DataFrame, direction: str, pattern: str) -> tuple[list[str], list[str]]:
    """Location and supporting factors on the last bar of window `w` (with indicators)."""
    r, p = w.iloc[-1], w.iloc[-2]
    loc: list[str] = []
    if _major_ma_touch(w):
        loc.append("major_ma")
    bt = rules._bow_tie(w)
    if (direction == "LONG" and bt == "BULL") or (direction == "SHORT" and bt == "BEAR"):
        loc.append("bow_tie")
    if pattern in CONTINUATION_PATTERNS:
        loc.append("continuation")

    sup: list[str] = []
    atr = float(r["atr14"]) if pd.notna(r["atr14"]) else 0.0
    if rules._is_near_round_level(float(r["close"]), atr) is not None:
        sup.append("round_number")
    if bool(r["is_nbb"]):
        sup.append("narrow_body_bar")
    if r["high"] <= p["high"] and r["low"] >= p["low"]:
        sup.append("inside_bar")
    avg_range = w["range"].iloc[-20:].mean()
    if pd.notna(avg_range) and r["range"] >= 2.0 * avg_range:
        sup.append("wide_range_bar")
    return loc, sup


def annotate(df: pd.DataFrame, hist: pd.DataFrame, window: int = 260) -> pd.DataFrame:
    """Add V3 columns to the replay output: weekly_trend, v3_location, v3_support, v3_pass."""
    out = hist.copy()
    if out.empty:
        for c in ("weekly_trend", "v3_location", "v3_support"):
            out[c] = pd.Series(dtype=str)
        out["v3_pass"] = pd.Series(dtype=bool)
        return out
    d = rules._add_indicators(df)
    pos = pd.Series(np.arange(len(d)), index=d.index)
    weekly = weekly_trend_by_day(df)
    wt, locs, sups, ok = [], [], [], []
    for date, direction, pattern in zip(out["date"], out["direction"], out["pattern"]):
        i = int(pos[date])
        w = d.iloc[max(0, i - window + 1): i + 1]
        loc, sup = factors_on_bar(w, direction, pattern)
        trend = weekly.iloc[i]
        opposed = (direction == "LONG" and trend == "bearish") or (direction == "SHORT" and trend == "bullish")
        wt.append(trend)
        locs.append(", ".join(LOCATION_LABEL[x] for x in loc))
        sups.append(", ".join(SUPPORT_LABEL[x] for x in sup))
        ok.append(bool(loc) and bool(sup) and not opposed)
    out["weekly_trend"] = wt
    out["v3_location"] = locs
    out["v3_support"] = sups
    out["v3_pass"] = ok
    return out


def why_not(row) -> str:
    """Short reason a signal fails V3 (for the UI)."""
    reasons = []
    if not row["v3_location"]:
        reasons.append("no location factor")
    if not row["v3_support"]:
        reasons.append("no supporting factor")
    d, t = row["direction"], row["weekly_trend"]
    if (d == "LONG" and t == "bearish") or (d == "SHORT" and t == "bullish"):
        reasons.append(f"weekly trend is {t}")
    return "; ".join(reasons)
