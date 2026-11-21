"""
Replay the R3 rules over a ticker's full daily history.

The rule logic in rules.py is used exactly as written: the same detector
functions, the same indicators and the same "at least MIN_FACTORS factors"
filter that rules.scan() applies. The only difference from scan() is that we
keep EVERY detector that fires on a given day, not just the single best one,
and we evaluate every past day, not only the latest bar.

tests/test_matches_original.py checks that this replay agrees with
rules.scan() day by day.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

import rules

# Same list, same order as rules.scan() uses for strategy_type="SWING" (daily).
DAILY_DETECTORS = [
    rules._detect_flag,
    rules._detect_inside_bar,
    rules._detect_wedge,
    rules._detect_reversal_bar_50,
    rules._detect_time_count,
    rules._detect_major_ma_bounce,
    rules._detect_extended_move_reversal,
    rules._detect_max_extension,
    rules._detect_double_top_bottom,
    rules._detect_wide_range_reversal,
]

HORIZONS = (1, 5, 20, 60)

# Detectors only look back a bounded number of bars (longest length check is
# 205 bars, longest lookback 51 bars). Passing a trailing window of this size
# with the indicators already computed on the full history is equivalent to
# passing the whole prefix, and much faster. The equivalence is tested.
_WINDOW = 260


@contextmanager
def _memoized_trend():
    """
    Speed-up only: rules._trend(d) is a pure function of the frame it gets and
    is called many times per bar (e.g. once per loop step inside _detect_flag).
    While replaying, cache its result per frame object. The rule code itself is
    not modified; the original function is restored afterwards.
    """
    original = rules._trend
    cache: dict = {}

    def cached(d):
        key = (id(d), len(d), d.index[-1] if len(d) else None)
        if key not in cache:
            cache[key] = original(d)
        return cache[key]

    rules._trend = cached
    try:
        yield cache
    finally:
        rules._trend = original


def candidates_on_bar(d: pd.DataFrame, min_confidence: int = rules.MIN_FACTORS) -> list[dict]:
    """
    Run every daily detector on the last bar of `d` (a frame that already has
    rules._add_indicators applied). Returns the candidates that pass the same
    factor filter as rules.scan(), each with a "confidence" key, in detector order.
    """
    out: list[dict] = []
    for fn in DAILY_DETECTORS:
        try:
            result = fn(d)
        except Exception:
            result = None
        if result is None:
            continue
        factor_count = len(set(result["factors"]))
        if factor_count < min_confidence:
            continue
        result["confidence"] = min(factor_count, 5)
        result["detector"] = fn.__name__
        out.append(result)
    return out


def replay(df: pd.DataFrame, min_confidence: int = rules.MIN_FACTORS) -> pd.DataFrame:
    """
    Evaluate every day in `df` that has enough history (rules.MIN_BARS).
    Returns one row per (date, firing pattern) with entry price, direction,
    confidence, factors and forward returns over HORIZONS trading days.
    """
    cols = ["date", "rule", "pattern", "direction", "confidence", "close", "factors"] + [f"ret_{h}d" for h in HORIZONS]
    if len(df) < rules.MIN_BARS:
        return pd.DataFrame(columns=cols)

    d = rules._add_indicators(df)
    closes = df["close"].to_numpy(dtype=float)
    n = len(d)
    rows: list[dict] = []

    with _memoized_trend() as trend_cache:
        for i in range(rules.MIN_BARS - 1, n):
            trend_cache.clear()
            window = d.iloc[max(0, i - _WINDOW + 1): i + 1]
            rows.extend(_rows_for_bar(d, i, window, closes, n, min_confidence))

    return pd.DataFrame(rows, columns=cols)


def _rows_for_bar(d, i, window, closes, n, min_confidence) -> list[dict]:
    rows = []
    for c in candidates_on_bar(window, min_confidence):
        row = {
            "date": d.index[i],
            "rule": c["detector"],
            "pattern": c["pattern"],
            "direction": c["direction"],
            "confidence": c["confidence"],
            "close": closes[i],
            "factors": ", ".join(sorted(set(c["factors"]))),
        }
        for h in HORIZONS:
            j = i + h
            row[f"ret_{h}d"] = closes[j] / closes[i] - 1.0 if j < n else np.nan
        rows.append(row)
    return rows


@dataclass
class HorizonStats:
    horizon: int
    sample: int
    win_rate: float
    average: float
    median: float
    worst: float
    best: float


def directional(returns: pd.Series, direction: str) -> pd.Series:
    """Return as experienced by someone following the rule (SHORT flips sign)."""
    return -returns if direction == "SHORT" else returns


def rule_stats(occurrences: pd.DataFrame, direction: str) -> list[HorizonStats]:
    """
    Stats per horizon for one rule's past occurrences. A "win" means the price
    moved in the rule's direction (up for LONG, down for SHORT). Averages,
    median and worst are in the rule's direction too.
    """
    stats = []
    for h in HORIZONS:
        r = directional(occurrences[f"ret_{h}d"].dropna(), direction)
        if r.empty:
            stats.append(HorizonStats(h, 0, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        stats.append(
            HorizonStats(
                horizon=h,
                sample=int(len(r)),
                win_rate=float((r > 0).mean()),
                average=float(r.mean()),
                median=float(r.median()),
                worst=float(r.min()),
                best=float(r.max()),
            )
        )
    return stats


def outlook(hist: pd.DataFrame, last_date, horizon: int) -> dict:
    """
    Bullish / bearish / unknown outlook for the latest bar.

    If no rule is confirmed on the latest bar the outlook is "unknown".
    Otherwise: take every PAST day on which any of today's firing patterns also
    fired (each day counted once even if several fired), and compare the close
    with the close `horizon` trading days later. Bullish % = share of those days
    where price was higher, bearish % = share where it was lower (unchanged
    days count toward neither).
    """
    today = hist[hist["date"] == last_date]
    patterns = sorted(set(today["pattern"]))
    if not patterns:
        return {"status": "unknown", "patterns": [], "sample": 0,
                "bullish": float("nan"), "bearish": float("nan"), "horizon": horizon}
    col = f"ret_{horizon}d"
    past = hist[(hist["pattern"].isin(patterns)) & (hist["date"] < last_date)].dropna(subset=[col])
    by_day = past.groupby("date")[col].first()  # same day -> same price outcome
    n = int(len(by_day))
    return {
        "status": "pattern" if n else "no_history",
        "patterns": patterns,
        "sample": n,
        "bullish": float((by_day > 0).mean()) if n else float("nan"),
        "bearish": float((by_day < 0).mean()) if n else float("nan"),
        "horizon": horizon,
    }
