"""
Yearly checks. With only ~25-45 yearly bars there aren't enough bars for
moving averages, so instead of patterns we use four simple year-on-year
checks, each compared with the previous calendar year:

  YEAR_INSIDE      the year's whole range stayed inside last year's range
  YEAR_ABOVE_HIGH  the year closed above last year's high
  YEAR_BELOW_LOW   the year closed below last year's low
  YEAR_UP / YEAR_DOWN  the year closed above / below where it opened

For the current (unfinished) year the check uses the year so far. History
uses finished years only, and the outcome is the next calendar year's change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHECK_TEXT = {
    "YEAR_INSIDE": "Inside last year's range",
    "YEAR_ABOVE_HIGH": "Above last year's high",
    "YEAR_BELOW_LOW": "Below last year's low",
    "YEAR_UP": "Up year (above the year's open)",
    "YEAR_DOWN": "Down year (below the year's open)",
}
# Yearly checks are facts about the year, not patterns that point up or down,
# so history simply reports how often the NEXT year went up.
CHECK_DIRECTION = {k: "NONE" for k in CHECK_TEXT}


def checks_for(bar: pd.Series, prev: pd.Series) -> list[str]:
    out = []
    if bar["high"] <= prev["high"] and bar["low"] >= prev["low"]:
        out.append("YEAR_INSIDE")
    if bar["close"] > prev["high"]:
        out.append("YEAR_ABOVE_HIGH")
    if bar["close"] < prev["low"]:
        out.append("YEAR_BELOW_LOW")
    if bar["close"] > bar["open"]:
        out.append("YEAR_UP")
    elif bar["close"] < bar["open"]:
        out.append("YEAR_DOWN")
    return out


def replay_yearly(years: pd.DataFrame, last_is_partial: bool = True) -> pd.DataFrame:
    """
    One row per (year, check). `years` includes the current partial year as the
    last row (timeframes.resample(..., keep_partial=True)); its ret_1y is NaN.
    """
    cols = ["date", "rule", "pattern", "direction", "confidence", "close", "factors", "ret_1y"]
    rows = []
    closes = years["close"].to_numpy(float)
    full = len(years) - (1 if last_is_partial else 0)  # rows [0, full) are finished years
    for i in range(1, len(years)):
        for chk in checks_for(years.iloc[i], years.iloc[i - 1]):
            rows.append({
                "date": years.index[i], "rule": "yearly", "pattern": chk,
                "direction": CHECK_DIRECTION[chk], "confidence": 1, "close": closes[i], "factors": "",
                "ret_1y": closes[i + 1] / closes[i] - 1.0 if i + 1 < full else np.nan,
            })
    return pd.DataFrame(rows, columns=cols)
