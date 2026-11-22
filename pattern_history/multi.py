"""
One call that builds every timeframe for a ticker:

    results = scan_all(daily_full_history)
    results["weekly"].bars, results["weekly"].hist

Daily uses the last 25 years (as before). Weekly, monthly and yearly use the
full price history Yahoo has, because bigger bars need more years.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import doc_rules, yearly
from .history import replay
from .timeframes import DAILY, MONTHLY, TIMEFRAMES, WEEKLY, YEARLY, Timeframe, bars_for, is_complete

DAILY_YEARS = 25


@dataclass
class TimeframeResult:
    tf: Timeframe
    bars: pd.DataFrame       # OHLCV bars on this timeframe
    hist: pd.DataFrame       # every past firing (original + doc patterns), annotated
    partial: bool = False    # yearly only: last bar is the year so far


def _patterns(bars: pd.DataFrame, tf: Timeframe, bigger: pd.Series | None) -> pd.DataFrame:
    orig = replay(bars, horizons=tf.horizons, suffix=tf.suffix, start=tf.start)
    new = doc_rules.replay_doc(bars, tf.horizons, tf.suffix, tf.start)
    both = pd.concat([x for x in (orig, new) if not x.empty], ignore_index=True) if not (orig.empty and new.empty) else orig
    both = both.sort_values("date", kind="stable").reset_index(drop=True)
    return doc_rules.annotate(bars, both, bigger)


def first_year_is_partial(first_day: pd.Timestamp) -> bool:
    """True if the price history starts after the first week of its calendar year (holidays allowed)."""
    return first_day > pd.Timestamp(first_day.year, 1, 1) + pd.Timedelta(days=7)


def scan_all(daily_full: pd.DataFrame, only: tuple[str, ...] = tuple(TIMEFRAMES)) -> dict[str, TimeframeResult]:
    cutoff = daily_full.index[-1] - pd.DateOffset(years=DAILY_YEARS)
    daily = daily_full[daily_full.index >= cutoff]
    weekly = bars_for(daily_full, WEEKLY)
    monthly = bars_for(daily_full, MONTHLY)
    out: dict[str, TimeframeResult] = {}
    if "daily" in only:
        out["daily"] = TimeframeResult(DAILY, daily, _patterns(daily, DAILY, doc_rules.bigger_picture(daily, weekly)))
    if "weekly" in only:
        out["weekly"] = TimeframeResult(WEEKLY, weekly, _patterns(weekly, WEEKLY, doc_rules.bigger_picture(weekly, monthly)))
    if "monthly" in only:
        out["monthly"] = TimeframeResult(MONTHLY, monthly, _patterns(monthly, MONTHLY, None))
    if "yearly" in only:
        years = bars_for(daily_full, YEARLY)
        if first_year_is_partial(daily_full.index[0]):
            years = years.iloc[1:]  # a stub first year isn't a fair "last year" to compare with
        partial = not is_complete(daily_full.index[-1], "Y")
        out["yearly"] = TimeframeResult(YEARLY, years, yearly.replay_yearly(years, partial), partial)
    return out
