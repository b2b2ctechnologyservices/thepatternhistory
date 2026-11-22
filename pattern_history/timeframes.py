"""
Weekly, monthly and yearly bars built from the daily prices.

A weekly bar is Monday to Friday, a monthly bar is a calendar month, a yearly
bar is a calendar year: open = first day's open, high = highest high, low =
lowest low, close = last day's close, volume = total. Each bar is dated by its
last trading day, so dates on the site are real trading days.

The bar that is still in progress (this week, this month, this year) is
dropped for weekly and monthly patterns, so a pattern can't appear and vanish
halfway through the week. Yearly checks keep it and label it "so far".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.tseries.offsets import BDay


@dataclass(frozen=True)
class Timeframe:
    key: str          # "daily", "weekly", ...
    label: str        # "Daily"
    unit: str         # "day" / "week" / ...
    freq: str | None  # pandas period frequency, None for daily
    horizons: tuple[int, ...]
    suffix: str       # forward-return column suffix: ret_{h}{suffix}
    start: int        # bars of history needed before the first bar is checked
    bigger: str | None  # the timeframe used as the "bigger picture"


DAILY = Timeframe("daily", "Daily", "day", None, (1, 5, 20, 60), "d", 220, "weekly")
# Weekly/monthly start after 60 bars (not 220 like daily): the patterns that
# need a 200-bar average simply don't fire until it exists.
WEEKLY = Timeframe("weekly", "Weekly", "week", "W-FRI", (1, 4, 13, 26), "w", 60, "monthly")
MONTHLY = Timeframe("monthly", "Monthly", "month", "M", (1, 3, 6, 12), "m", 60, None)
YEARLY = Timeframe("yearly", "Yearly", "year", "Y", (1,), "y", 2, None)

TIMEFRAMES = {tf.key: tf for tf in (DAILY, WEEKLY, MONTHLY, YEARLY)}

HORIZON_TEXT = {
    "d": {1: "next day", 5: "1 week later", 20: "1 month later", 60: "3 months later"},
    "w": {1: "next week", 4: "1 month later", 13: "3 months later", 26: "6 months later"},
    "m": {1: "next month", 3: "3 months later", 6: "6 months later", 12: "1 year later"},
    "y": {1: "next year"},
}


def is_complete(last_day: pd.Timestamp, freq: str) -> bool:
    """True if `last_day` is the final trading day of its week/month/year."""
    return (last_day + BDay(1)).to_period(freq) != last_day.to_period(freq)


def resample(daily: pd.DataFrame, freq: str, keep_partial: bool = False) -> pd.DataFrame:
    """Daily OHLCV -> weekly/monthly/yearly OHLCV, dated by each bar's last trading day."""
    if daily.empty:
        return daily.copy()
    period = daily.index.to_period(freq)
    g = daily.groupby(period)
    bars = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    last_days = pd.Series(daily.index, index=period).groupby(level=0).last()
    bars.index = pd.DatetimeIndex(last_days.to_numpy(), name="ts")
    if not keep_partial and not is_complete(daily.index[-1], freq):
        bars = bars.iloc[:-1]
    return bars.astype(float)


def bars_for(daily: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    if tf.freq is None:
        return daily
    return resample(daily, tf.freq, keep_partial=(tf.key == "yearly"))
