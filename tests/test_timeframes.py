import numpy as np
import pandas as pd

from pattern_history import doc_rules
from pattern_history.history import replay, rule_stats
from pattern_history.multi import scan_all
from pattern_history.timeframes import WEEKLY, is_complete, resample
from pattern_history.yearly import replay_yearly


def _daily(start="2000-01-03", periods=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, periods)))
    open_ = close * (1 + rng.normal(0, 0.005, periods))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, periods))
    vol = rng.integers(1_000_000, 5_000_000, periods).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_weekly_bar_is_monday_to_friday():
    d = _daily(periods=30)
    w = resample(d, "W-FRI", keep_partial=True)
    first_week = d.loc["2000-01-03":"2000-01-07"]
    row = w.iloc[0]
    assert w.index[0] == pd.Timestamp("2000-01-07")
    assert row["open"] == first_week["open"].iloc[0]
    assert row["close"] == first_week["close"].iloc[-1]
    assert row["high"] == first_week["high"].max() and row["low"] == first_week["low"].min()
    assert row["volume"] == first_week["volume"].sum()


def test_unfinished_week_and_month_are_dropped():
    d = _daily(periods=33)  # ends Wed 2000-02-16
    assert not is_complete(d.index[-1], "W-FRI")
    assert resample(d, "W-FRI").index[-1] == pd.Timestamp("2000-02-11")
    assert resample(d, "M").index[-1] == pd.Timestamp("2000-01-31")
    assert len(resample(d, "W-FRI", keep_partial=True)) == len(resample(d, "W-FRI")) + 1


def test_replay_uses_bar_horizons():
    d = resample(_daily(), "W-FRI")
    h = replay(d, horizons=WEEKLY.horizons, suffix="w", start=WEEKLY.start)
    assert [c for c in h.columns if c.startswith("ret_")] == ["ret_1w", "ret_4w", "ret_13w", "ret_26w"]
    if len(h):
        i = d.index.get_loc(h["date"].iloc[0])
        assert abs(h["ret_4w"].iloc[0] - (d["close"].iloc[i + 4] / d["close"].iloc[i] - 1)) < 1e-12
        assert len(rule_stats(h, "LONG", WEEKLY.horizons, "w")) == 4


def test_doc_rules_do_not_look_ahead():
    """A row's checklist must be the same whether or not later bars exist."""
    d = _daily(periods=1500, seed=3)
    full = doc_rules.replay_doc(d, (1,), "d", 220)
    assert len(full) > 0
    for date in full["date"].iloc[:5]:
        cut = d[d.index <= date]
        part = doc_rules.replay_doc(cut, (1,), "d", 220)
        a = full[full["date"] == date].drop(columns="ret_1d").reset_index(drop=True)
        b = part[part["date"] == date].drop(columns="ret_1d").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        ann_full = doc_rules.annotate(d, full[full["date"] == date])
        ann_cut = doc_rules.annotate(cut, part[part["date"] == date])
        assert list(ann_full["signs"]) == list(ann_cut["signs"])
        assert list(ann_full["location"]) == list(ann_cut["location"])


def test_doc_patterns_have_two_factors():
    d = _daily(periods=2500, seed=5)
    h = doc_rules.annotate(d, doc_rules.replay_doc(d, (1,), "d", 220))
    assert len(h) > 0
    assert (h["factor_count"] >= 2).all()
    assert (h["location"] != "").all()


def test_bigger_picture_uses_finished_bars_only():
    d = _daily(periods=1200)
    w = resample(d, "W-FRI")
    bp = doc_rules.bigger_picture(d, w)
    wk_trend = doc_rules.trend_series(__import__("rules")._add_indicators(w))
    friday = w.index[-10]
    # the Friday that finishes a week still sees the PREVIOUS week's trend
    assert bp[friday] == wk_trend.iloc[-11]
    assert bp[d.index[d.index > friday][0]] == wk_trend.iloc[-10]


def test_yearly_checks_and_partial_year():
    years = pd.DataFrame({
        "open": [10, 12, 11, 13], "high": [15, 14, 20, 18], "low": [8, 9, 10, 12],
        "close": [12, 11, 19, 14], "volume": [1, 1, 1, 1],
    }, index=pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-30", "2023-06-30"]), dtype=float)
    h = replay_yearly(years, last_is_partial=True)
    got = {(r.date.year, r.pattern) for r in h.itertuples()}
    assert (2021, "YEAR_INSIDE") in got and (2021, "YEAR_DOWN") in got
    assert (2022, "YEAR_ABOVE_HIGH") in got and (2022, "YEAR_UP") in got
    assert (2023, "YEAR_INSIDE") in got
    r2021 = h[(h.date.dt.year == 2021)]["ret_1y"].iloc[0]
    assert abs(r2021 - (19 / 11 - 1)) < 1e-12
    # 2022's "next year" is the unfinished 2023, so no outcome yet
    assert h[h.date.dt.year == 2022]["ret_1y"].isna().all()


def test_scan_all_builds_every_timeframe():
    r = scan_all(_daily(periods=6000, seed=7))
    assert set(r) == {"daily", "weekly", "monthly", "yearly"}
    assert r["weekly"].bars.index.is_monotonic_increasing
    assert "meets_rules" in r["daily"].hist.columns
    assert "ret_12m" in r["monthly"].hist.columns
