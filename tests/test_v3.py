import numpy as np
import pandas as pd

import rules
from pattern_history import v3
from pattern_history.history import replay


def _synthetic(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.01, n)))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": rng.integers(1e6, 5e6, n).astype(float)},
                        index=pd.bdate_range("2015-01-05", periods=n))


def test_weekly_trend_has_no_lookahead():
    df = _synthetic()
    full = v3.weekly_trend_by_day(df)
    cut = 1000
    part = v3.weekly_trend_by_day(df.iloc[: cut + 1])
    assert full.iloc[: cut + 1].tolist() == part.tolist()


def test_v3_is_subset_and_does_not_touch_original():
    df = _synthetic()
    h = replay(df)
    a = v3.annotate(df, h)
    assert len(a) == len(h) and a["pattern"].tolist() == h["pattern"].tolist()
    assert a["v3_pass"].sum() <= len(a)
    passed = a[a["v3_pass"]]
    assert (passed["v3_location"] != "").all() and (passed["v3_support"] != "").all()
    opposed = ((passed["direction"] == "LONG") & (passed["weekly_trend"] == "bearish")) | (
        (passed["direction"] == "SHORT") & (passed["weekly_trend"] == "bullish"))
    assert not opposed.any()
    assert rules.MIN_FACTORS == 2  # original rules untouched
