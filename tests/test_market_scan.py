import numpy as np
import pandas as pd

from pattern_history.market_scan import latest_patterns, scan_one, volume_ratio
from pattern_history.multi import scan_all
from tests.test_timeframes import _daily

COLS = ["pattern", "direction", "meets_rules", "signs", "location"]


def test_last_bar_scan_matches_full_replay():
    """The all-stocks list (last ~5 years, last bar only) must agree with the single-stock page."""
    full = _daily(periods=2600, seed=3)
    hist = scan_all(full, only=("daily",))["daily"].hist
    checked = 0
    for back in range(0, 120, 3):
        d = full.iloc[: len(full) - back] if back else full
        five = d[d.index >= d.index[-1] - pd.DateOffset(years=5)]
        got = latest_patterns(five)
        want = hist[hist["date"] == d.index[-1]]
        got = got[COLS].reset_index(drop=True) if len(got) else want[COLS].iloc[0:0].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want[COLS].reset_index(drop=True), check_dtype=False)
        checked += len(want)
    assert checked > 0


def test_volume_shocker():
    d = _daily(periods=300, seed=1)
    d["volume"] = 1_000_000.0
    d.iloc[-1, d.columns.get_loc("volume")] = 3_000_000.0
    assert volume_ratio(d) == 3.0
    rows = scan_one("X.NS", d)
    assert rows and all(r["volume_x"] == 3.0 for r in rows)
