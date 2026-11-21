import numpy as np
import pandas as pd

from pattern_history.data import DataError, normalize_ticker
from pattern_history.history import rule_stats

import pytest


def test_normalize_ticker():
    assert normalize_ticker(" aapl ") == "AAPL"
    assert normalize_ticker("reliance.ns") == "RELIANCE.NS"
    assert normalize_ticker("BRK.B") == "BRK-B"
    with pytest.raises(DataError):
        normalize_ticker("")
    with pytest.raises(DataError):
        normalize_ticker("not a ticker!!")


def test_short_rule_wins_when_price_falls():
    occ = pd.DataFrame({
        "ret_1d": [-0.02, 0.01, -0.03],
        "ret_5d": [-0.05, 0.04, np.nan],
        "ret_20d": [np.nan] * 3,
        "ret_60d": [0.1, 0.1, 0.1],
    })
    s = {x.horizon: x for x in rule_stats(occ, "SHORT")}
    assert s[1].sample == 3 and abs(s[1].win_rate - 2 / 3) < 1e-9
    assert abs(s[1].worst - (-0.01)) < 1e-9          # worst for a short = price up 1%
    assert s[5].sample == 2
    assert s[20].sample == 0
    assert s[60].win_rate == 0.0

    long = {x.horizon: x for x in rule_stats(occ, "LONG")}
    assert abs(long[1].worst - (-0.03)) < 1e-9


def test_yahoo_busy_is_retried_then_reported(monkeypatch):
    import yfinance as yf
    from yfinance.exceptions import YFRateLimitError

    from pattern_history import data

    calls = []

    def fake_download(*a, **k):
        calls.append(1)
        raise YFRateLimitError()

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(data.ProviderBusy):
        data._download("MU")
    assert len(calls) == 3


def test_outlook_unknown_and_dedup():
    from pattern_history.history import outlook

    d = pd.to_datetime
    hist = pd.DataFrame({
        "date": [d("2020-01-01"), d("2020-01-01"), d("2020-02-01"), d("2020-03-01"), d("2021-01-01")],
        "pattern": ["A", "B", "A", "C", "A"],
        "ret_20d": [0.05, 0.05, -0.02, 0.10, np.nan],
    })
    o = outlook(hist, d("2021-01-01"), 20)
    assert o["status"] == "pattern" and o["sample"] == 2          # 2020-01-01 counted once, C ignored
    assert o["bullish"] == 0.5 and o["bearish"] == 0.5
    assert outlook(hist, d("2022-01-01"), 20)["status"] == "unknown"
