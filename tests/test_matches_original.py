"""
Confirms the website's rule replay gives the same results as the original
rules.scan(), using the original code untouched.

For sampled days in each ticker's history we:
  1. run the ORIGINAL rules.scan() on all bars up to that day, and
  2. check the replay found the same pattern on that day, and that picking the
     best candidate the way scan() does reproduces scan()'s exact signal
     (pattern, direction, entry, stop, target, confidence, factors).

Live tickers (AKAM, AAPL, SABR) download from yfinance and are skipped if the
network is unavailable. The offline test uses the CSVs already in data/bars/.
"""

from pathlib import Path

import pandas as pd
import pytest

import rules
from pattern_history.history import candidates_on_bar, replay

ROOT = Path(__file__).resolve().parent.parent
LIVE_TICKERS = ["AKAM", "AAPL", "SABR"]
SAMPLE_EVERY = 20  # check every 20th trading day (plus every day a rule fired)


def _best_like_scan(symbol: str, d: pd.DataFrame, cands: list[dict]) -> rules.Signal:
    """Same selection and Signal construction as the tail of rules.scan()."""
    if not cands:
        return rules._flat(symbol)
    cands = sorted(
        cands,
        key=lambda x: (
            -x["confidence"],
            abs(x["target"] - x["entry"]) / max(abs(x["entry"] - x["stop"]), 1e-9),
        ),
    )
    best = cands[0]
    return rules._build_signal(
        symbol=symbol,
        direction=best["direction"],
        pattern=best["pattern"],
        entry=best["entry"],
        raw_stop=best["stop"],
        target=best["target"],
        factors=best["factors"],
        notes=f"trend={rules._trend(d)} tech={rules._technical_position(d)} bow_tie={rules._bow_tie(d)}",
    )


def _check_against_original(symbol: str, df: pd.DataFrame) -> int:
    hist = replay(df)
    fired_by_day = hist.groupby("date")["pattern"].apply(set).to_dict()

    days = set(range(rules.MIN_BARS - 1, len(df), SAMPLE_EVERY))
    days |= {df.index.get_loc(t) for t in fired_by_day}
    days.add(len(df) - 1)
    checked = 0

    for i in sorted(days):
        prefix = df.iloc[: i + 1]
        original = rules.scan(symbol, prefix)           # untouched original code
        d = rules._add_indicators(prefix)
        cands = candidates_on_bar(d)
        mine = _best_like_scan(symbol, d, cands)

        assert str(mine) == str(original), f"{symbol} {df.index[i].date()}: {mine} != {original}"
        assert mine.factors == original.factors
        assert mine.notes == original.notes

        replay_patterns = fired_by_day.get(df.index[i], set())
        assert replay_patterns == {c["pattern"] for c in cands}, f"{symbol} {df.index[i].date()}"
        if original.direction != "FLAT":
            assert original.pattern in replay_patterns
        checked += 1
    return checked


def _load_repo_csv(symbol: str) -> pd.DataFrame:
    path = ROOT / "data" / "bars" / f"{symbol}.csv"
    if not path.exists():
        pytest.skip("offline sample data not in this repo")
    df = pd.read_csv(ROOT / "data" / "bars" / f"{symbol}.csv", parse_dates=["ts"])
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


@pytest.mark.parametrize("symbol", ["AAPL", "SABR"])
def test_offline_repo_data_matches_original(symbol):
    df = _load_repo_csv(symbol)
    assert _check_against_original(symbol, df) > 0


@pytest.mark.parametrize("symbol", LIVE_TICKERS)
def test_live_25y_matches_original(symbol):
    from pattern_history.data import DataError, load_daily

    try:
        df = load_daily(symbol, years=25)
    except DataError as e:
        pytest.skip(f"no data for {symbol}: {e}")
    except Exception as e:  # network blocked etc.
        pytest.skip(f"yfinance unavailable: {e}")
    assert len(df) > 2000
    assert _check_against_original(symbol, df) > 50
