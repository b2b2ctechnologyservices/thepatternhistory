"""
Market-wide scan: which stocks show a daily pattern on the latest trading day.

Run once a day after the market closes (see scan/run_india.py and the GitHub
workflow). It checks only the LATEST daily bar of every stock, with the same
detectors, doc rules and checklist the single-stock page uses, and adds a
"volume shocker" flag (today's volume vs. the average of the 20 days before).

Checking only the last bar needs ~5 years of prices per stock (enough for the
200-day and 200-week averages), so a full market takes minutes, not hours.
tests/test_market_scan.py checks it gives the same patterns as the full
single-stock replay.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import doc_rules
from .history import replay
from .timeframes import DAILY, WEEKLY, bars_for

VOLUME_SHOCK = 2.0   # today's volume at least 2x the average of the previous 20 days
SCAN_PERIOD = "5y"


def latest_patterns(daily: pd.DataFrame) -> pd.DataFrame:
    """Patterns on the last daily bar only, annotated like the single-stock page."""
    if len(daily) < DAILY.start:
        return pd.DataFrame()
    n = len(daily)
    orig = replay(daily, horizons=DAILY.horizons, suffix=DAILY.suffix, start=n)
    new = doc_rules.replay_doc(daily, DAILY.horizons, DAILY.suffix, n)
    parts = [x for x in (orig, new) if not x.empty]
    if not parts:
        return pd.DataFrame()
    both = pd.concat(parts, ignore_index=True)
    weekly = bars_for(daily, WEEKLY)
    return doc_rules.annotate(daily, both, doc_rules.bigger_picture(daily, weekly))


def volume_ratio(daily: pd.DataFrame) -> float:
    """Latest volume divided by the average volume of the 20 trading days before it."""
    v = daily["volume"].to_numpy(float)
    if len(v) < 21:
        return float("nan")
    avg = v[-21:-1].mean()
    return float(v[-1] / avg) if avg > 0 else float("nan")


def scan_one(symbol: str, daily: pd.DataFrame) -> list[dict]:
    """One row per pattern on the latest bar; one row with pattern '' if only the volume flag is set."""
    if daily.empty or len(daily) < 25:
        return []
    last = daily.index[-1]
    close, prev = float(daily["close"].iloc[-1]), float(daily["close"].iloc[-2])
    base = {
        "symbol": symbol, "date": last.strftime("%Y-%m-%d"), "close": round(close, 2),
        "change_pct": round((close / prev - 1) * 100, 2) if prev else np.nan,
        "volume": int(daily["volume"].iloc[-1]), "volume_x": round(volume_ratio(daily), 2),
    }
    rows = []
    try:
        pats = latest_patterns(daily)
    except Exception:
        pats = pd.DataFrame()
    for r in pats.itertuples():
        rows.append({**base, "pattern": r.pattern, "rule": r.rule, "direction": r.direction,
                     "meets_rules": bool(r.meets_rules), "signs": r.signs, "location": r.location})
    if not rows and base["volume_x"] >= VOLUME_SHOCK:
        rows.append({**base, "pattern": "", "rule": "", "direction": "", "meets_rules": False,
                     "signs": "", "location": ""})
    return rows


# ── Data ────────────────────────────────────────────────────────────────────

def nse_universe() -> pd.DataFrame:
    """Every NSE stock Yahoo Finance lists, biggest first: symbol, name, market_cap."""
    import yfinance as yf
    from yfinance import EquityQuery as Q

    query = Q("and", [Q("eq", ["exchange", "NSI"]), Q("gt", ["intradaymarketcap", 0])])
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        for attempt in range(4):
            try:
                r = yf.screen(query, size=250, offset=offset, sortField="intradaymarketcap", sortAsc=False)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        total = r.get("total", 0)
        quotes = r.get("quotes", [])
        if not quotes:
            break
        rows += [{"symbol": q["symbol"], "name": q.get("longName") or q.get("shortName") or q["symbol"],
                  "market_cap": q.get("marketCap")} for q in quotes if q.get("quoteType") == "EQUITY"]
        offset += len(quotes)
    return pd.DataFrame(rows).drop_duplicates("symbol")


def _tidy(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "ts"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["volume"] = df["volume"].fillna(0)
    return df.astype(float)


def download_many(symbols: list[str], period: str = SCAN_PERIOD, chunk: int = 80, pause: float = 2.0):
    """Yield (symbol, daily OHLCV) for each symbol, downloading in chunks with retries."""
    import yfinance as yf

    for k in range(0, len(symbols), chunk):
        batch = symbols[k:k + chunk]
        raw = None
        for attempt in range(4):
            try:
                raw = yf.download(batch, period=period, interval="1d", auto_adjust=True, progress=False,
                                  threads=True, group_by="ticker")
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(10 * (attempt + 1))
        if raw is None or raw.empty:
            continue
        for s in batch:
            try:
                sub = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = _tidy(sub)
            except Exception:
                continue
            if not df.empty:
                yield s, df
        time.sleep(pause)
