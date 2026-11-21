"""
Daily price data for ThePatternHistory.

- Source: yfinance, split/dividend-adjusted (auto_adjust=True), full history.
- Cache: one parquet file per ticker under CACHE_DIR, refreshed once per day.
- Tickers: US symbols as-is (AAPL, BRK-B), Indian NSE symbols with ".NS".
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(os.getenv("PATTERN_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache" / "bars"))

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")


class DataError(Exception):
    """Raised when a ticker cannot be resolved to usable price data."""


def normalize_ticker(raw: str) -> str:
    """Upper-case, strip, and validate a user-entered ticker."""
    t = (raw or "").strip().upper().replace(" ", "")
    # Common habit: BRK.B -> BRK-B for Yahoo (but keep .NS / .BO suffixes)
    if "." in t and not t.endswith((".NS", ".BO")):
        t = t.replace(".", "-")
    if not t or not _TICKER_RE.match(t):
        raise DataError(f"'{raw}' doesn't look like a ticker. Try e.g. AAPL or RELIANCE.NS.")
    return t


def _cache_path(ticker: str) -> Path:
    safe = re.sub(r"[^A-Z0-9.\-]", "_", ticker)
    return CACHE_DIR / f"{safe}.parquet"


def _download(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "ts"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["volume"] = df["volume"].fillna(0)
    return df.astype(float)


def load_daily(ticker: str, years: int = 25, use_cache: bool = True) -> pd.DataFrame:
    """
    Return adjusted daily OHLCV (columns open/high/low/close/volume, ascending
    DatetimeIndex) for roughly the last `years` years.
    """
    ticker = normalize_ticker(ticker)
    path = _cache_path(ticker)

    df = None
    if use_cache and path.exists():
        modified = date.fromtimestamp(path.stat().st_mtime)
        if modified == date.today():
            try:
                df = pd.read_parquet(path)
            except Exception:
                df = None

    if df is None:
        df = _download(ticker)
        if df.empty:
            raise DataError(
                f"No price data found for {ticker}. Check the symbol "
                "(Indian NSE stocks need the .NS suffix, e.g. TCS.NS)."
            )
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
        except Exception:
            pass  # cache is best-effort

    if years:
        cutoff = df.index[-1] - pd.DateOffset(years=years)
        df = df[df.index >= cutoff]
    return df
