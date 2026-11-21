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


class ProviderBusy(DataError):
    """Yahoo Finance refused or dropped the request (rate limit / network)."""


_TRANSIENT = ("rate limit", "too many requests", "timed out", "timeout", "connection", "curl", "429", "503")


def _download(ticker: str, attempts: int = 3) -> pd.DataFrame:
    """
    Download full daily history. yf.download() swallows errors and returns an
    empty frame, so check yfinance's error log to tell "no such ticker" apart
    from "Yahoo is busy" (common on shared hosts like Streamlit Cloud), and
    retry the busy case with backoff.
    """
    import time

    import yfinance as yf
    import yfinance.shared as yf_shared

    raw, last_err = None, ""
    for attempt in range(attempts):
        try:
            raw = yf.download(
                ticker,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            last_err = str(getattr(yf_shared, "_ERRORS", {}).get(ticker, ""))
        except Exception as e:  # newer yfinance may raise instead
            raw, last_err = None, f"{type(e).__name__}: {e}"
        if raw is not None and not raw.empty:
            break
        if not any(k in last_err.lower() for k in _TRANSIENT) and "RateLimit" not in last_err:
            break  # a real "not found": retrying won't help
        if attempt < attempts - 1:
            time.sleep(2 * (attempt + 1))
    else:
        raise ProviderBusy(
            f"Yahoo Finance is busy right now and didn't send data for {ticker}. "
            "Please try again in a minute."
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


# ── Markets and company-name search ─────────────────────────────────────────

MARKETS = {
    "US": {"label": "🇺🇸 US", "exchanges": {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NAS", "NYS"}, "suffix": ""},
    "India": {"label": "🇮🇳 India (NSE)", "exchanges": {"NSI"}, "suffix": ".NS"},
}


def search_symbols(query: str, market: str, limit: int = 8) -> list[dict]:
    """Yahoo Finance search, filtered to stocks/ETFs on the chosen market."""
    import yfinance as yf

    exchanges = MARKETS[market]["exchanges"]
    try:
        quotes = yf.Search(query, max_results=20).quotes
    except Exception:
        return []
    out = []
    for q in quotes:
        if q.get("exchange") in exchanges and q.get("quoteType") in {"EQUITY", "ETF"}:
            out.append({
                "symbol": q["symbol"],
                "name": q.get("longname") or q.get("shortname") or q["symbol"],
            })
    return out[:limit]


def _has_recent_prices(symbol: str) -> bool:
    try:
        import yfinance as yf

        return not yf.Ticker(symbol).history(period="5d", auto_adjust=True).empty
    except Exception:
        return False


def resolve(query: str, market: str) -> tuple[list[dict], int]:
    """
    Turn what the user typed (ticker or company name) into candidate symbols
    for the chosen market. Returns (candidates, index of the best match).
    """
    q = (query or "").strip()
    if not q:
        return [], 0
    suffix = MARKETS[market]["suffix"]
    guess = q.upper().replace(" ", "")
    if suffix and not guess.endswith(suffix):
        guess = guess.split(".")[0] + suffix
    try:
        guess = normalize_ticker(guess)
    except DataError:
        guess = None
    matches = search_symbols(q, market)
    for i, m in enumerate(matches):
        if guess and m["symbol"].upper() == guess:
            return matches, i
    if guess and " " not in q and len(q) <= 12:
        # Also offer exactly what was typed, as a ticker (first if search found nothing).
        typed = {"symbol": guess, "name": f"{guess} (as typed)"}
        if not matches:
            return [typed], 0
        if _has_recent_prices(guess):
            return [typed] + matches, 0
        return matches + [typed], 0
    return matches, 0
