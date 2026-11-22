"""
Rules from Ritesh's rule document (Nick Santiago method), added next to the
original detectors in rules.py, which are not modified.

What this module adds, on any timeframe (daily, weekly or monthly bars):

1. Trend label (doc section 4): strong up / up / sideways / down / strong down,
   from the 20/50/200 simple averages and whether they are spaced apart.
2. Three location patterns (doc sections 6.1, 7.1, 7.3, 7.4):
     MAJOR_MA_SUPPORT / MAJOR_MA_RESISTANCE  first touch of the 50/100/200
         average after an extended move
     DROP_AND_POP / TAG_AND_TANK  first pullback to the 20 average after a
         breakout (or first rally to it after a breakdown)
     BOW_TIE_LONG / BOW_TIE_SHORT  price reaches the spot where the 20 and 50
         averages meet, coming from an extended move
   Like the original detectors they need 2+ factors: the location itself plus
   at least one supporting sign.
3. Supporting signs (doc section 6.2) for every pattern, old and new: prior
   pivot, round number, gap window, scene of the crime, Fibonacci level,
   inside bar, narrow body, wide-range bar, time count, trend agreement.
   A pattern's own label never counts as its own sign (an inside-bar pattern
   doesn't also score "inside bar").
4. The doc's entry checklist (sections 10.1, 13): a meaningful location, at
   least 2 factors in total, and the bigger-picture trend not against it.

"Extended move" follows doc section 20.1: at least 5 bars in a row without
touching the average, and at some point in the last 20 bars at least 1.5 ATR
away from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rules

EXT_ATR = 1.5         # doc 20.1: ATR multiple that counts as extended
EXT_BARS = 5          # doc 20.1: bars without touching the average
TOUCH_ATR = 0.10      # a close this close to an average counts as a touch
LEVEL_ATR = 0.25      # how close price must be to a level (pivot, gap, fib...)
LOOKBACK = 60         # how far back to look for pivots, gaps, scenes of the crime
FIB_LEVELS = (0.382, 0.5, 0.618)
DOC_TIME_COUNTS = {3, 7, 21}  # doc 8.1

TREND_TEXT = {"STRONG_UP": "strong up", "UP": "up", "NEUTRAL": "sideways",
              "DOWN": "down", "STRONG_DOWN": "strong down"}

PATTERN_TEXT = {
    "MAJOR_MA_SUPPORT": "Major average support",
    "MAJOR_MA_RESISTANCE": "Major average resistance",
    "DROP_AND_POP": "First pullback to the 20 (drop and pop)",
    "TAG_AND_TANK": "First rally to the 20 (tag and tank)",
    "BOW_TIE_LONG": "Bow tie support",
    "BOW_TIE_SHORT": "Bow tie resistance",
}

SIGN_TEXT = {
    "trend": "trend agrees", "prior_pivot": "prior pivot", "round_number": "round number",
    "gap_window": "gap window", "scene_of_crime": "scene of the crime", "fib": "Fibonacci level",
    "inside_bar": "inside bar", "narrow_body": "narrow body", "wide_range": "wide-range bar",
    "time_count": "time count",
}
LOCATION_TEXT = {"major_ma": "major average", "first_pullback_20": "first pullback to the 20",
                 "bow_tie": "bow tie", "continuation": "flag/wedge zone"}

# Signs that are a pattern's own definition, so they don't count twice.
_OWN_SIGNS = {
    "INSIDE_BAR": {"inside_bar"}, "TIME_COUNT": {"time_count", "narrow_body"},
    "EXTENDED_MOVE": {"narrow_body"}, "WIDE_RANGE": {"wide_range"},
}
_CONTINUATION = ("BULL_FLAG", "BEAR_FLAG", "BULL_WEDGE", "BEAR_WEDGE")


# ── Trend (doc 4) ────────────────────────────────────────────────────────────

def _spaced(fast: pd.Series, slow: pd.Series, bars: int = 5) -> tuple[pd.Series, pd.Series]:
    """(spaced up, spaced down): same side for `bars` bars and the gap not collapsing."""
    gap = fast - slow
    up = (gap > 0).astype(float).rolling(bars).min() == 1
    down = (gap < 0).astype(float).rolling(bars).min() == 1
    holding = gap.abs() >= 0.9 * gap.abs().shift(bars - 1)
    return up & holding, down & holding


def trend_series(d: pd.DataFrame) -> pd.Series:
    """Doc 4.1-4.3 trend label for every bar of an indicator frame."""
    c, m20, m50, m200 = d["close"], d["ma20"], d["ma50"], d["ma200"]
    up20, dn20 = _spaced(m20, m50)
    up200, dn200 = _spaced(m50, m200)
    bull = (c > m20) & (c > m50) & (m20 > m50) & up20
    bear = (c < m20) & (c < m50) & (m20 < m50) & dn20
    out = pd.Series("NEUTRAL", index=d.index)
    out[bull] = "UP"
    out[bull & (m50 > m200) & up200] = "STRONG_UP"
    out[bear] = "DOWN"
    out[bear & (m50 < m200) & dn200] = "STRONG_DOWN"
    return out


def bigger_picture(bars: pd.DataFrame, bigger_bars: pd.DataFrame) -> pd.Series:
    """
    For each bar, the trend on the bigger timeframe as of the last bigger bar
    that had FINISHED by then (daily -> last finished week, weekly -> last
    finished month). Bars with no finished bigger bar yet get NEUTRAL.
    """
    if bigger_bars.empty or bars.empty:
        return pd.Series("NEUTRAL", index=bars.index)
    tr = trend_series(rules._add_indicators(bigger_bars))
    # bigger bar is dated by its last trading day; it is known from the next day on
    known = pd.Series(tr.to_numpy(), index=tr.index + pd.Timedelta(days=1))
    out = known.reindex(known.index.union(bars.index)).ffill().reindex(bars.index)
    return out.fillna("NEUTRAL")


# ── Helpers on numpy arrays ─────────────────────────────────────────────────

class _Arrays:
    def __init__(self, d: pd.DataFrame):
        self.o = d["open"].to_numpy(float)
        self.h = d["high"].to_numpy(float)
        self.l = d["low"].to_numpy(float)
        self.c = d["close"].to_numpy(float)
        self.atr = d["atr14"].to_numpy(float)
        self.rng = d["range"].to_numpy(float)
        self.nbb = d["is_nbb"].to_numpy(bool)
        self.ma = {p: d[f"ma{p}"].to_numpy(float) for p in (20, 50, 100, 200)}
        self.avg_rng = d["range"].rolling(20).mean().to_numpy(float)
        self.n = len(d)


def _touch(a: _Arrays, level: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return ((a.l <= level) & (level <= a.h)) | (np.abs(a.c - level) <= TOUCH_ATR * a.atr)


def _extended_before(a: _Arrays, level: np.ndarray, touched: np.ndarray, i: int) -> bool:
    """Doc 20.1: no touch for EXT_BARS bars before i, and >= EXT_ATR ATR away in the last 20."""
    if i < 20 or np.isnan(a.atr[i]) or a.atr[i] <= 0:
        return False
    if touched[i - EXT_BARS:i].any():
        return False
    dist = np.abs(a.c[i - 20:i] - level[i - 20:i])
    return bool(np.nanmax(dist) >= EXT_ATR * a.atr[i]) if not np.isnan(dist).all() else False


def _pivots(a: _Arrays, lo: int, hi: int) -> list[float]:
    """Swing highs/lows (2 bars each side) with index in [lo, hi)."""
    out = []
    for k in range(max(lo, 2), min(hi, a.n - 2)):
        if a.h[k] > max(a.h[k - 2:k].max(), a.h[k + 1:k + 3].max()):
            out.append(a.h[k])
        if a.l[k] < min(a.l[k - 2:k].min(), a.l[k + 1:k + 3].min()):
            out.append(a.l[k])
    return out


def _bars_since_pivot(a: _Arrays, i: int) -> int | None:
    """Bars since the most recent 1-bar-each-side swing high/low before i (as rules.py defines pivots)."""
    for k in range(i - 1, max(i - rules.PIVOT_LOOKBACK, 1), -1):
        if k + 1 > i:
            continue
        if (a.h[k] > a.h[k - 1] and a.h[k] > a.h[k + 1]) or (a.l[k] < a.l[k - 1] and a.l[k] < a.l[k + 1]):
            return i - k
    return None


# ── Supporting signs (doc 6.2) ──────────────────────────────────────────────

def signs_at(a: _Arrays, i: int, direction: str, trend: str) -> list[str]:
    atr = a.atr[i]
    if np.isnan(atr) or atr <= 0 or i < 2:
        return []
    tol = LEVEL_ATR * atr
    edge = a.l[i] if direction == "LONG" else a.h[i]  # the side of the bar that tests the level
    near = lambda lvl: abs(edge - lvl) <= tol or abs(a.c[i] - lvl) <= tol or a.l[i] <= lvl <= a.h[i]
    out: list[str] = []

    if (direction == "LONG" and trend in ("UP", "STRONG_UP")) or (direction == "SHORT" and trend in ("DOWN", "STRONG_DOWN")):
        out.append("trend")

    lo = max(0, i - LOOKBACK)
    if any(abs(edge - p) <= tol for p in _pivots(a, lo, i - 2)):
        out.append("prior_pivot")

    if rules._is_near_round_level(float(a.c[i]), float(atr)) is not None:
        out.append("round_number")

    # gap window: an earlier gap whose zone was not filled since, reached today
    for k in range(i - 1, max(lo, 1) - 1, -1):
        if a.l[k] > a.h[k - 1]:      # gap up: zone [h[k-1], l[k]]
            z0, z1 = a.h[k - 1], a.l[k]
            filled = a.l[k + 1:i].min() <= z0 if k + 1 < i else False
        elif a.h[k] < a.l[k - 1]:    # gap down: zone [h[k], l[k-1]]
            z0, z1 = a.h[k], a.l[k - 1]
            filled = a.h[k + 1:i].max() >= z1 if k + 1 < i else False
        else:
            continue
        if not filled and a.l[i] <= z1 + tol and a.h[i] >= z0 - tol and k < i - 1:
            out.append("gap_window")
            break

    # scene of the crime: where an earlier big move started (open of a wide-range bar)
    for k in range(lo + 20, i - 4):
        if not np.isnan(a.avg_rng[k - 1]) and a.rng[k] >= 2.0 * a.avg_rng[k - 1] and near(a.o[k]):
            out.append("scene_of_crime")
            break

    # Fibonacci 38/50/62% of the last swing (highest high / lowest low of 40 bars)
    w0 = max(0, i - 40)
    hh, ll = a.h[w0:i].max(), a.l[w0:i].min()
    if hh - ll >= 3 * atr:
        if any(near(hh - f * (hh - ll)) for f in FIB_LEVELS):
            out.append("fib")

    if a.h[i] <= a.h[i - 1] and a.l[i] >= a.l[i - 1]:
        out.append("inside_bar")
    if a.nbb[i]:
        out.append("narrow_body")
    if not np.isnan(a.avg_rng[i - 1]) and a.rng[i] >= 2.0 * a.avg_rng[i - 1]:
        out.append("wide_range")
    if _bars_since_pivot(a, i) in DOC_TIME_COUNTS:
        out.append("time_count")
    return out


# ── Location patterns (doc 6.1, 7.1, 7.3, 7.4) ───────────────────────────────

def _locations_at(a: _Arrays, i: int, trend: str, touched: dict) -> list[tuple[str, str, str]]:
    """[(pattern, direction, location code)] for the doc's location patterns on bar i."""
    out = []
    c = a.c[i]
    # Major 50/100/200 average, first touch after an extended move
    for p in (50, 100, 200):
        ma = a.ma[p]
        if np.isnan(ma[i]) or not touched[p][i] or not _extended_before(a, ma, touched[p], i):
            continue
        came_from_above = np.nanmean(a.c[i - EXT_BARS:i]) > ma[i]
        if came_from_above and c > ma[i] and trend not in ("DOWN", "STRONG_DOWN"):
            out.append(("MAJOR_MA_SUPPORT", "LONG", "major_ma"))
        elif not came_from_above and c < ma[i] and trend not in ("UP", "STRONG_UP"):
            out.append(("MAJOR_MA_RESISTANCE", "SHORT", "major_ma"))
        break  # nearest qualifying average only
    # First pullback to the 20 after a breakout / first rally to it after a breakdown
    ma20 = a.ma[20]
    if not np.isnan(ma20[i]) and touched[20][i] and _extended_before(a, ma20, touched[20], i):
        if trend in ("UP", "STRONG_UP") and (a.c[i - EXT_BARS:i] > ma20[i - EXT_BARS:i]).all() and c > ma20[i]:
            out.append(("DROP_AND_POP", "LONG", "first_pullback_20"))
        if trend in ("DOWN", "STRONG_DOWN") and (a.c[i - EXT_BARS:i] < ma20[i - EXT_BARS:i]).all() and c < ma20[i]:
            out.append(("TAG_AND_TANK", "SHORT", "first_pullback_20"))
    # Bow tie: 20 and 50 averages meeting, reached from an extended move
    m20, m50 = a.ma[20][i], a.ma[50][i]
    if not (np.isnan(m20) or np.isnan(m50)) and abs(m20 - m50) <= 0.25 * a.atr[i]:
        mid = (a.ma[20] + a.ma[50]) / 2
        if touched["bow"][i] and _extended_before(a, mid, touched["bow"], i):
            if np.nanmean(a.c[i - EXT_BARS:i]) > mid[i] and c > mid[i]:
                out.append(("BOW_TIE_LONG", "LONG", "bow_tie"))
            elif np.nanmean(a.c[i - EXT_BARS:i]) < mid[i] and c < mid[i]:
                out.append(("BOW_TIE_SHORT", "SHORT", "bow_tie"))
    return out


def _touch_map(a: _Arrays) -> dict:
    t = {p: _touch(a, a.ma[p]) for p in (20, 50, 100, 200)}
    t["bow"] = _touch(a, (a.ma[20] + a.ma[50]) / 2)
    return t


def location_codes(a: _Arrays, i: int, direction: str, pattern: str, trend: str, touched: dict) -> list[str]:
    """Location factors for ANY pattern (original or new) on bar i, in its direction."""
    codes = [loc for _, dr, loc in _locations_at(a, i, trend, touched) if dr == direction]
    if pattern.startswith(_CONTINUATION):
        codes.append("continuation")
    return list(dict.fromkeys(codes))


def replay_doc(df: pd.DataFrame, horizons, suffix: str, start: int) -> pd.DataFrame:
    """Same output columns as history.replay(), for the doc's location patterns."""
    cols = ["date", "rule", "pattern", "direction", "confidence", "close", "factors"] + [f"ret_{h}{suffix}" for h in horizons]
    if len(df) < max(start, 25):
        return pd.DataFrame(columns=cols)
    d = rules._add_indicators(df)
    a, trend = _Arrays(d), trend_series(d).to_numpy()
    touched = _touch_map(a)
    rows = []
    for i in range(max(start - 1, 21), a.n):
        for pattern, direction, loc in _locations_at(a, i, trend[i], touched):
            signs = signs_at(a, i, direction, trend[i])
            if len(signs) < 1:  # location + at least one supporting sign = 2 factors
                continue
            row = {"date": d.index[i], "rule": "doc_" + loc, "pattern": pattern, "direction": direction,
                   "confidence": min(1 + len(signs), 5), "close": a.c[i],
                   "factors": ", ".join([loc] + signs)}
            for h in horizons:
                j = i + h
                row[f"ret_{h}{suffix}"] = a.c[j] / a.c[i] - 1.0 if j < a.n else np.nan
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def annotate(df: pd.DataFrame, hist: pd.DataFrame, bigger: pd.Series | None = None) -> pd.DataFrame:
    """
    Add the doc's checklist to every row of a replay (original or doc patterns):
      trend        this timeframe's trend label (plain words)
      bigger       bigger-picture trend (plain words), or "" if none
      location     location factors (plain words)
      signs        supporting signs (plain words)
      factor_count location (0/1) + number of supporting signs
      meets_rules  location, 2+ factors, bigger picture not against it
    """
    out = hist.copy()
    new_cols = ["trend", "bigger", "location", "signs"]
    if out.empty:
        for col in new_cols:
            out[col] = pd.Series(dtype=str)
        out["factor_count"] = pd.Series(dtype=int)
        out["meets_rules"] = pd.Series(dtype=bool)
        return out
    d = rules._add_indicators(df)
    a, trend = _Arrays(d), trend_series(d)
    touched = _touch_map(a)
    pos = pd.Series(np.arange(len(d)), index=d.index)
    big = bigger.reindex(d.index).fillna("NEUTRAL") if bigger is not None else None
    T, B, L, S, F, M = [], [], [], [], [], []
    for date, direction, pattern in zip(out["date"], out["direction"], out["pattern"]):
        i = int(pos[date])
        tr = trend.iloc[i]
        own = next((v for k, v in _OWN_SIGNS.items() if pattern.startswith(k)), set())
        signs = [s for s in signs_at(a, i, direction, tr) if s not in own]
        locs = location_codes(a, i, direction, pattern, tr, touched)
        bp = big.iloc[i] if big is not None else "NEUTRAL"
        against = (direction == "LONG" and bp == "STRONG_DOWN") or (direction == "SHORT" and bp == "STRONG_UP")
        count = (1 if locs else 0) + len(signs)
        T.append(TREND_TEXT[tr])
        B.append(TREND_TEXT[bp] if big is not None else "")
        L.append(", ".join(LOCATION_TEXT[x] for x in locs))
        S.append(", ".join(SIGN_TEXT[x] for x in signs))
        F.append(count)
        M.append(bool(locs) and count >= 2 and not against)
    out["trend"], out["bigger"], out["location"], out["signs"] = T, B, L, S
    out["factor_count"], out["meets_rules"] = F, M
    return out


def why_not(row) -> str:
    """Plain reason a pattern misses the doc's checklist."""
    reasons = []
    if not row["location"]:
        reasons.append("not at a meaningful location")
    if row["factor_count"] < 2:
        reasons.append("fewer than 2 factors")
    if (row["direction"] == "LONG" and row["bigger"] == "strong down") or \
       (row["direction"] == "SHORT" and row["bigger"] == "strong up"):
        reasons.append(f"bigger-picture trend is {row['bigger']}")
    return "; ".join(reasons)
