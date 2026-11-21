"""
R3 – Trading rules engine.

Nick Santiago methodology, coded deterministically over OHLCV bars.

Data contract:
    df.index   : DatetimeIndex ascending
    df.columns : open, high, low, close, volume

Usage:
    from market_data import load_bars
    from rules import scan

    df = load_bars("AAPL", n_days=300)
    sig = scan("AAPL", df)
    print(sig)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import pandas as pd


# ── Tunables ────────────────────────────────────────────────────────────────

MIN_BARS = 220
MIN_BARS_INTRADAY = 50   # ~2 trading days of 15m bars
MIN_FACTORS = 2

NBB_BODY_ATR_RATIO = 0.28
NBB_MAX_FULL_RANGE_ATR = 1.25

TIME_COUNT_IDEAL = 7
TIME_COUNT_VALID = {3, 6, 7, 8, 9, 10, 14, 21}

EXTENDED_ATR_MULT = 2.5
MAJOR_MIN_DISTANCE_ATR = 2.0

FLAG_MAX_RETRACE = 0.30
FLAG_MIN_POLE_ATR = 1.8
FLAG_CONSOL_BARS = range(3, 13)

WEDGE_LOOKBACK = 12
DOUBLE_TOP_TOL = 0.015
ROUND_LEVEL_TOL_ATR = 0.20
PIVOT_LOOKBACK = 40


# ── Output ──────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    symbol: str
    direction: str
    pattern: str
    confidence: int
    entry: float
    stop: float
    target: float
    r_multiple: float = 0.0
    factors: list[str] = field(default_factory=list)
    notes: str = ""
    strategy_type: str = "SWING"   # "SWING" | "DAY_TRADE"
    timeframe: str = "daily"       # "daily" | "5m" | "15m" | "60m"

    def __post_init__(self) -> None:
        if self.direction != "FLAT" and self.entry and self.stop:
            risk = abs(self.entry - self.stop)
            reward = abs(self.target - self.entry)
            self.r_multiple = round(reward / risk, 2) if risk > 0 else 0.0

    @property
    def rationale(self) -> str:
        """Human-readable explanation for summaries and logs."""
        lines = [
            f"{self.strategy_type} {self.direction} {self.symbol} | {self.pattern} "
            f"| conf={self.confidence} R={self.r_multiple}x | tf={self.timeframe}",
            f"  Entry={self.entry:.2f}  Stop={self.stop:.2f}  Target={self.target:.2f}",
            f"  Factors: {', '.join(self.factors)}",
        ]
        if self.notes:
            lines.append(f"  Context: {self.notes}")
        return "\n".join(lines)

    def __str__(self) -> str:
        if self.direction == "FLAT":
            return f"[{self.symbol}] FLAT"
        return (
            f"[{self.symbol}] {self.strategy_type} {self.direction} pattern={self.pattern} "
            f"conf={self.confidence} entry={self.entry:.2f} stop={self.stop:.2f} "
            f"target={self.target:.2f} R={self.r_multiple}x tf={self.timeframe}"
        )


def _flat(symbol: str, strategy_type: str = "SWING", timeframe: str = "daily") -> Signal:
    return Signal(symbol, "FLAT", "none", 0, 0.0, 0.0, 0.0,
                  strategy_type=strategy_type, timeframe=timeframe)


# ── Indicators ───────────────────────────────────────────────────────────────

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ma20"]  = _sma(d["close"], 20)
    d["ma50"]  = _sma(d["close"], 50)
    d["ma100"] = _sma(d["close"], 100)
    d["ma200"] = _sma(d["close"], 200)
    d["atr14"] = _atr(d, 14)

    body = (d["close"] - d["open"]).abs()
    full_range = d["high"] - d["low"]

    d["body"] = body
    d["range"] = full_range
    d["is_red"] = d["close"] < d["open"]
    d["is_green"] = d["close"] > d["open"]
    d["is_nbb"] = (
        (body <= NBB_BODY_ATR_RATIO * d["atr14"])
        & (full_range <= NBB_MAX_FULL_RANGE_ATR * d["atr14"])
    )
    return d


# ── Helpers ─────────────────────────────────────────────────────────────────

def _ma_spacing_widening(d: pd.DataFrame) -> bool:
    """True if the MA20-MA50 gap has expanded over the last 5 bars."""
    if len(d) < 5:
        return False
    recent = d.iloc[-5:]
    gaps = (recent["ma20"] - recent["ma50"]).abs()
    if gaps.isna().any():
        return False
    return float(gaps.iloc[-1]) > float(gaps.iloc[0])


def _trend(d: pd.DataFrame) -> str:
    """
    Returns STRONG_UP, UP, STRONG_DOWN, DOWN, or NEUTRAL.
    STRONG = all three MAs aligned AND MA20-MA50 spacing widening.
    """
    r = d.iloc[-1]
    if pd.isna(r["ma20"]) or pd.isna(r["ma50"]) or pd.isna(r["ma200"]):
        return "NEUTRAL"
    ma20, ma50, ma200 = float(r["ma20"]), float(r["ma50"]), float(r["ma200"])
    widening = _ma_spacing_widening(d)
    if ma20 > ma50 > ma200:
        return "STRONG_UP" if widening else "UP"
    if ma20 < ma50 < ma200:
        return "STRONG_DOWN" if widening else "DOWN"
    return "NEUTRAL"


def _is_uptrend(d: pd.DataFrame) -> bool:
    return _trend(d) in {"UP", "STRONG_UP"}


def _is_downtrend(d: pd.DataFrame) -> bool:
    return _trend(d) in {"DOWN", "STRONG_DOWN"}


def _technical_position(d: pd.DataFrame) -> str:
    r = d.iloc[-1]
    if pd.isna(r["ma20"]) or pd.isna(r["ma50"]):
        return "MIXED"
    if r["close"] > r["ma20"] > r["ma50"]:
        return "STRONG"
    if r["close"] < r["ma50"]:
        return "WEAK"
    return "MIXED"


def _bow_tie(d: pd.DataFrame) -> Optional[str]:
    """Detect MA20/MA50 crossover OR convergence (gap narrows 30%+)."""
    if len(d) < 4:
        return None
    recent = d.iloc[-4:]
    a0, b0 = recent["ma20"].iloc[0], recent["ma50"].iloc[0]
    a1, b1 = recent["ma20"].iloc[-1], recent["ma50"].iloc[-1]
    if pd.isna(a0) or pd.isna(b0) or pd.isna(a1) or pd.isna(b1):
        return None
    # Crossover
    if a0 <= b0 and a1 > b1:
        return "BULL"
    if a0 >= b0 and a1 < b1:
        return "BEAR"
    # Convergence: gap narrowed by 30%+ (approaching crossover)
    prior_gap = abs(float(a0) - float(b0))
    curr_gap = abs(float(a1) - float(b1))
    if prior_gap > 0 and curr_gap < 0.70 * prior_gap:
        # MA20 below MA50 but rising toward it -> bullish
        if float(a1) < float(b1) and float(a1) > float(a0):
            return "BULL"
        # MA20 above MA50 but falling toward it -> bearish
        if float(a1) > float(b1) and float(a1) < float(a0):
            return "BEAR"
    return None


def _nearest_round_level(price: float) -> float:
    if price >= 100:
        step = 5.0
    elif price >= 20:
        step = 1.0
    elif price >= 5:
        step = 0.5
    else:
        step = 0.25
    return round(price / step) * step


def _is_near_round_level(price: float, atr: float) -> Optional[float]:
    level = _nearest_round_level(price)
    if atr <= 0:
        return None
    if abs(price - level) <= ROUND_LEVEL_TOL_ATR * atr:
        return level
    return None


def _offset_stop(direction: str, raw_stop: float, atr: float) -> float:
    atr = atr if pd.notna(atr) and atr > 0 else 0.25
    buffer_amt = max(0.12 * atr, 0.02)

    level = _nearest_round_level(raw_stop)
    too_close = abs(raw_stop - level) <= max(0.10 * atr, 0.03)

    if direction == "LONG":
        stop = raw_stop - buffer_amt
        if too_close:
            stop = min(stop, level - buffer_amt)
    else:
        stop = raw_stop + buffer_amt
        if too_close:
            stop = max(stop, level + buffer_amt)

    return stop


def _pivot_high(d: pd.DataFrame, idx: int) -> bool:
    if idx < 1 or idx >= len(d) - 1:
        return False
    return d["high"].iloc[idx] > d["high"].iloc[idx - 1] and d["high"].iloc[idx] > d["high"].iloc[idx + 1]


def _pivot_low(d: pd.DataFrame, idx: int) -> bool:
    if idx < 1 or idx >= len(d) - 1:
        return False
    return d["low"].iloc[idx] < d["low"].iloc[idx - 1] and d["low"].iloc[idx] < d["low"].iloc[idx + 1]


def _last_pivot(d: pd.DataFrame, side: str, lookback: int = PIVOT_LOOKBACK) -> Optional[tuple[int, float]]:
    window = d.iloc[-lookback:]
    if len(window) < 5:
        return None

    for i in range(len(window) - 2, 1, -1):
        if side == "high" and _pivot_high(window, i):
            return i, float(window["high"].iloc[i])
        if side == "low" and _pivot_low(window, i):
            return i, float(window["low"].iloc[i])
    return None


def _measured_move(start: float, end: float, from_price: float, direction: str) -> float:
    size = abs(end - start)
    return from_price + size if direction == "LONG" else from_price - size


def _is_extended_from_ma(d: pd.DataFrame, ma_period: int = 20) -> bool:
    r = d.iloc[-1]
    ma = r[f"ma{ma_period}"]
    atr = r["atr14"]
    if pd.isna(ma) or pd.isna(atr) or atr <= 0:
        return False
    return abs(r["close"] - ma) >= EXTENDED_ATR_MULT * atr


def _ma_significance(d: pd.DataFrame, period: int) -> str:
    """
    Classify the current MA touch:
      MAJOR  - first clean approach from well-extended distance (>= 2 ATR away)
      MINOR  - repeated touch or has been consolidating near MA
      UNKNOWN - unclear context; Rule 3.5: skip the trade
    """
    if len(d) < 20:
        return "UNKNOWN"
    ma_col = f"ma{period}"
    ma_now = d[ma_col].iloc[-1]
    atr_now = d["atr14"].iloc[-1]
    if pd.isna(ma_now) or pd.isna(atr_now) or atr_now <= 0:
        return "UNKNOWN"

    touch_threshold = 0.15 * atr_now
    prior = d.iloc[-21:-1]
    prior_ma = prior[ma_col]
    prior_close = prior["close"]
    prior_distances = (prior_close - prior_ma).abs()

    touches = (prior_distances <= touch_threshold).sum()
    max_prior_dist = (prior_distances / atr_now).max() if atr_now > 0 else 0.0

    if touches == 0 and max_prior_dist >= MAJOR_MIN_DISTANCE_ATR:
        # Clean first approach from well-extended territory
        return "MAJOR"
    if touches >= 1:
        # Was near this MA before — consolidation or multiple touches
        return "MINOR"
    # Price was close but never extended far enough — context unclear
    return "UNKNOWN"


def _factor_pack(
    d: pd.DataFrame,
    direction: str,
    include_trend: bool = True,
    include_nbb: bool = True,
) -> list[str]:
    factors: list[str] = []
    r = d.iloc[-1]
    atr = r["atr14"] if pd.notna(r["atr14"]) else 0.0

    if include_trend:
        tr = _trend(d)
        if direction == "LONG":
            if tr == "STRONG_UP":
                factors.append("strong_trend_up")
            elif tr == "UP":
                factors.append("trend_up")
        if direction == "SHORT":
            if tr == "STRONG_DOWN":
                factors.append("strong_trend_down")
            elif tr == "DOWN":
                factors.append("trend_down")

    tp = _technical_position(d)
    if direction == "LONG" and tp == "STRONG":
        factors.append("strong_technical_position")
    if direction == "SHORT" and tp == "WEAK":
        factors.append("weak_technical_position")

    if include_nbb and bool(r["is_nbb"]):
        factors.append("narrow_body_bar")

    round_level = _is_near_round_level(float(r["close"]), float(atr) if atr else 0.0)
    if round_level is not None:
        factors.append(f"round_number_{round_level:g}")

    avg_vol = d["volume"].iloc[-20:].mean()
    if pd.notna(avg_vol) and avg_vol > 0 and r["volume"] >= 1.2 * avg_vol:
        factors.append("above_average_volume")

    bt = _bow_tie(d)
    if direction == "LONG" and bt == "BULL":
        factors.append("bow_tie_bull")
    if direction == "SHORT" and bt == "BEAR":
        factors.append("bow_tie_bear")

    return factors


def _build_signal(
    symbol: str,
    direction: str,
    pattern: str,
    entry: float,
    raw_stop: float,
    target: float,
    factors: list[str],
    notes: str = "",
    strategy_type: str = "SWING",
    timeframe: str = "daily",
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,
        pattern=pattern,
        confidence=min(max(len(set(factors)), 1), 5),
        entry=round(entry, 2),
        stop=round(raw_stop, 2),
        target=round(target, 2),
        factors=sorted(set(factors)),
        notes=notes,
        strategy_type=strategy_type,
        timeframe=timeframe,
    )


# ── Pattern detectors ───────────────────────────────────────────────────────

def _detect_inside_bar(d: pd.DataFrame) -> Optional[dict]:
    if len(d) < 5:
        return None

    prev = d.iloc[-2]
    curr = d.iloc[-1]
    atr = curr["atr14"]
    if pd.isna(atr) or atr <= 0:
        return None

    inside = curr["high"] <= prev["high"] and curr["low"] >= prev["low"]
    if not inside:
        return None

    if prev["is_red"] and _is_downtrend(d):
        entry = curr["close"]
        raw_stop = _offset_stop("SHORT", float(prev["high"]), float(atr))
        pole_high = d["high"].iloc[-4:-1].max()
        target = entry - max(pole_high - prev["low"], atr)
        factors = ["inside_bar", "prior_red_bar", "close_based_invalidation"] + _factor_pack(d, "SHORT")
        return {
            "direction": "SHORT",
            "pattern": "INSIDE_BAR_BEARISH",
            "entry": entry,
            "stop": raw_stop,
            "target": target,
            "factors": factors,
        }

    if prev["is_green"] and _is_uptrend(d):
        entry = curr["close"]
        raw_stop = _offset_stop("LONG", float(prev["low"]), float(atr))
        pole_low = d["low"].iloc[-4:-1].min()
        target = entry + max(prev["high"] - pole_low, atr)
        factors = ["inside_bar", "prior_green_bar", "close_based_invalidation"] + _factor_pack(d, "LONG")
        return {
            "direction": "LONG",
            "pattern": "INSIDE_BAR_BULLISH",
            "entry": entry,
            "stop": raw_stop,
            "target": target,
            "factors": factors,
        }

    return None


def _detect_reversal_bar_50(d: pd.DataFrame) -> Optional[dict]:
    if len(d) < 3:
        return None
    prev = d.iloc[-2]
    curr = d.iloc[-1]
    atr = curr["atr14"]
    if pd.isna(atr) or atr <= 0:
        return None

    prev_mid = (prev["high"] + prev["low"]) / 2.0

    if prev["is_green"] and curr["close"] < prev_mid and _is_downtrend(d):
        entry = curr["close"]
        stop = _offset_stop("SHORT", float(curr["high"]), float(atr))
        target = entry - max(prev["high"] - prev["low"], atr)
        factors = ["reversal_bar_through_50pct", "bearish_reversal"] + _factor_pack(d, "SHORT", include_nbb=False)
        return {
            "direction": "SHORT",
            "pattern": "REVERSAL_BAR_50_BEARISH",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }

    if prev["is_red"] and curr["close"] > prev_mid and _is_uptrend(d):
        entry = curr["close"]
        stop = _offset_stop("LONG", float(curr["low"]), float(atr))
        target = entry + max(prev["high"] - prev["low"], atr)
        factors = ["reversal_bar_through_50pct", "bullish_reversal"] + _factor_pack(d, "LONG", include_nbb=False)
        return {
            "direction": "LONG",
            "pattern": "REVERSAL_BAR_50_BULLISH",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }

    return None


def _detect_time_count(d: pd.DataFrame) -> Optional[dict]:
    if len(d) < 20:
        return None

    is_up   = _is_uptrend(d)
    is_down = _is_downtrend(d)
    if not (is_up or is_down):
        return None
    if not bool(d["is_nbb"].iloc[-1]):
        return None

    search = d.iloc[-20:]
    pivot = _last_pivot(search, "high" if is_up else "low", 20)
    if pivot is None:
        return None

    pivot_idx, pivot_price = pivot
    count = len(search) - 1 - pivot_idx
    if count not in TIME_COUNT_VALID:
        return None

    atr = d["atr14"].iloc[-1]
    if is_up:
        entry = d["close"].iloc[-1]
        stop = _offset_stop("LONG", float(d["low"].iloc[-1]), float(atr))
        target = float(pivot_price)
        factors = [f"time_count_{count}", "narrow_body_bar"] + _factor_pack(d, "LONG", include_nbb=False)
        return {
            "direction": "LONG",
            "pattern": f"TIME_COUNT_LONG_{count}",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }

    entry = d["close"].iloc[-1]
    stop = _offset_stop("SHORT", float(d["high"].iloc[-1]), float(atr))
    target = float(pivot_price)
    factors = [f"time_count_{count}", "narrow_body_bar"] + _factor_pack(d, "SHORT", include_nbb=False)
    return {
        "direction": "SHORT",
        "pattern": f"TIME_COUNT_SHORT_{count}",
        "entry": entry,
        "stop": stop,
        "target": target,
        "factors": factors,
    }


def _detect_flag(d: pd.DataFrame) -> Optional[dict]:
    if len(d) < 20:
        return None

    atr = d["atr14"].iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None

    for n in FLAG_CONSOL_BARS:
        consol = d.iloc[-n:]
        if len(consol) < n:
            continue

        pole = d.iloc[-(n + 3):-n]
        if len(pole) < 1:
            continue

        consol_high = float(consol["high"].max())
        consol_low = float(consol["low"].min())
        pole_high = float(pole["high"].max())
        pole_low = float(pole["low"].min())
        pole_size = pole_high - pole_low

        if pole_size < FLAG_MIN_POLE_ATR * atr:
            continue

        if _is_downtrend(d):
            if pole["close"].iloc[-1] >= pole["close"].iloc[0]:
                continue
            max_retrace = pole_low + FLAG_MAX_RETRACE * pole_size
            if consol_high > max_retrace:
                continue
            if d["close"].iloc[-1] >= consol_low:
                continue

            entry = float(d["close"].iloc[-1])
            stop = _offset_stop("SHORT", consol_high, float(atr))
            target = entry - pole_size
            factors = ["bear_flag", f"consol_{n}_bars", "flagpole"] + _factor_pack(d, "SHORT")
            return {
                "direction": "SHORT",
                "pattern": "BEAR_FLAG",
                "entry": entry,
                "stop": stop,
                "target": target,
                "factors": factors,
            }

        if _is_uptrend(d):
            if pole["close"].iloc[-1] <= pole["close"].iloc[0]:
                continue
            min_retrace = pole_high - FLAG_MAX_RETRACE * pole_size
            if consol_low < min_retrace:
                continue
            if d["close"].iloc[-1] <= consol_high:
                continue

            entry = float(d["close"].iloc[-1])
            stop = _offset_stop("LONG", consol_low, float(atr))
            target = entry + pole_size
            factors = ["bull_flag", f"consol_{n}_bars", "flagpole"] + _factor_pack(d, "LONG")
            return {
                "direction": "LONG",
                "pattern": "BULL_FLAG",
                "entry": entry,
                "stop": stop,
                "target": target,
                "factors": factors,
            }

    return None


def _detect_wedge(d: pd.DataFrame) -> Optional[dict]:
    if len(d) < WEDGE_LOOKBACK + 3:
        return None

    atr = d["atr14"].iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None

    look = d.iloc[-WEDGE_LOOKBACK:]
    closes = look["close"].reset_index(drop=True)

    if _is_downtrend(d):
        start = float(look["high"].iloc[0])
        trough = float(look["low"].min())
        rebound = float(look["high"].iloc[-1])
        bars_down = sum(closes.diff().fillna(0) <= 0)

        if bars_down < 4:
            return None
        if rebound <= trough:
            return None

        retr = (rebound - trough) / max(start - trough, 1e-9)
        if not (0.35 <= retr <= 0.75):
            return None

        entry = float(d["close"].iloc[-1])
        stop = _offset_stop("SHORT", float(look["high"].max()), float(atr))
        target = _measured_move(start, trough, entry, "SHORT")
        factors = ["wedge", "multiple_bars_down", "symmetry_setup"] + _factor_pack(d, "SHORT")
        return {
            "direction": "SHORT",
            "pattern": "BEAR_WEDGE",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }

    if _is_uptrend(d):
        start = float(look["low"].iloc[0])
        peak = float(look["high"].max())
        pullback = float(look["low"].iloc[-1])
        bars_up = sum(closes.diff().fillna(0) >= 0)

        if bars_up < 4:
            return None
        if pullback >= peak:
            return None

        retr = (peak - pullback) / max(peak - start, 1e-9)
        if not (0.35 <= retr <= 0.75):
            return None

        entry = float(d["close"].iloc[-1])
        stop = _offset_stop("LONG", float(look["low"].min()), float(atr))
        target = _measured_move(start, peak, entry, "LONG")
        factors = ["wedge", "multiple_bars_up", "symmetry_setup"] + _factor_pack(d, "LONG")
        return {
            "direction": "LONG",
            "pattern": "BULL_WEDGE",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }

    return None


def _detect_major_ma_bounce(d: pd.DataFrame) -> Optional[dict]:
    if len(d) < 205:
        return None

    r = d.iloc[-1]
    atr = r["atr14"]
    if pd.isna(atr) or atr <= 0:
        return None

    for period in (20, 50, 100, 200):
        ma = r[f"ma{period}"]
        if pd.isna(ma):
            continue

        touched = r["low"] <= ma <= r["high"] or abs(r["close"] - ma) <= 0.10 * atr
        if not touched:
            continue

        significance = _ma_significance(d, period)
        if significance == "UNKNOWN":
            continue  # Rule 3.5: unclear context -> no trade

        if _is_uptrend(d) and r["close"] >= ma:
            entry = float(r["close"])
            stop = _offset_stop("LONG", float(r["low"]), float(atr))
            target = float(entry + 2 * atr)
            sig_tag = "first_touch_from_extended" if significance == "MAJOR" else "minor_ma_touch"
            factors = [f"ma{period}_{significance.lower()}", sig_tag] + _factor_pack(d, "LONG")
            return {
                "direction": "LONG",
                "pattern": f"MA{period}_{significance}_BOUNCE",
                "entry": entry,
                "stop": stop,
                "target": target,
                "factors": factors,
            }

        if _is_downtrend(d) and r["close"] <= ma:
            entry = float(r["close"])
            stop = _offset_stop("SHORT", float(r["high"]), float(atr))
            target = float(entry - 2 * atr)
            sig_tag = "first_touch_from_extended" if significance == "MAJOR" else "minor_ma_touch"
            factors = [f"ma{period}_{significance.lower()}", sig_tag] + _factor_pack(d, "SHORT")
            return {
                "direction": "SHORT",
                "pattern": f"MA{period}_{significance}_REJECTION",
                "entry": entry,
                "stop": stop,
                "target": target,
                "factors": factors,
            }

    return None


def _detect_extended_move_reversal(d: pd.DataFrame) -> Optional[dict]:
    if not _is_extended_from_ma(d, 20):
        return None

    r = d.iloc[-1]
    atr = r["atr14"]
    ma20 = r["ma20"]
    if pd.isna(atr) or atr <= 0 or pd.isna(ma20):
        return None

    if not bool(r["is_nbb"]):
        return None

    if r["close"] > ma20:
        entry = float(r["close"])
        stop = _offset_stop("SHORT", float(r["high"]), float(atr))
        target = float(ma20)
        factors = ["extended_above_ma20", "narrow_body_bar", "mean_reversion"] + _factor_pack(d, "SHORT", include_trend=False)
        return {
            "direction": "SHORT",
            "pattern": "EXTENDED_MOVE_FADE",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }

    entry = float(r["close"])
    stop = _offset_stop("LONG", float(r["low"]), float(atr))
    target = float(ma20)
    factors = ["extended_below_ma20", "narrow_body_bar", "mean_reversion"] + _factor_pack(d, "LONG", include_trend=False)
    return {
        "direction": "LONG",
        "pattern": "EXTENDED_MOVE_BOUNCE",
        "entry": entry,
        "stop": stop,
        "target": target,
        "factors": factors,
    }


def _detect_double_top_bottom(d: pd.DataFrame) -> Optional[dict]:
    if len(d) < 30:
        return None

    window = d.iloc[-30:]
    atr = d["atr14"].iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None

    highs = window["high"]
    lows = window["low"]
    vols = window["volume"]

    first_top_idx = highs.iloc[:-5].idxmax()
    second_top = float(highs.iloc[-5:].max())
    first_top = float(highs.loc[first_top_idx])

    if abs(second_top - first_top) / max(first_top, 1e-9) <= DOUBLE_TOP_TOL:
        first_vol = float(vols.loc[first_top_idx])
        second_vol = float(vols.iloc[-5:].max())
        if second_vol < 0.85 * first_vol:
            entry = float(d["close"].iloc[-1])
            stop = _offset_stop("SHORT", second_top, float(atr))
            neckline = float(window["low"].iloc[window.index.get_loc(first_top_idx):].min())
            target = neckline - (first_top - neckline)
            factors = ["double_top", "lower_volume_second_top"] + _factor_pack(d, "SHORT", include_nbb=False)
            return {
                "direction": "SHORT",
                "pattern": "DOUBLE_TOP",
                "entry": entry,
                "stop": stop,
                "target": target,
                "factors": factors,
            }

    first_bottom_idx = lows.iloc[:-5].idxmin()
    second_bottom = float(lows.iloc[-5:].min())
    first_bottom = float(lows.loc[first_bottom_idx])

    if abs(second_bottom - first_bottom) / max(first_bottom, 1e-9) <= DOUBLE_TOP_TOL:
        first_vol = float(vols.loc[first_bottom_idx])
        second_vol = float(vols.iloc[-5:].max())
        if second_vol < 0.85 * first_vol:
            entry = float(d["close"].iloc[-1])
            stop = _offset_stop("LONG", second_bottom, float(atr))
            neckline = float(window["high"].iloc[window.index.get_loc(first_bottom_idx):].max())
            target = neckline + (neckline - first_bottom)
            factors = ["double_bottom", "lower_volume_second_bottom"] + _factor_pack(d, "LONG", include_nbb=False)
            return {
                "direction": "LONG",
                "pattern": "DOUBLE_BOTTOM",
                "entry": entry,
                "stop": stop,
                "target": target,
                "factors": factors,
            }

    return None


def _detect_wide_range_reversal(d: pd.DataFrame) -> Optional[dict]:
    if len(d) < 25:
        return None

    r = d.iloc[-1]
    atr = r["atr14"]
    if pd.isna(atr) or atr <= 0:
        return None

    avg_range = d["range"].iloc[-20:].mean()
    avg_vol = d["volume"].iloc[-20:].mean()
    if pd.isna(avg_range) or pd.isna(avg_vol):
        return None

    is_wide = r["range"] >= 2.0 * avg_range
    is_heavy = r["volume"] >= 1.5 * avg_vol
    if not (is_wide and is_heavy):
        return None

    if _is_uptrend(d) and r["is_green"]:
        entry = float(r["close"])
        stop = _offset_stop("SHORT", float(r["high"]), float(atr))
        target = float(entry - 2 * atr)
        factors = ["wide_range_bar", "heavy_volume", "exhaustion_at_high"] + _factor_pack(d, "SHORT", include_nbb=False)
        return {
            "direction": "SHORT",
            "pattern": "WIDE_RANGE_EXHAUSTION_HIGH",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }

    if _is_downtrend(d) and r["is_red"]:
        entry = float(r["close"])
        stop = _offset_stop("LONG", float(r["low"]), float(atr))
        target = float(entry + 2 * atr)
        factors = ["wide_range_bar", "heavy_volume", "selling_exhaustion_low"] + _factor_pack(d, "LONG", include_nbb=False)
        return {
            "direction": "LONG",
            "pattern": "WIDE_RANGE_EXHAUSTION_LOW",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }

    return None


def _detect_max_extension(d: pd.DataFrame) -> Optional[dict]:
    """
    Rubber-band extension model (Rule 11).

    When a bar's distance from MA20 exceeds the maximum distance seen in the
    prior 50 bars, the stock is at an unprecedented extension — mean reversion
    back to the MA20 is the play.

    Long setup : price is below MA20 by more than prior max -> bounce to MA20
    Short setup: price is above MA20 by more than prior max -> fade to MA20
    """
    if len(d) < 55:
        return None

    r = d.iloc[-1]
    ma20 = r["ma20"]
    atr = r["atr14"]
    if pd.isna(ma20) or pd.isna(atr) or atr <= 0:
        return None

    current_dist = float(r["close"]) - float(ma20)  # positive = above MA20

    # Prior 50-bar max absolute distance (exclude current bar)
    prior = d.iloc[-51:-1]
    prior_dists = (prior["close"] - prior["ma20"]).abs()
    if prior_dists.isna().all():
        return None
    prior_max = float(prior_dists.max())

    if abs(current_dist) <= prior_max:
        return None  # Not exceeding historical extension — no setup

    # Must exceed by at least 0.5 ATR to be meaningful
    if abs(current_dist) < prior_max + 0.5 * atr:
        return None

    if current_dist > 0:
        # Above MA20 at max extension -> fade (SHORT) to MA20
        entry = float(r["close"])
        stop = _offset_stop("SHORT", float(r["high"]), float(atr))
        target = float(ma20)
        factors = ["max_extension_above_ma20", "rubber_band_reversal", "mean_reversion"] + _factor_pack(
            d, "SHORT", include_trend=False
        )
        return {
            "direction": "SHORT",
            "pattern": "MAX_EXTENSION_FADE",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }
    else:
        # Below MA20 at max extension -> bounce (LONG) to MA20
        entry = float(r["close"])
        stop = _offset_stop("LONG", float(r["low"]), float(atr))
        target = float(ma20)
        factors = ["max_extension_below_ma20", "rubber_band_reversal", "mean_reversion"] + _factor_pack(
            d, "LONG", include_trend=False
        )
        return {
            "direction": "LONG",
            "pattern": "MAX_EXTENSION_BOUNCE",
            "entry": entry,
            "stop": stop,
            "target": target,
            "factors": factors,
        }


# ── Scan ────────────────────────────────────────────────────────────────────

def scan(
    symbol: str,
    df: pd.DataFrame,
    min_confidence: int = MIN_FACTORS,
    strategy_type: str = "SWING",
    timeframe: str = "daily",
) -> Signal:
    """
    Scan a single symbol's OHLCV DataFrame for the best setup.

    Works on any timeframe — pass strategy_type="DAY_TRADE" and timeframe="5m"
    for intraday scans.  min_bars is relaxed automatically for intraday data.
    """
    min_bars = MIN_BARS_INTRADAY if strategy_type == "DAY_TRADE" else MIN_BARS
    if len(df) < min_bars:
        return _flat(symbol, strategy_type=strategy_type, timeframe=timeframe)

    d = _add_indicators(df)

    # Day-trading: skip multi-day patterns (double top/bottom, wedge)
    if strategy_type == "DAY_TRADE":
        detectors = [
            _detect_flag,
            _detect_inside_bar,
            _detect_reversal_bar_50,
            _detect_time_count,
            _detect_extended_move_reversal,
            _detect_max_extension,
            _detect_wide_range_reversal,
        ]
    else:
        detectors = [
            _detect_flag,
            _detect_inside_bar,
            _detect_wedge,
            _detect_reversal_bar_50,
            _detect_time_count,
            _detect_major_ma_bounce,
            _detect_extended_move_reversal,
            _detect_max_extension,
            _detect_double_top_bottom,
            _detect_wide_range_reversal,
        ]

    candidates: list[dict] = []
    for fn in detectors:
        try:
            result = fn(d)
        except Exception:
            result = None
        if result is None:
            continue

        factor_count = len(set(result["factors"]))
        if factor_count < min_confidence:
            continue
        result["confidence"] = min(factor_count, 5)
        candidates.append(result)

    if not candidates:
        return _flat(symbol, strategy_type=strategy_type, timeframe=timeframe)

    candidates.sort(
        key=lambda x: (
            -x["confidence"],
            abs(x["target"] - x["entry"]) / max(abs(x["entry"] - x["stop"]), 1e-9),
        )
    )
    best = candidates[0]

    return _build_signal(
        symbol=symbol,
        direction=best["direction"],
        pattern=best["pattern"],
        entry=best["entry"],
        raw_stop=best["stop"],
        target=best["target"],
        factors=best["factors"],
        notes=f"trend={_trend(d)} tech={_technical_position(d)} bow_tie={_bow_tie(d)}",
        strategy_type=strategy_type,
        timeframe=timeframe,
    )


if __name__ == "__main__":
    import sys
    from market_data import load_bars

    symbols = sys.argv[1:] or ["AAPL", "MSFT", "SPY", "QQQ", "NVDA"]

    print(f"\n{'=' * 70}")
    print(f"  R3 RULE SCAN — {len(symbols)} symbol(s)")
    print(f"{'=' * 70}\n")

    for sym in symbols:
        df = load_bars(sym, n_days=320)
        if df.empty:
            print(f"  {sym:<6} no data")
            continue
        print(" ", scan(sym, df))
    print()