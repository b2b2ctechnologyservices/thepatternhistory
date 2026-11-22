"""
Plain-English names for the factor codes the original detectors attach to a
signal, and a split between the pattern's own labels and independent signs.

The bot's confidence is the number of distinct codes, capped at 5
(rules._build_signal). Each detector starts its list with labels that just
describe the pattern itself (e.g. "ma20_minor", "minor_ma_touch"), then adds
the shared factor pack (rules._factor_pack): trend, technical position,
narrow body, round number, volume, bow tie. Only the pack items are
independent evidence. Display only; rules.py is unchanged.
"""

from __future__ import annotations

import re

_FIXED = {
    # shared factor pack (independent)
    "trend_up": "Uptrend", "strong_trend_up": "Strong uptrend",
    "trend_down": "Downtrend", "strong_trend_down": "Strong downtrend",
    "strong_technical_position": "Strong position vs moving averages",
    "weak_technical_position": "Weak position vs moving averages",
    "narrow_body_bar": "Narrow-body bar",
    "above_average_volume": "Above-average volume",
    "bow_tie_bull": "Bullish bow tie", "bow_tie_bear": "Bearish bow tie",
    # pattern self-labels
    "inside_bar": "Inside bar", "prior_red_bar": "Prior bar was red",
    "prior_green_bar": "Prior bar was green", "close_based_invalidation": "Stop is on a close",
    "reversal_bar_through_50pct": "Reversal bar through 50% of prior bar",
    "bearish_reversal": "Bearish reversal", "bullish_reversal": "Bullish reversal",
    "bear_flag": "Bear flag", "bull_flag": "Bull flag", "flagpole": "Flagpole",
    "wedge": "Wedge", "multiple_bars_down": "Several bars down", "multiple_bars_up": "Several bars up",
    "symmetry_setup": "Symmetry setup",
    "first_touch_from_extended": "First touch after an extended move",
    "minor_ma_touch": "Minor moving-average touch",
    "extended_above_ma20": "Stretched above the 20-day average",
    "extended_below_ma20": "Stretched below the 20-day average",
    "mean_reversion": "Mean reversion",
    "double_top": "Double top", "double_bottom": "Double bottom",
    "lower_volume_second_top": "Lower volume on 2nd top",
    "lower_volume_second_bottom": "Lower volume on 2nd bottom",
    "wide_range_bar": "Wide-range bar", "heavy_volume": "Heavy volume",
    "exhaustion_at_high": "Exhaustion at the high", "selling_exhaustion_low": "Selling exhaustion at the low",
    "max_extension_above_ma20": "Max stretch above the 20-day average",
    "max_extension_below_ma20": "Max stretch below the 20-day average",
    "rubber_band_reversal": "Rubber-band reversal",
}

_PACK_FIXED = {"trend_up", "strong_trend_up", "trend_down", "strong_trend_down",
               "strong_technical_position", "weak_technical_position",
               "narrow_body_bar", "above_average_volume", "bow_tie_bull", "bow_tie_bear"}

# Detectors where a narrow-body bar is part of the pattern's own definition.
_NBB_IS_PATTERN = ("TIME_COUNT", "EXTENDED_MOVE")


def label(code: str) -> str:
    if code in _FIXED:
        return _FIXED[code]
    if m := re.fullmatch(r"round_number_(.+)", code):
        return f"Near round number {m.group(1)}"
    if m := re.fullmatch(r"ma(\d+)_(major|minor)", code):
        return f"{m.group(1)}-day average ({m.group(2)})"
    if m := re.fullmatch(r"time_count_(\d+)", code):
        return f"Time count {m.group(1)}"
    if m := re.fullmatch(r"consol_(\d+)_bars", code):
        return f"{m.group(1)}-bar consolidation"
    return code.replace("_", " ").capitalize()


def is_independent(code: str, pattern: str) -> bool:
    if code == "narrow_body_bar" and pattern.startswith(_NBB_IS_PATTERN):
        return False
    return code in _PACK_FIXED or code.startswith("round_number_")


def split(pattern: str, factors) -> tuple[list[str], list[str]]:
    """(pattern's own labels, independent signs) as plain names, de-duplicated, in order."""
    if isinstance(factors, str):
        factors = [f.strip() for f in factors.split(",") if f.strip()]
    own, ind, seen = [], [], set()
    for f in factors:
        if f in seen:
            continue
        seen.add(f)
        (ind if is_independent(f, pattern) else own).append(label(f))
    return own, ind
