"""Plain-English names and descriptions for each rule (detector) in rules.py."""

RULES = [
    ("_detect_inside_bar", "Inside bar",
     "Today's whole price range fits inside yesterday's, during a trend: a pause before the trend continues."),
    ("_detect_reversal_bar_50", "50% reversal bar",
     "Today wipes out more than half of yesterday's counter-move, in the direction of the trend."),
    ("_detect_time_count", "Time count",
     "A quiet candle appears 3, 6, 7, 8, 9, 10, 14 or 21 days after the last swing high/low: a timing-based turn."),
    ("_detect_flag", "Flag",
     "A sharp move, a short tight pause, then a breakout from the pause in the same direction."),
    ("_detect_wedge", "Wedge pullback",
     "In a trend, price pulls back 35–75% of the recent swing: bets the trend continues."),
    ("_detect_major_ma_bounce", "Moving-average bounce",
     "Price touches its 20/50/100/200-day average and holds with the trend ('major' = first touch after being far away)."),
    ("_detect_extended_move_reversal", "Overstretched snap-back",
     "Price is very far from its 20-day average and prints a quiet candle: bets on a snap back."),
    ("_detect_max_extension", "Record stretch",
     "Price is further from its 20-day average than at any time in the last 50 days: bets on a return to the average."),
    ("_detect_double_top_bottom", "Double top / bottom",
     "A second peak (or low) near the first one, on lighter volume: bets on a reversal."),
    ("_detect_wide_range_reversal", "Climax bar",
     "A huge candle on heavy volume late in a trend: bets the move is exhausted."),
]

RULE_NAME = {k: name for k, name, _ in RULES}
RULE_DESC = {k: desc for k, _, desc in RULES}
