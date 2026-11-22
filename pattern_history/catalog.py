"""Plain-English names and descriptions for each rule: the detectors in rules.py, the rule-document
patterns in doc_rules.py, and the yearly checks."""

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

# Patterns from the rule document (pattern_history/doc_rules.py).
DOC_RULES = [
    ("doc_major_ma", "Major average test",
     "After a big move away, the price comes back and touches its 50, 100 or 200-bar average for the first "
     "time, and holds there: a level that often stops a move."),
    ("doc_first_pullback_20", "First pullback to the 20",
     "In a clear trend, the price had pulled well away from its 20-bar average and now comes back to touch it "
     "for the first time (called 'drop and pop' going up, 'tag and tank' going down)."),
    ("doc_bow_tie", "Bow tie",
     "The 20 and 50-bar averages are meeting, and the price reaches that spot coming from a big move: "
     "a level where the move can stall."),
]
RULES = RULES + DOC_RULES

# Yearly checks (pattern_history/yearly.py), keyed by pattern code.
YEARLY_CHECKS = [
    ("YEAR_INSIDE", "Inside last year's range",
     "The year's highest and lowest prices both stayed inside last year's range: a quiet year."),
    ("YEAR_ABOVE_HIGH", "Above last year's high",
     "The year closed above the highest price of the year before."),
    ("YEAR_BELOW_LOW", "Below last year's low",
     "The year closed below the lowest price of the year before."),
    ("YEAR_UP", "Up year", "The year closed above the price it opened the year at."),
    ("YEAR_DOWN", "Down year", "The year closed below the price it opened the year at."),
]
YEARLY_NAME = {k: n for k, n, _ in YEARLY_CHECKS}
YEARLY_DESC = {k: d for k, _, d in YEARLY_CHECKS}

RULE_NAME = {k: name for k, name, _ in RULES}
RULE_DESC = {k: desc for k, _, desc in RULES}
