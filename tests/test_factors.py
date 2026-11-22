from pattern_history.factors import label, split


def test_akam_example():
    own, ind = split("MA20_MINOR_REJECTION",
                     "ma20_minor, minor_ma_touch, round_number_85, trend_down, weak_technical_position")
    assert own == ["20-day average (minor)", "Minor moving-average touch"]
    assert ind == ["Near round number 85", "Downtrend", "Weak position vs moving averages"]


def test_narrow_body_is_pattern_for_time_count():
    own, ind = split("TIME_COUNT_LONG_9", ["time_count_9", "narrow_body_bar", "trend_up"])
    assert "Narrow-body bar" in own and ind == ["Uptrend"]


def test_nbb_independent_elsewhere_and_dedup():
    own, ind = split("BULL_FLAG", ["bull_flag", "consol_5_bars", "flagpole", "narrow_body_bar", "narrow_body_bar"])
    assert ind == ["Narrow-body bar"] and own == ["Bull flag", "5-bar consolidation", "Flagpole"]
    assert label("round_number_2.5") == "Near round number 2.5"
