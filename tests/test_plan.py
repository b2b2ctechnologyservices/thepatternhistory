import rules
from pattern_history.plan import todays_plans
from tests.test_v3 import _synthetic


def test_top_pick_matches_original_scan_on_many_days():
    df = _synthetic(n=1200, seed=3)
    checked = 0
    for end in range(rules.MIN_BARS + 5, len(df), 7):
        part = df.iloc[: end + 1]
        sig = rules.scan("X", part)
        _, plans = todays_plans("X", part)
        if sig.direction == "FLAT":
            assert plans == []
            continue
        top = plans[0]
        assert (top["pattern"], top["entry"], top["stop"], top["target"], top["r"], top["confidence"]) == (
            sig.pattern, sig.entry, sig.stop, sig.target, sig.r_multiple, sig.confidence)
        assert top["decision"] == ("TAKE" if sig.r_multiple >= 1.5 else "SKIP")
        assert all(p["decision"] == "NOT CONSIDERED" for p in plans[1:])
        checked += 1
    assert checked > 10
