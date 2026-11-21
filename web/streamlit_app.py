"""
ThePatternHistory: one-page Streamlit app.

Run locally:
    pip install -r web/requirements.txt
    streamlit run web/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: rules.py, pattern_history/

import altair as alt
import pandas as pd
import streamlit as st

from pattern_history.data import DataError, load_daily, normalize_ticker
from pattern_history.history import HORIZONS, directional, replay, rule_stats

SMALL_SAMPLE = 10
DISCLAIMER = (
    "Historical statistics for education only. Not investment advice. "
    "Past patterns do not guarantee future results."
)

st.set_page_config(page_title="ThePatternHistory", page_icon="📈", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=200)
def get_history(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_daily(ticker, years=25)
    return prices, replay(prices)


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x * 100:+.2f}%"


def friendly(pattern: str) -> str:
    return pattern.replace("_", " ").title().replace("Ma", "MA")


# ── Header ──────────────────────────────────────────────────────────────────
st.title("ThePatternHistory")
st.caption(
    "Type a ticker to see which of our price-pattern rules fire on the latest daily bar, "
    "and what happened every past time the same rule fired."
)

with st.form("scan"):
    c1, c2 = st.columns([4, 1])
    raw = c1.text_input(
        "Ticker",
        value=st.session_state.get("ticker", ""),
        placeholder="e.g. AAPL, AKAM, or RELIANCE.NS for India (NSE)",
        label_visibility="collapsed",
    )
    submitted = c2.form_submit_button("Scan", type="primary", width="stretch")

if submitted:
    st.session_state["ticker"] = raw

ticker_raw = st.session_state.get("ticker", "")
if not ticker_raw:
    st.info("US tickers work as-is (AAPL, MSFT, BRK-B). Indian NSE stocks need the .NS suffix (TCS.NS).")
    st.divider()
    st.caption(DISCLAIMER)
    st.stop()

try:
    ticker = normalize_ticker(ticker_raw)
    with st.spinner(f"Loading 25 years of {ticker} and replaying every rule on every day… (first scan of a ticker can take ~30–60s)"):
        prices, hist = get_history(ticker)
except DataError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # network / provider hiccups
    st.error(f"Couldn't load data for {ticker_raw.strip().upper()} right now. Please try again in a minute.")
    st.caption(f"Details: {e}")
    st.stop()

last_date = prices.index[-1]
last_close = float(prices["close"].iloc[-1])
st.subheader(f"{ticker} · last close {last_close:,.2f} on {last_date:%d %b %Y}")
st.caption(
    f"{len(prices):,} daily bars from {prices.index[0]:%b %Y} (split/dividend-adjusted, Yahoo Finance). "
    "Rules need ~220 bars of history before they can fire."
)

today = hist[hist["date"] == last_date]
firing = list(dict.fromkeys(today["pattern"]))

# ── Evidence export (all occurrences of rules firing today) ─────────────────
if firing:
    st.success(f"{len(firing)} rule(s) firing on the latest bar: " + ", ".join(friendly(p) for p in firing))
else:
    st.warning("No rules are firing on the latest bar for this ticker.")

show = firing
with st.expander("Explore the history of any rule for this ticker"):
    all_patterns = sorted(hist["pattern"].unique())
    extra = st.multiselect("Rules", [p for p in all_patterns if p not in firing], format_func=friendly)
    show = firing + extra

if show:
    evidence = hist[hist["pattern"].isin(show)].copy()
    evidence.insert(0, "ticker", ticker)
    for h in HORIZONS:
        evidence[f"ret_{h}d_in_rule_direction"] = [
            -r if d == "SHORT" else r for r, d in zip(evidence[f"ret_{h}d"], evidence["direction"])
        ]
    evidence["date"] = evidence["date"].dt.strftime("%Y-%m-%d")
    st.download_button(
        "Download evidence (CSV)",
        evidence.to_csv(index=False).encode(),
        file_name=f"{ticker}_pattern_evidence_{last_date:%Y%m%d}.csv",
        mime="text/csv",
    )

# ── One section per rule ────────────────────────────────────────────────────
for pattern in show:
    occ_all = hist[hist["pattern"] == pattern].sort_values("date")
    direction = occ_all["direction"].iloc[0]
    past = occ_all[occ_all["date"] < last_date]  # exclude today's (outcome unknown)
    arrow = "📈 bullish (LONG)" if direction == "LONG" else "📉 bearish (SHORT)"

    st.divider()
    tag = " — firing today" if pattern in firing else ""
    st.markdown(f"### {friendly(pattern)}{tag}")
    st.caption(f"Direction: {arrow} · `{pattern}`")
    if pattern in firing:
        st.caption("Today's factors: " + today[today["pattern"] == pattern]["factors"].iloc[0])

    n_past = len(past)
    if n_past < SMALL_SAMPLE:
        st.warning(
            f"⚠️ Small sample: this rule fired only {n_past} time(s) before in {ticker}'s history. "
            "Treat the statistics below as anecdotal."
        )
    if n_past == 0:
        continue

    # Stats per horizon
    stats = rule_stats(past, direction)
    stat_df = pd.DataFrame(
        {
            "After": [f"{s.horizon} day{'s' if s.horizon > 1 else ''}" for s in stats],
            "Sample size": [s.sample for s in stats],
            "Win rate": ["–" if pd.isna(s.win_rate) else f"{s.win_rate:.0%}" for s in stats],
            "Average": [pct(s.average) for s in stats],
            "Median": [pct(s.median) for s in stats],
            "Worst": [pct(s.worst) for s in stats],
            "Best": [pct(s.best) for s in stats],
        }
    )
    st.markdown("**Outcomes, measured in the rule's direction** (a win = price moved the way the rule expects)")
    st.dataframe(stat_df, hide_index=True, width="stretch")

    # Chart with occurrence markers
    chart_px = prices[["close"]].reset_index().rename(columns={"ts": "date"})
    line = alt.Chart(chart_px).mark_line(strokeWidth=1, color="#7a8699").encode(
        x=alt.X("date:T", title=None), y=alt.Y("close:Q", title="Close (adjusted)", scale=alt.Scale(zero=False))
    )
    marks = occ_all[["date", "close", "ret_20d"]].copy()
    dir20 = directional(marks["ret_20d"], direction)
    marks["Outcome (20d)"] = ["Pending" if pd.isna(r) else ("Win" if r > 0 else "Loss") for r in dir20]
    marks["20d return"] = [pct(r) for r in marks["ret_20d"]]
    points = alt.Chart(marks).mark_point(filled=True, size=45).encode(
        x="date:T",
        y="close:Q",
        color=alt.Color(
            "Outcome (20d):N",
            scale=alt.Scale(domain=["Win", "Loss", "Pending"], range=["#1a9850", "#d73027", "#4575b4"]),
        ),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("close:Q", format=",.2f"), "20d return", "Outcome (20d)"],
    )
    st.altair_chart((line + points).properties(height=260), width="stretch")

    # Occurrence table: losers included, worst first toggle
    table = past[["date", "close"] + [f"ret_{h}d" for h in HORIZONS]].copy()
    table["Result (20d)"] = [
        "Pending" if pd.isna(r) else ("✅ Win" if r > 0 else "❌ Loss")
        for r in directional(table["ret_20d"], direction)
    ]
    table = table.sort_values("date", ascending=False)
    view = pd.DataFrame(
        {
            "Date": table["date"].dt.strftime("%Y-%m-%d"),
            "Price": table["close"].map(lambda v: f"{v:,.2f}"),
            **{f"{h}d": table[f"ret_{h}d"].map(pct) for h in HORIZONS},
            "Result (20d)": table["Result (20d)"],
        }
    )
    losses = (table["Result (20d)"] == "❌ Loss").sum()
    st.markdown(
        f"**Every past occurrence ({n_past})** — {losses} lost after 20 days. "
        "Returns shown are plain price change."
    )
    st.dataframe(view, hide_index=True, width="stretch", height=min(38 + 35 * len(view), 400))

st.divider()
st.caption(
    "How it works: each rule is checked on the close of every trading day using only data available "
    "up to that day. Returns are close-to-close after 1, 5, 20 and 60 trading days. "
    "Overlapping signals on consecutive days are each counted."
)
st.caption(f"**{DISCLAIMER}**")
