"""
ThePatternHistory: one-page Streamlit app.

Run locally:
    pip install -r web/requirements.txt
    streamlit run web/streamlit_app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: rules.py, pattern_history/

import altair as alt
import pandas as pd
import streamlit as st

from pattern_history.catalog import RULE_DESC, RULE_NAME, RULES
from pattern_history.data import MARKETS, DataError, ProviderBusy, load_daily, resolve
from pattern_history.history import HORIZONS, directional, outlook, replay, rule_stats

SMALL_SAMPLE = 10
DISCLAIMER = (
    "Historical statistics for education only. Not investment advice. "
    "Past patterns do not guarantee future results."
)
HORIZON_LABEL = {1: "next day", 5: "1 week (5 trading days)", 20: "1 month (20 trading days)", 60: "3 months (60 trading days)"}

ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(page_title="ThePatternHistory", page_icon=str(ASSETS / "icon.png"), layout="wide")


def _b64(name: str) -> str:
    import base64

    return base64.b64encode((ASSETS / name).read_bytes()).decode()


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=200)
def get_history(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_daily(ticker, years=25)
    return prices, replay(prices)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=500)
def get_candidates(query: str, market: str):
    return resolve(query, market)


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x * 100:+.2f}%"


def share(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


def friendly(pattern: str) -> str:
    return pattern.replace("_", " ").title().replace("Ma", "MA")


def method_text(ticker: str, horizon: int, o: dict) -> str:
    if o["status"] == "unknown":
        return (
            f"None of our 10 rules is confirmed on {ticker}'s latest daily bar, so there is no pattern "
            "to look up in history and the outlook is Unknown. (A rule is 'confirmed' only when its "
            "pattern is present AND at least 2 supporting signs agree, e.g. trend direction, "
            "above-average volume, a quiet narrow candle, or price near a round number.)"
        )
    return (
        f"Step 1: today these rule(s) are confirmed on {ticker}: {', '.join(friendly(p) for p in o['patterns'])}. "
        f"Step 2: we went back through ~25 years of {ticker}'s daily prices and found every past trading day "
        f"when the same pattern(s) were confirmed, using only data available up to that day "
        f"({o['sample']} days; a day where two of them fired counts once). "
        f"Step 3: for each of those days we compared the closing price with the close {horizon} trading "
        f"day(s) later. Bullish % = share of days where the price was higher; Bearish % = share where it "
        "was lower. Today's signal is not counted because its outcome is not known yet. "
        "Past patterns do not guarantee future results."
    )


# ── Header and inputs ───────────────────────────────────────────────────────
st.markdown(
    f"""<picture>
      <source media="(prefers-color-scheme: dark)" srcset="data:image/png;base64,{_b64('logo-dark.png')}">
      <img src="data:image/png;base64,{_b64('logo.png')}" alt="ThePatternHistory"
           style="width:min(420px,85vw);height:auto;margin:0 0 0.5rem -8px;border-radius:12px">
    </picture>""",
    unsafe_allow_html=True,
)
st.caption("Pick a market, type a ticker or company name, and see what our pattern rules say today, "
           "backed by every time the same pattern appeared in the past.")

market = st.radio(
    "1. Market",
    list(MARKETS),
    format_func=lambda m: MARKETS[m]["label"],
    horizontal=True,
    key="market",
)

with st.form("scan"):
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    query = c1.text_input(
        "2. Ticker or company name",
        placeholder="e.g. MU or Micron" if market == "US" else "e.g. SBIN or State Bank of India",
    )
    submitted = c2.form_submit_button("Scan", type="primary", width="stretch")

if submitted:
    st.session_state["query"] = (query, market)

q, q_market = st.session_state.get("query", ("", market))
if not q or q_market != market:
    st.info("US tickers work as-is (AAPL, BRK-B). For India just type the NSE symbol (SBIN) or the company name — no suffix needed.")
    st.divider()
    st.caption(DISCLAIMER)
    st.stop()

with st.spinner("Looking up the company…"):
    candidates, best = get_candidates(q, market)
if not candidates:
    st.error(f"Couldn't find '{q}' on {MARKETS[market]['label']}. Try the ticker symbol, or switch market.")
    st.stop()

if len(candidates) > 1:
    choice = st.selectbox(
        "Matches",
        range(len(candidates)),
        index=best,
        format_func=lambda i: f"{candidates[i]['symbol']} — {candidates[i]['name']}",
        help="Not the company you meant? Pick another match.",
    )
else:
    choice = 0
ticker = candidates[choice]["symbol"]
company = candidates[choice]["name"]

try:
    with st.spinner(f"Loading 25 years of {ticker} and checking every rule on every day… "
                    "(first scan of a stock takes ~30–60s; repeat scans are instant)"):
        prices, hist = get_history(ticker)
except ProviderBusy as e:
    st.error(str(e))
    st.stop()
except DataError as e:
    st.error(str(e) if market == "US" else f"No price data found for {ticker} on NSE. Check the symbol or try the company name.")
    st.stop()
except Exception as e:
    st.error("Couldn't load data right now. Please try again in a minute.")
    st.caption(f"Details: {e}")
    st.stop()

last_date = prices.index[-1]
last_close = float(prices["close"].iloc[-1])
currency = "₹" if market == "India" else "$"
st.subheader(f"{company} ({ticker})")
st.caption(f"Last close {currency}{last_close:,.2f} on {last_date:%d %b %Y} · {len(prices):,} daily bars since "
           f"{prices.index[0]:%b %Y} (split/dividend-adjusted, Yahoo Finance)")

today = hist[hist["date"] == last_date]
firing = list(dict.fromkeys(today["pattern"]))

# ── Outlook ─────────────────────────────────────────────────────────────────
st.markdown("#### What history says")
horizon = st.segmented_control(
    "Looking ahead", HORIZONS, default=20, format_func=lambda h: HORIZON_LABEL[h].split(" (")[0],
    key="horizon",
) or 20
o = outlook(hist, last_date, horizon)
tip = method_text(ticker, horizon, o)

m1, m2, m3 = st.columns(3)
if o["status"] == "unknown":
    m1.metric("Bullish", "–", help=tip)
    m2.metric("Bearish", "–", help=tip)
    m3.metric("Unknown", "100%", help=tip)
    st.info("**Unknown: no pattern is confirmed today.** None of our 10 rules is confirmed on the latest bar, "
            "so there is no historical evidence to go on. The table below shows exactly what was checked.")
else:
    m1.metric("Bullish", share(o["bullish"]), help=tip)
    m2.metric("Bearish", share(o["bearish"]), help=tip)
    m3.metric("Past cases", f"{o['sample']}", help=tip)
    st.caption(f"Over the {HORIZON_LABEL[horizon]} after the same pattern(s) appeared in {ticker}'s past. "
               "Hover over (or tap) the ? icons for how this is calculated.")
    if o["sample"] < SMALL_SAMPLE:
        st.warning(f"⚠️ Small sample: only {o['sample']} past case(s). Treat these percentages as anecdotal.")
with st.expander("How is this calculated?"):
    st.write(tip)

# ── What the bot checked ────────────────────────────────────────────────────
st.markdown("#### What our bot checked today")
rows = []
for key, name, desc in RULES:
    fired_today = today[today["rule"] == key]
    past_rule = hist[(hist["rule"] == key) & (hist["date"] < last_date)]
    if fired_today.empty:
        status = "— Not confirmed"
    else:
        dirs = sorted(set(fired_today["direction"]))
        status = " / ".join("✅ Bullish" if d == "LONG" else "✅ Bearish" for d in dirs)
        status += " — " + ", ".join(friendly(p) for p in fired_today["pattern"])
    rows.append({"Rule": name, "Today": status, "Times seen in 25 yrs": len(past_rule), "What it looks for": desc})
checked = pd.DataFrame(rows)
st.dataframe(
    checked, hide_index=True, width="stretch",
    column_config={"What it looks for": st.column_config.TextColumn(width="large")},
)
st.caption(f"{len(firing)} of 10 rules confirmed on the latest bar. A rule counts only when its pattern is present "
           "and at least 2 supporting signs agree.")

# ── Excel download ──────────────────────────────────────────────────────────
evidence = hist[hist["pattern"].isin(firing)] if firing else hist
evidence = evidence.copy()
evidence.insert(0, "ticker", ticker)
evidence["rule"] = evidence["rule"].map(RULE_NAME)
evidence["date"] = evidence["date"].dt.strftime("%Y-%m-%d")
summary_rows = []
for h in HORIZONS:
    oh = outlook(hist, last_date, h)
    summary_rows.append({
        "Horizon": HORIZON_LABEL[h],
        "Outlook": "Unknown (no pattern confirmed)" if oh["status"] == "unknown" else "From past cases",
        "Bullish %": None if pd.isna(oh["bullish"]) else round(oh["bullish"] * 100, 1),
        "Bearish %": None if pd.isna(oh["bearish"]) else round(oh["bearish"] * 100, 1),
        "Past cases": oh["sample"],
        "How it is calculated": method_text(ticker, h, oh),
    })
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    pd.DataFrame({
        "Field": ["Company", "Ticker", "Market", "Latest bar", "Rules confirmed today", "Disclaimer"],
        "Value": [company, ticker, MARKETS[market]["label"], f"{last_date:%Y-%m-%d}",
                  ", ".join(friendly(p) for p in firing) or "None (Unknown)", DISCLAIMER],
    }).to_excel(xw, sheet_name="Summary", index=False)
    pd.DataFrame(summary_rows).to_excel(xw, sheet_name="Outlook", index=False)
    checked.to_excel(xw, sheet_name="Rules checked", index=False)
    evidence.to_excel(xw, sheet_name="Past cases", index=False)
st.download_button(
    "Download evidence (Excel)",
    buf.getvalue(),
    file_name=f"{ticker}_pattern_evidence_{last_date:%Y%m%d}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Includes the outlook for every horizon, how it is calculated, what was checked, and every past case.",
)

# ── Detail per confirmed rule ───────────────────────────────────────────────
for pattern in firing:
    occ_all = hist[hist["pattern"] == pattern].sort_values("date")
    direction = occ_all["direction"].iloc[0]
    rule_key = occ_all["rule"].iloc[0]
    past = occ_all[occ_all["date"] < last_date]
    arrow = "📈 bullish" if direction == "LONG" else "📉 bearish"

    st.divider()
    st.markdown(f"### {RULE_NAME[rule_key]}: {friendly(pattern)} ({arrow})")
    st.caption(RULE_DESC[rule_key])
    st.caption("Supporting signs today: " + today[today["pattern"] == pattern]["factors"].iloc[0])

    n_past = len(past)
    if n_past < SMALL_SAMPLE:
        st.warning(f"⚠️ Small sample: this pattern was confirmed only {n_past} time(s) before in {ticker}'s history.")
    if n_past == 0:
        continue

    stats = rule_stats(past, direction)
    st.dataframe(pd.DataFrame({
        "After": [HORIZON_LABEL[s.horizon].split(" (")[0] for s in stats],
        "Cases": [s.sample for s in stats],
        "Went the rule's way": [share(s.win_rate) for s in stats],
        "Average": [pct(s.average) for s in stats],
        "Median": [pct(s.median) for s in stats],
        "Worst": [pct(s.worst) for s in stats],
        "Best": [pct(s.best) for s in stats],
    }), hide_index=True, width="stretch")
    st.caption("Returns in this table are measured in the rule's direction (for a bearish rule, a price drop counts as a gain).")

    chart_px = prices[["close"]].reset_index().rename(columns={"ts": "date"})
    line = alt.Chart(chart_px).mark_line(strokeWidth=1, color="#7a8699").encode(
        x=alt.X("date:T", title=None), y=alt.Y("close:Q", title="Close (adjusted)", scale=alt.Scale(zero=False))
    )
    marks = occ_all[["date", "close", "ret_20d"]].copy()
    marks["Outcome (1 month)"] = [
        "Pending" if pd.isna(r) else ("Went rule's way" if r > 0 else "Went against")
        for r in directional(marks["ret_20d"], direction)
    ]
    marks["1-month move"] = [pct(r) for r in marks["ret_20d"]]
    points = alt.Chart(marks).mark_point(filled=True, size=45).encode(
        x="date:T", y="close:Q",
        color=alt.Color("Outcome (1 month):N", scale=alt.Scale(
            domain=["Went rule's way", "Went against", "Pending"], range=["#1a9850", "#d73027", "#4575b4"])),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("close:Q", format=",.2f"), "1-month move", "Outcome (1 month)"],
    )
    st.altair_chart((line + points).properties(height=240), width="stretch")

    table = past.sort_values("date", ascending=False)
    res = directional(table["ret_20d"], direction)
    st.markdown(f"**Every past case ({n_past})**: {(res <= 0).sum()} went against the rule after 1 month. "
                "Moves shown are plain price change.")
    st.dataframe(pd.DataFrame({
        "Date": table["date"].dt.strftime("%Y-%m-%d"),
        "Price": table["close"].map(lambda v: f"{v:,.2f}"),
        **{HORIZON_LABEL[h].split(" (")[0]: table[f"ret_{h}d"].map(pct) for h in HORIZONS},
        "After 1 month": ["Pending" if pd.isna(r) else ("✅ Rule's way" if r > 0 else "❌ Against") for r in res],
    }), hide_index=True, width="stretch", height=min(38 + 35 * n_past, 380))

st.divider()
st.caption(f"**{DISCLAIMER}**")
