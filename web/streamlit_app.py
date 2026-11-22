"""
ThePatternHistory: one-page Streamlit app.

Run locally:
    pip install -r web/requirements.txt
    streamlit run web/streamlit_app.py
"""

from __future__ import annotations

import html
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: rules.py, pattern_history/

import altair as alt
import pandas as pd
import streamlit as st

from pattern_history.catalog import RULE_DESC, RULE_NAME, RULES
from pattern_history.data import MARKETS, DataError, ProviderBusy, load_daily, resolve
from pattern_history import v3
from pattern_history.factors import split as split_factors
from pattern_history.plan import MIN_R, todays_plans
from pattern_history.history import HORIZONS, directional, outlook, replay, rule_stats

SMALL_SAMPLE = 10
DISCLAIMER = (
    "Historical statistics for education only. Not investment advice, and not a recommendation to buy or "
    "sell any security. Hypothetical trades show how a fixed set of rules would behave; nobody is told to "
    "place them. Past patterns do not guarantee future results."
)
HORIZON_LABEL = {1: "next day", 5: "1 week (5 trading days)", 20: "1 month (20 trading days)", 60: "3 months (60 trading days)"}
DECISION_TEXT = {"TAKE": "would have taken it", "SKIP": "would have skipped it",
                 "NOT CONSIDERED": "would not have looked at it"}

ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(page_title="ThePatternHistory", page_icon=str(ASSETS / "icon.png"), layout="wide")


def _b64(name: str) -> str:
    import base64

    return base64.b64encode((ASSETS / name).read_bytes()).decode()


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=200)
def get_history(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_daily(ticker, years=25)
    return prices, v3.annotate(prices, replay(prices))


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=200)
def get_plans(ticker: str, _prices: pd.DataFrame, last_date) -> tuple[dict, list[dict]]:
    return todays_plans(ticker, _prices)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=500)
def get_candidates(query: str, market: str):
    return resolve(query, market)


def pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x * 100:+.2f}%"


def share(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


def friendly(pattern: str) -> str:
    return pattern.replace("_", " ").title().replace("Ma", "MA")


def esc(s) -> str:
    return html.escape(str(s))


RULESET_NOTE = {
    "Original": "Rule set: Original (the bot's rules exactly as coded: a pattern counts when at least 2 of its "
                "signs agree).",
    "Version 3": "Rule set: Version 3 (a pattern counts only if it is at a location — a major moving average, a bow "
                 "tie, or a flag/wedge zone — AND has at least one supporting factor — round number, narrow body, "
                 "inside bar or wide-range bar — AND the weekly trend does not point the other way).",
}


def method_text(ticker: str, horizon: int, o: dict, ruleset: str = "Original") -> str:
    note = " " + RULESET_NOTE[ruleset]
    if o["status"] == "unknown":
        return (
            f"None of our 10 rules is confirmed on {ticker}'s latest daily bar under this rule set, so there is no pattern "
            "to look up in history and the outlook is Unknown." + note
        )
    return note.strip() + " " + (
        f"Step 1: today these rule(s) are confirmed on {ticker}: {', '.join(friendly(p) for p in o['patterns'])}. "
        f"Step 2: we went back through ~25 years of {ticker}'s daily prices and found every past trading day "
        f"when the same pattern(s) were confirmed, using only data available up to that day "
        f"({o['sample']} days; a day where two of them fired counts once). "
        f"Step 3: for each of those days we compared the closing price with the close {horizon} trading "
        f"day(s) later. Bullish % = share of days where the price was higher; Bearish % = share where it "
        "was lower. Today's signal is not counted because its outcome is not known yet. "
        "Past patterns do not guarantee future results."
    )


# ── Look and feel ───────────────────────────────────────────────────────────
st.markdown("""<style>
:root{--slate:#24414A;--teal:#2F7F86;--sea:#6FB3B0;--mist:#EEF5F2;--line:#D7E5E1;--muted:#5E7479;
      --green:#2E8B62;--red:#B8574B;--amber:#B7832F}
.block-container{padding-top:1.2rem!important;padding-bottom:1rem!important;max-width:1120px}
header[data-testid="stHeader"]{display:none}
.tph-logo img{height:44px;width:auto;display:block;margin-top:2px}
.st-key-topbar{border-bottom:1px solid var(--line);padding-bottom:.6rem;margin-bottom:.4rem}
.st-key-answer{border-top:4px solid var(--teal)!important;background:#fff}
.tph-strip{margin-bottom:6px;display:flex;align-items:center;gap:12px;padding:12px 16px;border-radius:12px;font-size:15px;line-height:1.35}
.tph-strip.done{background:#E4F2EC;border:1px solid #BFDCCF;color:var(--slate)}
.tph-strip.busy{background:#fff;border:1px solid var(--line)}
.tph-tick{width:26px;height:26px;border-radius:50%;background:var(--green);color:#fff;display:flex;
          align-items:center;justify-content:center;font-weight:700;flex:none}
.tph-spin{width:24px;height:24px;border-radius:50%;border:3px solid var(--mist);border-top-color:var(--teal);
          animation:tphspin .8s linear infinite;flex:none}
@keyframes tphspin{to{transform:rotate(360deg)}}
.tph-bar{height:6px;background:var(--mist);border-radius:3px;overflow:hidden;margin-top:6px;position:relative}
.tph-bar i{position:absolute;height:100%;width:35%;background:var(--teal);border-radius:3px;animation:tphbar 1.4s ease-in-out infinite}
@keyframes tphbar{0%{left:-35%}100%{left:100%}}
.tph-co h2{font-size:1.7rem;font-weight:700;margin:0;padding:0;line-height:1.2;color:var(--slate)!important}
.tph-co h2 span{color:var(--muted);font-weight:500}
.tph-meta{color:var(--muted);font-size:.85rem;margin-top:2px}
.tph-chips{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 4px}
.tph-chip{font-size:.8rem;padding:4px 10px;border-radius:20px;background:var(--mist);color:var(--slate);font-weight:500}
.tph-chip.down{background:#F6E3E0;color:var(--red)} .tph-chip.up{background:#E4F2EC;color:var(--green)}
.tph-verdict{display:flex;gap:12px;align-items:center;border-radius:10px;padding:12px;margin:10px 0 6px;font-size:.9rem}
.tph-verdict.SKIP{background:#FBF5EA;color:#6B5A3A} .tph-verdict.TAKE{background:#E4F2EC;color:#24513F}
.tph-verdict.NONE{background:var(--mist);color:var(--slate)}
.tph-badge{font-weight:800;font-size:.8rem;letter-spacing:.06em;padding:6px 11px;border-radius:8px;color:#fff;flex:none}
.tph-badge.SKIP{background:var(--amber)} .tph-badge.TAKE{background:var(--green)} .tph-badge.NONE{background:var(--muted)}
.tph-label{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:4px}
.tph-split{height:10px;border-radius:5px;display:flex;overflow:hidden;margin:4px 0 6px;background:#ddd}
.tph-tag{font-size:.75rem;font-weight:600;padding:3px 9px;border-radius:6px;white-space:nowrap}
.tph-tag.SHORT{background:#F6E3E0;color:var(--red)} .tph-tag.LONG{background:#E4F2EC;color:var(--green)}
.tph-hyp{font-size:.8rem;color:var(--muted);background:var(--mist);border-radius:8px;padding:6px 10px;margin:2px 0 8px}
.tph-foot{color:var(--muted);font-size:.8rem;text-align:center;padding:18px 8px 4px;border-top:1px solid var(--line);margin-top:24px}
.tph-hero h1{font-size:1.6rem;margin:.6rem 0 .2rem;padding:0;color:var(--slate)}
[data-testid="stMetricValue"]{font-weight:700}
.st-key-bull [data-testid="stMetricValue"]{color:var(--green)} .st-key-bear [data-testid="stMetricValue"]{color:var(--red)}
@media (max-width:640px){.tph-co h2{font-size:1.35rem}.tph-logo img{height:36px}
  .st-key-stats [data-testid="stHorizontalBlock"],.st-key-plan [data-testid="stHorizontalBlock"]{flex-wrap:nowrap;gap:.6rem}
  .st-key-stats [data-testid="stColumn"],.st-key-plan [data-testid="stColumn"]{min-width:0!important;width:auto!important;flex:1 1 0!important}
  [data-testid="stMetricValue"]{font-size:1.55rem}}
</style>""", unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="tph-foot">{esc(DISCLAIMER)}</div>', unsafe_allow_html=True)


# ── Top bar: logo, market, search (one step) ────────────────────────────────
with st.container(key="topbar"):
    c_logo, c_mkt, c_search = st.columns([1.15, 0.9, 1.9], vertical_alignment="center")
    c_logo.markdown(f'<div class="tph-logo"><img src="data:image/png;base64,{_b64("logo.png")}" '
                    'alt="ThePatternHistory"></div>', unsafe_allow_html=True)
    market = c_mkt.segmented_control(
        "Market", list(MARKETS), default="US", format_func=lambda m: MARKETS[m]["label"],
        key="market", label_visibility="collapsed",
    ) or "US"
    with c_search.form("scan", border=False):
        f1, f2 = st.columns([4, 1.2], vertical_alignment="bottom", gap="small")
        query = f1.text_input(
            "Ticker or company name", label_visibility="collapsed",
            placeholder="🔍  Ticker or company, e.g. MU or Micron" if market == "US"
            else "🔍  NSE symbol or company, e.g. SBIN",
        )
        submitted = f2.form_submit_button("Scan", type="primary", width="stretch")

if submitted and query.strip():
    st.session_state["query"] = (query.strip(), market)

q, q_market = st.session_state.get("query", ("", market))
if not q or q_market != market:
    st.markdown(
        '<div class="tph-hero"><h1>What happened every time this pattern showed up before?</h1></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Pick **US** or **India**, type a ticker or company name and press **Scan**. That's it: we check 10 "
        "chart-pattern rules on today's bar, then look up every past day the same pattern appeared in "
        "~25 years of prices and show what happened next."
    )
    st.caption("US tickers work as-is (AAPL, BRK-B). For India type the NSE symbol (SBIN) or the company name — "
               "no suffix needed.")
    footer()
    st.stop()

# ── Scan: resolve the name, load data, show a clear busy / done state ──────
strip = st.empty()


def busy(text: str, sub: str) -> None:
    strip.markdown(
        f'<div class="tph-strip busy"><span class="tph-spin"></span><div style="flex:1"><b>{esc(text)}</b>'
        f'<div style="color:var(--muted);font-size:.85rem">{esc(sub)}</div>'
        '<div class="tph-bar"><i></i></div></div></div>', unsafe_allow_html=True)


busy(f"Finding '{q}'…", f"Looking it up on {MARKETS[market]['label']}")
candidates, best = get_candidates(q, market)
if not candidates:
    strip.error(f"Couldn't find '{q}' on {MARKETS[market]['label']}. Try the ticker symbol, or switch market.")
    footer()
    st.stop()

pick_key = f"pick::{market}::{q}"
choice = st.session_state.get(pick_key, best)
choice = choice if 0 <= choice < len(candidates) else best
ticker = candidates[choice]["symbol"]
company = candidates[choice]["name"]

busy(f"Scanning {company} ({ticker})…",
     "Loading ~25 years of daily prices and checking every rule on every day. "
     "The first scan of a stock takes ~30–60s; repeat scans are instant.")
try:
    prices, hist = get_history(ticker)
except ProviderBusy as e:
    strip.error(str(e))
    footer()
    st.stop()
except DataError as e:
    strip.error(str(e) if market == "US" else f"No price data found for {ticker} on NSE. Check the symbol or try the company name.")
    footer()
    st.stop()
except Exception as e:
    strip.error(f"Couldn't load data right now. Please try again in a minute. (Details: {e})")
    footer()
    st.stop()

last_date = prices.index[-1]
last_close = float(prices["close"].iloc[-1])
currency = "₹" if market == "India" else "$"
context, plans = get_plans(ticker, prices, last_date)
plan_by_pattern = {p["pattern"]: p for p in plans}

with strip.container():
    s1, s2 = st.columns([5, 1.3], vertical_alignment="center") if len(candidates) > 1 else (st.container(), None)
    s1.markdown(
        f'<div class="tph-strip done"><span class="tph-tick">✓</span><span><b>Scan complete:</b> '
        f'{esc(company)} ({esc(ticker)}) · {len(prices):,} days checked</span></div>', unsafe_allow_html=True)
    if s2 is not None:
        with s2.popover("Not this company?", width="stretch"):
            st.radio(
                "Other matches", range(len(candidates)), index=choice, key=pick_key,
                format_func=lambda i: f"{candidates[i]['symbol']} — {candidates[i]['name']}",
            )

# ── Answer card ─────────────────────────────────────────────────────────────
with st.container(border=True, key="answer"):
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        trend = str(context["trend"])
        tcls = "down" if "DOWN" in trend.upper() else "up" if "UP" in trend.upper() else ""
        st.markdown(
            f'<div class="tph-co"><h2 style="color:#24414A;font-weight:700">{esc(company)} <span style="color:#5E7479;font-weight:500">{esc(ticker)}</span></h2>'
            f'<div class="tph-meta">Last close {currency}{last_close:,.2f} · {last_date:%d %b %Y} · '
            f'{len(prices):,} daily bars since {prices.index[0]:%b %Y} (split/dividend-adjusted, Yahoo Finance)</div>'
            f'<div class="tph-chips"><span class="tph-chip {tcls}">Trend: {esc(trend)}</span>'
            f'<span class="tph-chip">Position: {esc(context["technical"])}</span>'
            f'<span class="tph-chip">Bow tie: {esc(context["bow_tie"])}</span></div></div>',
            unsafe_allow_html=True,
        )
        if not plans:
            kind, text = "NONE", "The original bot's rules find no setup today, so they would not have taken a trade."
            badge = "NO SETUP"
        else:
            top = plans[0]
            kind = badge = top["decision"]
            setup = f"{'bullish' if top['direction'] == 'LONG' else 'bearish'} {friendly(top['pattern'])}"
            if kind == "TAKE":
                text = (f"The original bot's rules would have taken a {setup} setup today. "
                        "See the hypothetical trade in its tab below.")
            else:
                text = f"The original bot's rules would have skipped today: their top pick is {setup}, but {top['reason']}."
        st.markdown(f'<div class="tph-verdict {kind}"><span class="tph-badge {kind}">{esc(badge)}</span>'
                    f'<span>{esc(text)}</span></div>', unsafe_allow_html=True)
        st.caption("This describes how a fixed set of rules behaves. It is not a recommendation to buy or sell.")
        ruleset = st.segmented_control(
            "Rule set", ["Original", "Version 3"], default="Original", key="ruleset",
            help="Original = the bot's rules exactly as coded. Version 3 = the same patterns, but only when they "
                 "happen at a real location with a supporting factor and the weekly trend agrees.",
        ) or "Original"

    hist_all = hist
    hist = hist_all if ruleset == "Original" else hist_all[hist_all["v3_pass"]]
    today_all = hist_all[hist_all["date"] == last_date]
    today = hist[hist["date"] == last_date]
    firing = list(dict.fromkeys(today["pattern"]))

    with right:
        horizon = st.segmented_control(
            "What history says", HORIZONS, default=20, format_func=lambda h: HORIZON_LABEL[h].split(" (")[0],
            key="horizon",
        ) or 20
        o = outlook(hist, last_date, horizon)
        tip = method_text(ticker, horizon, o, ruleset)
        m1, m2, m3 = st.container(key="stats").columns(3)
        if o["status"] == "unknown":
            m1.metric("Bullish", "–", help=tip)
            m2.metric("Bearish", "–", help=tip)
            m3.metric("Outlook", "Unknown", help=tip)
            extra = ""
            if ruleset == "Version 3" and len(today_all):
                extra = (" The Original rules do confirm "
                         + ", ".join(friendly(p) for p in dict.fromkeys(today_all["pattern"]))
                         + " today: switch the rule set to Original to see its history.")
            st.info(f"**Unknown under the {ruleset} rules:** no pattern is confirmed today, so there is no "
                    "historical evidence to go on." + extra)
        else:
            with m1.container(key="bull"):
                st.metric("Bullish", share(o["bullish"]), help=tip)
            with m2.container(key="bear"):
                st.metric("Bearish", share(o["bearish"]), help=tip)
            m3.metric("Past cases", f"{o['sample']}", help=tip)
            b, r = (o["bullish"] or 0) * 100, (o["bearish"] or 0) * 100
            st.markdown(f'<div class="tph-split"><i style="width:{b:.1f}%;background:var(--green)"></i>'
                        f'<span style="flex:1"></span><i style="width:{r:.1f}%;background:var(--red)"></i></div>',
                        unsafe_allow_html=True)
            st.caption(f"Over the {HORIZON_LABEL[horizon]} after the same pattern(s) appeared in {ticker}'s past. "
                       "Hover over (or tap) the ? icons for how this is calculated.")
            if o["sample"] < SMALL_SAMPLE:
                st.warning(f"Small sample: only {o['sample']} past case(s). Treat these percentages as anecdotal.")
        with st.expander("How is this calculated?"):
            st.write(tip)

# ── What the bot checked (table data, also used for the Excel file) ────────
rows = []
for key, name, desc in RULES:
    fired_today = today_all[today_all["rule"] == key]
    past_orig = hist_all[(hist_all["rule"] == key) & (hist_all["date"] < last_date)]
    if fired_today.empty:
        status, status_v3 = "— Not confirmed", "— Not confirmed"
    else:
        dirs = sorted(set(fired_today["direction"]))
        status = " / ".join("✅ Bullish" if d == "LONG" else "✅ Bearish" for d in dirs)
        status += " — " + ", ".join(friendly(p) for p in fired_today["pattern"])
        parts = []
        for _, rr in fired_today.iterrows():
            if rr["v3_pass"]:
                parts.append(f"✅ passes ({rr['v3_location']}; {rr['v3_support']})")
            else:
                parts.append("✗ filtered out: " + v3.why_not(rr))
        status_v3 = " / ".join(parts)
    rows.append({"Rule": name, "Original": status, "Version 3": status_v3,
                 "Seen in 25 yrs (Original / V3)": f"{len(past_orig)} / {int(past_orig['v3_pass'].sum())}",
                 "What it looks for": desc})
checked = pd.DataFrame(rows)

# ── Tabs: one per confirmed rule, then all rules, then download ─────────────
tab_names = [friendly(p) for p in firing] + ["All 10 rules checked", "Download"]
tabs = st.tabs(tab_names)

for tab, pattern in zip(tabs, firing):
    with tab:
        occ_all = hist[hist["pattern"] == pattern].sort_values("date")
        direction = occ_all["direction"].iloc[0]
        rule_key = occ_all["rule"].iloc[0]
        past = occ_all[occ_all["date"] < last_date]
        n_past = len(past)

        h1, h2 = st.columns([5, 1], vertical_alignment="center")
        h1.markdown(f"**{RULE_NAME[rule_key]}** · {RULE_DESC[rule_key]}")
        h2.markdown(f'<div style="text-align:right"><span class="tph-tag {direction}">'
                    f'{"📈 Bullish" if direction == "LONG" else "📉 Bearish"}</span></div>', unsafe_allow_html=True)
        own, indep = split_factors(pattern, today[today["pattern"] == pattern]["factors"].iloc[0])
        st.caption(f"**Supporting signs today ({len(indep)}):** " + (", ".join(indep) or "none")
                   + "  \n**Part of the pattern itself:** " + (", ".join(own) or "none"))

        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            pl = plan_by_pattern.get(pattern)
            if pl:
                st.markdown('<div class="tph-label">If a trade were taken, it could look like this</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="tph-hyp">Hypothetical levels worked out by the original bot\'s code. '
                            'An illustration of the rules, not a recommendation.</div>', unsafe_allow_html=True)
                plan_box = st.container(key="plan")
                a1, a2, a3 = plan_box.columns(3)
                a1.metric("Entry", f"{pl['entry']:,.2f}")
                a2.metric("Stop", f"{pl['stop']:,.2f}")
                a3.metric("Target", f"{pl['target']:,.2f}")
                a4, a5, _ = plan_box.columns(3)
                a4.metric("Reward:risk", f"{pl['r']:.2f}×",
                          help=f"The original rules require at least {MIN_R}× before they would take a trade.")
                own_n, ind_n = (len(x) for x in split_factors(pattern, pl["factors"]))
                a5.metric("Confidence", f"{pl['confidence']}/5",
                          delta=f"{ind_n} independent sign{'s' if ind_n != 1 else ''}", delta_color="off",
                          delta_arrow="off",
                          help=(f"The bot counts every distinct label, capped at 5. {own_n} of them describe "
                                f"the pattern itself, so {ind_n} are separate supporting signs."))
                msg = (f"**The original rules {DECISION_TEXT.get(pl['decision'], pl['decision'].lower())}:** "
                       f"{pl['reason']}" + (" · their top pick today" if pl["top_pick"] else ""))
                (st.success if pl["decision"] == "TAKE" else st.warning if pl["decision"] == "SKIP" else st.info)(msg)

        with c_right:
            if n_past:
                st.markdown('<div class="tph-label">Every past time on the chart</div>', unsafe_allow_html=True)
                chart_px = prices[["close"]].reset_index().rename(columns={"ts": "date"})
                line = alt.Chart(chart_px).mark_line(strokeWidth=1, color="#9AAEB1").encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("close:Q", title="Close (adjusted)", scale=alt.Scale(zero=False)),
                )
                marks = occ_all[["date", "close", "ret_20d"]].copy()
                marks["Outcome (1 month)"] = [
                    "Pending" if pd.isna(x) else ("Went rule's way" if x > 0 else "Went against")
                    for x in directional(marks["ret_20d"], direction)
                ]
                marks["1-month move"] = [pct(x) for x in marks["ret_20d"]]
                points = alt.Chart(marks).mark_point(filled=True, size=45).encode(
                    x="date:T", y="close:Q",
                    color=alt.Color("Outcome (1 month):N", legend=alt.Legend(orient="bottom", title=None),
                                    scale=alt.Scale(domain=["Went rule's way", "Went against", "Pending"],
                                                    range=["#2E8B62", "#B8574B", "#2F7F86"])),
                    tooltip=[alt.Tooltip("date:T"), alt.Tooltip("close:Q", format=",.2f"), "1-month move",
                             "Outcome (1 month)"],
                )
                st.altair_chart((line + points).properties(height=240), width="stretch")


        st.markdown(f'<div class="tph-label">Every past time on {esc(ticker)} ({n_past})</div>',
                    unsafe_allow_html=True)
        if n_past < SMALL_SAMPLE:
            st.warning(f"Small sample: this pattern was confirmed only {n_past} time(s) before in "
                       f"{ticker}'s history.")
        if n_past:
            stats = rule_stats(past, direction)
            st.dataframe(pd.DataFrame({
                "After": [HORIZON_LABEL[s.horizon].split(" (")[0] for s in stats],
                "Cases": [s.sample for s in stats],
                "Went rule's way": [share(s.win_rate) for s in stats],
                "Average": [pct(s.average) for s in stats],
                "Median": [pct(s.median) for s in stats],
                "Worst": [pct(s.worst) for s in stats],
                "Best": [pct(s.best) for s in stats],
            }), hide_index=True, width="stretch")
            st.caption("Returns in this table are measured in the rule's direction "
                       "(for a bearish rule, a price drop counts as a gain).")

        if n_past:
            table = past.sort_values("date", ascending=False)
            res = directional(table["ret_20d"], direction)
            with st.expander(f"Every past case ({n_past}): {(res <= 0).sum()} went against the rule after 1 month"):
                st.caption("Moves shown are plain price change.")
                st.dataframe(pd.DataFrame({
                    "Date": table["date"].dt.strftime("%Y-%m-%d"),
                    "Price": table["close"].map(lambda v: f"{v:,.2f}"),
                    **{HORIZON_LABEL[h].split(" (")[0]: table[f"ret_{h}d"].map(pct) for h in HORIZONS},
                    "After 1 month": ["Pending" if pd.isna(x) else ("✅ Rule's way" if x > 0 else "❌ Against")
                                      for x in res],
                }), hide_index=True, width="stretch", height=min(38 + 35 * n_past, 380))

with tabs[-2]:
    side = []
    for name, hs in (("Original", hist_all), ("Version 3", hist_all[hist_all["v3_pass"]])):
        oo = outlook(hs, last_date, horizon)
        side.append({
            "Rule set": name,
            "Confirmed today": ", ".join(friendly(p) for p in oo["patterns"]) or "None → Unknown",
            "Bullish": share(oo["bullish"]), "Bearish": share(oo["bearish"]), "Past cases": oo["sample"],
        })
    st.markdown(f"**Original vs Version 3** ({HORIZON_LABEL[horizon].split(' (')[0]} ahead)")
    st.dataframe(pd.DataFrame(side), hide_index=True, width="stretch")
    st.markdown("**Every rule, checked on the latest bar**")
    st.dataframe(
        checked, hide_index=True, width="stretch",
        column_config={"What it looks for": st.column_config.TextColumn(width="large")},
    )
    st.caption(f"Original: {today_all['rule'].nunique()} of 10 rules confirmed on the latest bar. "
               f"Version 3: {today_all[today_all['v3_pass']]['rule'].nunique()} of 10. "
               f"Weekly trend (completed weeks): {v3.weekly_trend_by_day(prices).iloc[-1]}.")

# ── Download ────────────────────────────────────────────────────────────────
fired_any = list(dict.fromkeys(today_all["pattern"]))
evidence = hist_all[hist_all["pattern"].isin(fired_any)] if fired_any else hist_all
evidence = evidence.copy()
evidence.insert(0, "ticker", ticker)
evidence["rule"] = evidence["rule"].map(RULE_NAME)
evidence["date"] = evidence["date"].dt.strftime("%Y-%m-%d")
summary_rows = []
for h, rs in [(h, rs) for rs in ("Original", "Version 3") for h in HORIZONS]:
    oh = outlook(hist_all if rs == "Original" else hist_all[hist_all["v3_pass"]], last_date, h)
    summary_rows.append({
        "Rule set": rs,
        "Horizon": HORIZON_LABEL[h],
        "Outlook": "Unknown (no pattern confirmed)" if oh["status"] == "unknown" else "From past cases",
        "Bullish %": None if pd.isna(oh["bullish"]) else round(oh["bullish"] * 100, 1),
        "Bearish %": None if pd.isna(oh["bearish"]) else round(oh["bearish"] * 100, 1),
        "Past cases": oh["sample"],
        "How it is calculated": method_text(ticker, h, oh, rs),
    })
with tabs[-1]:
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
    st.markdown("Everything behind the numbers on this page: the outlook for every horizon, how it is "
                "calculated, every rule checked, and every past case.")
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button(
        "⬇ Excel (all evidence)", buf.getvalue(),
        file_name=f"{ticker}_pattern_evidence_{last_date:%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
    )
    d2.download_button(
        "⬇ CSV (past cases)", evidence.to_csv(index=False).encode(),
        file_name=f"{ticker}_past_cases_{last_date:%Y%m%d}.csv", mime="text/csv", width="stretch",
    )

footer()
