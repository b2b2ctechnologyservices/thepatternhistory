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

import rules
from pattern_history import doc_rules
from pattern_history.catalog import RULE_DESC, RULE_NAME, RULES, YEARLY_CHECKS, YEARLY_DESC, YEARLY_NAME
from pattern_history.data import MARKETS, DataError, ProviderBusy, load_daily, resolve
from pattern_history.factors import is_independent, label as factor_label, split as split_factors
from pattern_history.plan import MIN_CONFIDENCE, MIN_R, todays_plans
from pattern_history.history import directional, outlook, rule_stats
from pattern_history.multi import DAILY_YEARS, scan_all
from pattern_history.timeframes import TIMEFRAMES

SMALL_SAMPLE = 10
DISCLAIMER = (
    "Historical statistics for education only. Not investment advice, and not a recommendation to buy or "
    "sell any security. Hypothetical trades show how a fixed set of rules would behave; nobody is told to "
    "place them. Past patterns do not guarantee future results."
)
# Look-ahead labels per timeframe (keyed by the forward-return suffix: d/w/m/y).
HZS = {
    "d": {1: "1 day", 5: "1 week", 20: "1 month", 60: "3 months"},
    "w": {1: "1 week", 4: "1 month", 13: "3 months", 26: "6 months"},
    "m": {1: "1 month", 3: "3 months", 6: "6 months", 12: "1 year"},
    "y": {1: "1 year"},
}
HZS_LONG = {
    "d": {1: "the next trading day", 5: "1 week (5 trading days)", 20: "1 month (20 trading days)",
          60: "3 months (60 trading days)"},
    "w": {1: "the next week", 4: "1 month (4 weeks)", 13: "3 months (13 weeks)", 26: "6 months (26 weeks)"},
    "m": {1: "the next month", 3: "3 months", 6: "6 months", 12: "1 year (12 months)"},
    "y": {1: "the next calendar year"},
}
DEFAULT_H = {"d": 20, "w": 4, "m": 3, "y": 1}
# How each timeframe's latest bar is described ("on ... chart <WHEN>").
WHEN = {"daily": "today", "weekly": "in the latest finished week", "monthly": "in the latest finished month",
        "yearly": "this year so far"}
BIGGER = {"daily": "weekly", "weekly": "monthly"}
TREND_TEXT = {
    "UP": "the price has mostly been rising over recent weeks",
    "STRONG_UP": "the price has been rising over recent weeks, and the rise is speeding up",
    "DOWN": "the price has mostly been falling over recent weeks",
    "STRONG_DOWN": "the price has been falling over recent weeks, and the fall is speeding up",
    "NEUTRAL": "no clear up or down direction lately",
}
POSITION_TEXT = {
    "STRONG": "the price is above its 20-day average price, which is above its 50-day average, a sign of strength",
    "WEAK": "the price is below its 50-day average price, a sign of weakness",
    "MIXED": "the price is neither clearly above nor below its 20- and 50-day average prices",
}
BOWTIE_TEXT = {
    "none": "no 'bow tie' right now (that's when the 20-day and 50-day average prices cross, or close in fast "
            "on each other, which can come before a change of direction)",
    "BULL": "the 20-day average price has just crossed above the 50-day one, or is rising fast toward it, "
            "which can come before a rise",
    "BEAR": "the 20-day average price has just crossed below the 50-day one, or is falling fast toward it, "
            "which can come before a fall",
}

ASSETS = Path(__file__).resolve().parent / "assets"
st.set_page_config(page_title="ThePatternHistory", page_icon=str(ASSETS / "icon.png"), layout="wide")


def _b64(name: str) -> str:
    import base64

    return base64.b64encode((ASSETS / name).read_bytes()).decode()


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=200)
def get_history(ticker: str) -> dict:
    """Every timeframe for a ticker: {"daily": TimeframeResult, "weekly": ..., "monthly": ..., "yearly": ...}."""
    return scan_all(load_daily(ticker, years=0))


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
    import re

    return re.sub(r"\bMa(\d*)\b", r"MA\1", pattern.replace("_", " ").title())


def esc(s) -> str:
    return html.escape(str(s))


def points(direction: str) -> str:
    if direction == "NONE":
        return "is a simple fact about the year, not a pattern that points up or down"
    return "points up (it suggests the price may rise)" if direction == "LONG" \
        else "points down (it suggests the price may fall)"


def tile(value: str, label: str, note: str, cls: str = "") -> str:
    return (f'<div class="tph-tile"><div class="tph-tl">{esc(label)}</div><div class="tph-tv {cls}">{esc(value)}</div>'
            f'<div class="tph-tn">{esc(note)}</div></div>')


def tiles(items: list[str], cols: int = 3) -> None:
    st.markdown(f'<div class="tph-tiles c{cols}">' + "".join(items) + "</div>", unsafe_allow_html=True)


def plain_reason(pl: dict, currency: str) -> str:
    """The rules' TAKE / SKIP decision, in words a first-time visitor understands."""
    risk, reward = abs(pl["entry"] - pl["stop"]), abs(pl["target"] - pl["entry"])
    money = f"risking {currency}{risk:,.2f} per share to possibly gain {currency}{reward:,.2f}"
    if pl["decision"] == "NOT CONSIDERED":
        return ("No. The rules only ever look at the day's strongest pattern, and this isn't it, "
                "so they would ignore it.")
    if pl["decision"] == "TAKE":
        return (f"Yes, they would. The possible gain is {pl['r']:.2f} times the possible loss ({money}); the rules "
                f"want at least {MIN_R}×. And {pl['confidence']} signs line up; they want at least {MIN_CONFIDENCE}.")
    if pl["confidence"] < MIN_CONFIDENCE:
        return (f"No. The rules want at least {MIN_CONFIDENCE} signs lining up before they act, "
                f"and found only {pl['confidence']}.")
    if "reward:risk" in pl["reason"]:
        return (f"No, but it's close. The rules only act when the possible gain is at least {MIN_R} times the "
                f"possible loss. Here it's {pl['r']:.2f} times ({money}), just short, so they would pass.")
    return "No. The rules couldn't place sensible exit prices for this pattern, so they would pass."


def confidence_html(pattern: str, factors, confidence: int) -> str:
    """Every clue behind the confidence score, marked as independent or part of the pattern itself."""
    if isinstance(factors, str):
        factors = [f.strip() for f in factors.split(",") if f.strip()]
    codes = list(dict.fromkeys(factors))
    ind = [c for c in codes if is_independent(c, pattern)]
    own = [c for c in codes if c not in ind]
    cap = f", so the score is capped at {confidence}" if len(codes) > confidence else ""
    rows = "".join(f'<li><span class="tph-dot ind"></span><b>{esc(factor_label(c))}</b> '
                   '<span>independent sign</span></li>' for c in ind)
    rows += "".join(f'<li><span class="tph-dot own"></span>{esc(factor_label(c))} '
                    '<span>describes the pattern itself</span></li>' for c in own)
    return (
        f'<div class="tph-conf"><h4>How the {confidence}/5 confidence is worked out</h4>'
        f"These rules give 1 point for every different clue they note about today's setup, up to 5. "
        f"They noted <b>{len(codes)} clue{'s' if len(codes) != 1 else ''}</b>{cap}:<ul>{rows}</ul>"
        f"<b>{len(ind)}</b> of them are <b>independent signs</b>: separate evidence that agrees with the pattern "
        f"(the trend, where the price sits against its averages, volume, a nearby round number, a quiet candle, "
        f"a bow tie). The other {len(own)} only describe the pattern itself, so they don't add evidence. "
        "That means the score can look stronger than it is; the independent signs are the ones worth weighing.</div>"
    )


def method_text(ticker: str, horizon: int, o: dict, tf) -> str:
    unit = tf.unit
    span = f"~{DAILY_YEARS} years" if tf.key == "daily" else "the full price history"
    if o["status"] == "unknown":
        return (f"None of our {tf.label.lower()} patterns showed up on {ticker}'s latest {tf.label.lower()} price "
                "bar, so there is nothing to look up in history.")
    return (
        f"Step 1: {WHEN[tf.key]} these pattern(s) showed up on {ticker}: "
        f"{', '.join(friendly(p) for p in o['patterns'])}. "
        f"Step 2: we went back through {span} of {ticker}'s {tf.label.lower()} prices and found every past "
        f"{unit} when the same pattern showed up, using only the prices known up to then "
        f"({o['sample']} {unit}s; a {unit} where two of them showed up counts once). "
        f"Step 3: for each of those {unit}s we compared the closing price with the close "
        f"{HZS_LONG[tf.suffix][horizon]} later. "
        f"'Went up' = share of {unit}s where the price was higher; 'Went down' = share where it was lower. "
        f"The latest {unit} isn't counted because we don't know its outcome yet. "
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
.tph-tiles{display:grid;gap:10px;margin:6px 0 8px}
.tph-tiles.c3{grid-template-columns:repeat(3,minmax(0,1fr))} .tph-tiles.c2{grid-template-columns:repeat(2,minmax(0,1fr))}
.tph-tile{background:var(--mist);border-radius:10px;padding:10px 12px}
.tph-tl{font-size:.78rem;color:var(--muted);font-weight:600}
.tph-tv{font-size:1.7rem;font-weight:700;color:var(--slate);line-height:1.2}
.tph-tv.g{color:var(--green)} .tph-tv.r{color:var(--red)}
.tph-tn{font-size:.76rem;color:var(--muted);line-height:1.3;margin-top:2px}
.tph-box{border-radius:10px;padding:12px 14px;margin:10px 0 6px;font-size:.92rem;line-height:1.45}
.tph-box.find{background:var(--mist)}
.tph-box h4{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 4px;padding:0}
.tph-short{font-size:.95rem;background:#fff;border-left:4px solid var(--teal);padding:8px 12px;margin:6px 0}
.tph-ctx{font-size:.8rem;color:var(--muted);line-height:1.45;margin-top:4px}
.tph-ctx b{color:var(--slate)}
.tph-band{border-radius:8px;padding:6px 12px;font-size:.85rem;font-weight:600;margin:2px 0 10px}
.tph-next{font-size:.95rem;color:var(--muted);padding:8px 12px;border:1px dashed var(--line);border-radius:10px}
.tph-conf{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:4px 0 10px;font-size:.85rem;line-height:1.45}
.tph-conf h4{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 4px;padding:0}
.tph-conf ul{list-style:none;margin:6px 0;padding:0} .tph-conf li{margin:2px 0}
.tph-conf li span:last-child{color:var(--muted);font-size:.78rem;margin-left:4px}
.tph-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.tph-dot.ind{background:var(--green)} .tph-dot.own{background:#C9D6D3}
@media (max-width:640px){.tph-verdict{flex-direction:column;align-items:flex-start}.tph-tv{font-size:1.35rem}.tph-tiles{gap:6px}.tph-tile{padding:8px}}
</style>""", unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="tph-foot">{esc(DISCLAIMER)}</div>', unsafe_allow_html=True)


# ── Top bar: logo, then country, then search ─────────────────────────────────
THEMES = {  # accent, deep accent, soft tint, band colour, band text
    "US": ("#2F7F86", "#24414A", "#EEF5F2", "#E3EFEC", "🇺🇸  US market · NYSE & Nasdaq · prices in $"),
    "India": ("#C46A1C", "#6B3A12", "#FBF1E6", "#F8E6D2", "🇮🇳  India market · NSE · prices in ₹"),
}
with st.container(key="topbar"):
    c_logo, c_mkt, c_search = st.columns([1.15, 0.9, 1.9], vertical_alignment="bottom")
    c_logo.markdown(f'<div class="tph-logo"><img src="data:image/png;base64,{_b64("logo.png")}" '
                    'alt="ThePatternHistory"></div>', unsafe_allow_html=True)
    market = c_mkt.segmented_control(
        "1. Choose a market", list(MARKETS), default=None, format_func=lambda m: MARKETS[m]["label"],
        key="market",
    )
    query, submitted = "", False
    if market:
        with c_search.form("scan", border=False):
            f1, f2 = st.columns([4, 1.2], vertical_alignment="bottom", gap="small")
            query = f1.text_input(
                "2. Type a ticker or company name",
                placeholder="e.g. MU or Micron" if market == "US" else "e.g. SBIN or State Bank of India",
            )
            submitted = f2.form_submit_button("Scan", type="primary", width="stretch")
    else:
        c_search.markdown('<div class="tph-next">← Start here: choose <b>US</b> or <b>India</b>, '
                          'then the search box appears.</div>', unsafe_allow_html=True)

if market:
    acc, deep, tint, band, band_text = THEMES[market]
    st.markdown(f"""<style>
:root{{--teal:{acc};--slate:{deep};--mist:{tint}}}
.st-key-topbar{{border-bottom:3px solid {acc}!important}}
[data-testid="stBaseButton-primaryFormSubmit"]{{background:{acc}!important;border-color:{acc}!important}}
button[data-variant="segmented_control"][aria-checked="true"]{{background:{tint}!important;color:{acc}!important;border-color:{acc}!important}}
button[data-variant="segmented_control"][aria-checked="true"] *{{color:{acc}!important}}
.stTabs [aria-selected="true"]{{color:{acc}!important}} .stTabs [data-baseweb="tab-highlight"]{{background:{acc}!important}}
</style><div class="tph-band" style="background:{band};color:{deep}">{band_text}</div>""",
                unsafe_allow_html=True)

if submitted and query.strip():
    st.session_state["query"] = (query.strip(), market)

q, q_market = st.session_state.get("query", ("", market))
if not market or not q or q_market != market:
    st.markdown(
        '<div class="tph-hero"><h1>What happened every time this pattern showed up before?</h1></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Traders look for repeating shapes in price charts, called **patterns**. This site checks a stock for "
        "well-known patterns on four timeframes: **daily, weekly, monthly and yearly** price bars. Then it looks "
        "up **every past time the same pattern showed up** in that stock's own history, and shows you what the "
        "price did next.\n\n"
        "**Daily, weekly and monthly** bars are checked for 13 patterns (the 10 classic ones plus 3 from a "
        "trend-and-level method: a first test of a major average, a first pullback to the 20-bar average, and "
        "a bow tie), along with supporting signs such as prior turning points, round numbers, price gaps and "
        "Fibonacci levels. **Yearly** bars are too few for those patterns, so for them we use simple checks "
        "against last year's range.\n\n"
        "**How to use it:** first choose **US** or **India** at the top, then type a ticker or company name "
        "and press **Scan**.\n\n"
        "**What you'll see:**\n"
        "1. **Today's pattern**: pick Daily, Weekly, Monthly or Yearly to see which patterns (if any) showed "
        "up, and what they mean.\n"
        "2. **What happened before**: how often the price went up or down afterwards, in the stock's own past.\n"
        "3. **A hypothetical trade**: what a trade could look like if someone followed a fixed set of rules. "
        "It's an illustration, not advice."
    )
    st.caption("US tickers work as-is (AAPL, BRK-B). For India type the NSE symbol (SBIN) or the company name; "
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
     "Loading its price history and checking every pattern on every day, week, month and year. "
     "The first scan of a stock takes ~30–60s; repeat scans are instant.")
try:
    results = get_history(ticker)
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

daily_prices = results["daily"].bars
currency = "₹" if market == "India" else "$"
context, plans = get_plans(ticker, daily_prices, daily_prices.index[-1])
plan_by_pattern = {p["pattern"]: p for p in plans}
counts = " · ".join(f"{len(r.bars):,} {r.tf.unit}s" for r in results.values())


def pattern_name(pattern: str) -> str:
    if pattern in YEARLY_NAME:
        return YEARLY_NAME[pattern]
    rule = rule_of.get(pattern)
    return RULE_NAME[rule] if rule in RULE_NAME else friendly(pattern)


def pattern_desc(rule: str, pattern: str) -> str:
    return YEARLY_DESC.get(pattern) or RULE_DESC.get(rule, "")


with strip.container():
    s1, s2 = st.columns([5, 1.3], vertical_alignment="center") if len(candidates) > 1 else (st.container(), None)
    s1.markdown(
        f'<div class="tph-strip done"><span class="tph-tick">✓</span><span><b>Scan complete:</b> '
        f'{esc(company)} ({esc(ticker)}) · checked {esc(counts)}</span></div>', unsafe_allow_html=True)
    if s2 is not None:
        with s2.popover("Not this company?", width="stretch"):
            st.radio(
                "Other matches", range(len(candidates)), index=choice, key=pick_key,
                format_func=lambda i: f"{candidates[i]['symbol']} — {candidates[i]['name']}",
            )

# ── Answer card ─────────────────────────────────────────────────────────────
with st.container(border=True, key="answer"):
    tf_key = st.segmented_control(
        "3. Pick a timeframe", list(TIMEFRAMES), default="daily", key="timeframe",
        format_func=lambda k: TIMEFRAMES[k].label,
        help="Each bar on the chart can be one day, one week (Mon–Fri), one month or one calendar year. "
             "Bigger bars show the bigger picture, but there are fewer of them in history.",
    ) or "daily"
    res = results[tf_key]
    tf, prices, hist = res.tf, res.bars, res.hist
    HZ, HORIZONS, sfx = HZS[tf.suffix], tf.horizons, tf.suffix
    main_h = DEFAULT_H[sfx]
    if prices.empty:
        st.info(f"There isn't enough price history for {ticker} to build {tf.label.lower()} bars yet.")
        footer()
        st.stop()
    last_date = prices.index[-1]
    last_close = float(prices["close"].iloc[-1])
    today = hist[hist["date"] == last_date]
    firing = list(dict.fromkeys(today["pattern"]))
    rule_of = dict(zip(today["pattern"], today["rule"]))
    dir_of = dict(zip(today["pattern"], today["direction"]))
    n_checks = len(YEARLY_CHECKS) if tf_key == "yearly" else len(RULES)
    what = "yearly checks" if tf_key == "yearly" else f"{n_checks} patterns"

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        chips, ctx = [], []
        if tf_key == "daily":
            trend, tech, bow = str(context["trend"]), str(context["technical"]), str(context["bow_tie"])
            tcls = "down" if "DOWN" in trend else "up" if "UP" in trend else ""
            chips += [f'<span class="tph-chip {tcls}">Trend: {esc(trend.replace("_", " "))}</span>',
                      f'<span class="tph-chip">Position: {esc(tech)}</span>',
                      f'<span class="tph-chip">Bow tie: {esc(bow)}</span>']
            if trend in TREND_TEXT:
                ctx.append(f"<b>Trend {esc(trend.replace('_', ' ').lower())}:</b> {esc(TREND_TEXT[trend])}.")
            if tech in POSITION_TEXT:
                ctx.append(f"<b>Position {esc(tech.lower())}:</b> {esc(POSITION_TEXT[tech])}.")
            if bow in BOWTIE_TEXT:
                ctx.append(f"<b>Bow tie {esc(bow.lower())}:</b> {esc(BOWTIE_TEXT[bow])}.")
        elif tf_key != "yearly":
            t = doc_rules.TREND_TEXT[doc_rules.trend_series(rules._add_indicators(prices)).iloc[-1]]
            tcls = "down" if "down" in t else "up" if "up" in t else ""
            chips.append(f'<span class="tph-chip {tcls}">{esc(tf.label)} trend: {esc(t)}</span>')
            ctx.append(f"<b>{esc(tf.label)} trend {esc(t)}:</b> based on the 20, 50 and 200-{esc(tf.unit)} "
                       "average prices and how far apart they are.")
        if tf_key in BIGGER:
            big = results[BIGGER[tf_key]].bars
            bt = doc_rules.TREND_TEXT[doc_rules.trend_series(rules._add_indicators(big)).iloc[-1]] if len(big) else "sideways"
            bcls = "down" if "down" in bt else "up" if "up" in bt else ""
            chips.append(f'<span class="tph-chip {bcls}">Bigger picture ({BIGGER[tf_key]}): {esc(bt)}</span>')
            ctx.append(f"<b>Bigger picture {esc(bt)}:</b> the trend on the {BIGGER[tf_key]} chart, which usually "
                       "matters more than the smaller one.")
        if tf_key == "daily":
            when = f"Last close {currency}{last_close:,.2f} on {last_date:%d %b %Y}"
        elif tf_key == "yearly":
            when = f"Price {currency}{last_close:,.2f} on {last_date:%d %b %Y} ({last_date:%Y} so far)"
        else:
            when = f"Latest finished {tf.unit} ended {last_date:%d %b %Y}, close {currency}{last_close:,.2f}"
        st.markdown(
            f'<div class="tph-co"><h2 style="color:var(--slate);font-weight:700">{esc(company)} '
            f'<span style="color:#5E7479;font-weight:500">{esc(ticker)}</span></h2>'
            f'<div class="tph-meta">{esc(when)} · '
            f'{len(prices):,} {tf.unit}s of prices since {prices.index[0]:%b %Y}</div>'
            f'<div class="tph-chips">{"".join(chips)}</div>'
            f'<div class="tph-ctx">{" ".join(ctx)}</div></div>',
            unsafe_allow_html=True,
        )

        # What we found today
        noun = "check" if tf_key == "yearly" else "pattern"
        if not firing:
            found = (f"<b>No pattern {WHEN[tf_key]}.</b> None of our {what} showed up on {esc(ticker)}'s "
                     f"{tf.label.lower()} chart. That's normal: most {tf.unit}s have none. Try another "
                     "timeframe or stock.")
        elif len(firing) == 1:
            p = firing[0]
            found = (f"We found <b>1 {noun}</b> on {esc(ticker)}'s {tf.label.lower()} chart {WHEN[tf_key]}: "
                     f"<b>{esc(pattern_name(p))}</b>, which {esc(points(dir_of[p]))}.<br>"
                     f"<span style='color:var(--muted)'>What it is: {esc(pattern_desc(rule_of[p], p))}</span>")
        else:
            items = "".join(f"<li><b>{esc(pattern_name(p))}</b>, which {esc(points(dir_of[p]))}</li>" for p in firing)
            found = (f"We found <b>{len(firing)} {noun}s</b> on {esc(ticker)}'s {tf.label.lower()} chart "
                     f"{WHEN[tf_key]}:"
                     f"<ul style='margin:4px 0 0 18px;padding:0'>{items}</ul>"
                     "<span style='color:var(--muted)'>Each one has its own tab below with what it means.</span>")
        st.markdown(f'<div class="tph-box find"><h4>{esc(tf.label)} pattern</h4>{found}</div>', unsafe_allow_html=True)

        # Would the rules act on it?
        if plans and tf_key == "daily":
            top = plans[0]
            kind = top["decision"]
            badge = "WOULD ACT" if kind == "TAKE" else "WOULD PASS"
            text = (f"Looking at the strongest pattern today ({pattern_name(top['pattern'])}): "
                    + plain_reason(top, currency))
            st.markdown(
                f'<div class="tph-verdict {kind}"><span class="tph-badge {kind}">{badge}</span><span>'
                f'<b>Would these trading rules act on it?</b><br>{esc(text)}</span></div>',
                unsafe_allow_html=True)
            st.caption("We test every pattern against one fixed set of trading rules, so you can see how a "
                       "disciplined approach would treat it. It is not a recommendation to buy or sell.")

    with right:
        st.markdown('<div class="tph-label">What happened after this pattern in the past</div>',
                    unsafe_allow_html=True)
        horizon = st.segmented_control(
            "Look ahead", HORIZONS, default=main_h, format_func=lambda h: HZ[h], key=f"horizon_{tf_key}",
            help="How far after each past pattern we checked the price.",
        ) or main_h
        o = outlook(hist, last_date, horizon, sfx)
        tip = method_text(ticker, horizon, o, tf)
        n = o["sample"]
        if o["status"] == "unknown":
            st.markdown(f'<div class="tph-short">No pattern {WHEN[tf_key]}, so there is nothing in the past to '
                        'compare with. We only show history when a pattern shows up.</div>', unsafe_allow_html=True)
        elif not n:
            st.markdown(f'<div class="tph-short">This pattern has never shown up before on {esc(ticker)} with a '
                        'known outcome, so there is no history to go on.</div>', unsafe_allow_html=True)
        else:
            k_up, k_dn = round(o["bullish"] * n), round(o["bearish"] * n)
            tiles([
                tile(share(o["bullish"]), "Went up", f"Price was higher {HZ[horizon]} later", "g"),
                tile(share(o["bearish"]), "Went down", f"Price was lower {HZ[horizon]} later", "r"),
                tile(f"{n}", "Past cases", f"{tf.unit.capitalize()}s this showed up before on {ticker}"),
            ])
            st.markdown(f'<div class="tph-split"><i style="width:{o["bullish"] * 100:.1f}%;background:var(--green)"></i>'
                        f'<span style="flex:1"></span><i style="width:{o["bearish"] * 100:.1f}%;background:var(--red)">'
                        '</i></div>', unsafe_allow_html=True)
            dirs = {dir_of[p] for p in o["patterns"] if p in dir_of}
            verdict = ""
            if len(dirs) == 1 and dirs <= {"LONG", "SHORT"}:
                agree = o["bullish"] if dirs == {"LONG"} else o["bearish"]
                verdict = (" So history has mostly agreed with this pattern." if agree >= 0.55 else
                           " So history has more often gone against this pattern." if agree <= 0.45 else
                           " So history is close to a coin flip here.")
            st.markdown(f'<div class="tph-short"><b>In short:</b> in {k_up} of {n} past cases, {esc(ticker)}\'s price '
                        f'was higher {HZ[horizon]} later, and in {k_dn} it was lower.{esc(verdict)}</div>',
                        unsafe_allow_html=True)
            if n < SMALL_SAMPLE:
                st.warning(f"Only {n} past case(s): too few to read much into. Treat these percentages as anecdotes.")
        with st.expander("How is this calculated?"):
            st.write(tip)

# ── Tabs: one per pattern found today, then all patterns, then download ────
names = [pattern_name(p) for p in firing]
tab_names = [f"{nm} ({friendly(p)})" if names.count(nm) > 1 else nm for nm, p in zip(names, firing)]
tabs = st.tabs(tab_names + [f"All {tf.label.lower()} {'checks' if tf_key == 'yearly' else 'patterns'}", "Download"])

for tab, pattern in zip(tabs, firing):
    with tab:
        occ_all = hist[hist["pattern"] == pattern].sort_values("date")
        direction = occ_all["direction"].iloc[0]
        rule_key = occ_all["rule"].iloc[0]
        past = occ_all[occ_all["date"] < last_date]
        n_past = len(past)

        h1, h2 = st.columns([5, 1], vertical_alignment="center")
        h1.markdown(f"**{pattern_name(pattern)}** ({friendly(pattern)}). {pattern_desc(rule_key, pattern)}")
        tag = {"LONG": "📈 Points up", "SHORT": "📉 Points down"}.get(direction, "➖ Just a fact")
        h2.markdown(f'<div style="text-align:right"><span class="tph-tag {direction}">{tag}</span></div>',
                    unsafe_allow_html=True)
        if tf_key != "yearly":
            r0 = today[today["pattern"] == pattern].iloc[0]
            sign_list = [x for x in str(r0["signs"]).split(", ") if x]
            check = ("✅ **Meets the rule-document checklist**: it is at a meaningful location, has at least 2 "
                     "factors, and the bigger-picture trend isn't strongly against it."
                     if r0["meets_rules"] else "✗ **Misses the rule-document checklist**: " + doc_rules.why_not(r0) + ".")
            st.caption(f"**Supporting signs {WHEN[tf_key]} ({len(sign_list)}):** " + (", ".join(sign_list) or "none")
                       + ". These are extra clues that agree with the pattern; more signs usually means a stronger "
                         "setup.  \n**At a meaningful location:** " + (str(r0["location"]) or "no")
                       + f" · **Factors in total:** {int(r0['factor_count'])}.  \n" + check)

        c_left, c_right = st.columns(2, gap="large")
        with c_left:
            pl = plan_by_pattern.get(pattern) if tf_key == "daily" else None
            if pl:
                risk, reward = abs(pl["entry"] - pl["stop"]), abs(pl["target"] - pl["entry"])
                bet = ("a bet that the price <b>falls</b> (a 'short' trade): it would gain if the price drops"
                       if pl["direction"] == "SHORT" else
                       "a bet that the price <b>rises</b> (a 'long' trade): it would gain if the price goes up")
                st.markdown('<div class="tph-label">If a trade were taken, it could look like this</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="tph-hyp">To make this pattern easier to picture, here is what a trade '
                            f'based on it could look like if someone followed these rules exactly. It would be {bet}. '
                            '<b>Entry</b> is where it would start, <b>Stop</b> is where it would get out to keep a '
                            'loss small, and <b>Target</b> is where it would take its gain. This is an example for '
                            'learning, not a suggestion to trade.</div>', unsafe_allow_html=True)
                tiles([
                    tile(f"{pl['entry']:,.2f}", "Entry", "Price where the trade would start"),
                    tile(f"{pl['stop']:,.2f}", "Stop", "Exit here to cap the loss if it goes wrong"),
                    tile(f"{pl['target']:,.2f}", "Target", "Exit here to take the gain if it goes right"),
                ])
                ind_n = len(split_factors(pattern, pl["factors"])[1])
                tiles([
                    tile(f"{pl['r']:.2f}×", "Reward : risk",
                         f"Possible gain ÷ possible loss: risking {currency}{risk:,.2f} to possibly gain "
                         f"{currency}{reward:,.2f} per share. The rules want {MIN_R}× or more."),
                    tile(f"{pl['confidence']}/5", "Confidence",
                         f"Clues these rules noted (max 5). {ind_n} of them are independent signs; "
                         "see the breakdown below."),
                ], cols=2)
                st.markdown(confidence_html(pattern, pl["factors"], pl["confidence"]), unsafe_allow_html=True)
                msg = "**Would these rules act on it?** " + plain_reason(pl, currency)
                (st.success if pl["decision"] == "TAKE" else st.warning if pl["decision"] == "SKIP" else st.info)(msg)
            elif tf_key != "daily":
                st.info(f"Hypothetical trade examples are only worked out for daily patterns. For {tf.label.lower()} "
                        "bars we show the history only.")
            elif rule_key.startswith("doc_"):
                st.info("Hypothetical trade examples are only worked out for the 10 classic patterns. For this "
                        "one we show the history only.")
            else:
                st.info("These rules didn't work out trade levels for this pattern today.")

        with c_right:
            if n_past:
                st.markdown('<div class="tph-label">Every past time, on the price chart</div>', unsafe_allow_html=True)
                chart_px = prices[["close"]].reset_index().rename(columns={"ts": "date"})
                line = alt.Chart(chart_px).mark_line(strokeWidth=1, color="#9AAEB1").encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("close:Q", title=f"Price ({currency})", scale=alt.Scale(zero=False)),
                )
                mcol, after = f"ret_{main_h}{sfx}", f"After {HZ[main_h]}"
                good, bad = (("Went up", "Went down") if direction == "NONE"
                             else ("Went the pattern's way", "Went the other way"))
                marks = occ_all[["date", "close", mcol]].copy()
                marks[after] = ["Not known yet" if pd.isna(x) else (good if x > 0 else bad)
                                for x in directional(marks[mcol], direction)]
                marks[f"{HZ[main_h]} move"] = [pct(x) for x in marks[mcol]]
                pts = alt.Chart(marks).mark_point(filled=True, size=45).encode(
                    x="date:T", y="close:Q",
                    color=alt.Color(f"{after}:N", legend=alt.Legend(orient="bottom", title=None),
                                    scale=alt.Scale(domain=[good, bad, "Not known yet"],
                                                    range=["#2E8B62", "#B8574B", "#2F7F86"])),
                    tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("close:Q", format=",.2f", title="Price"),
                             f"{HZ[main_h]} move", after],
                )
                st.altair_chart((line + pts).properties(height=240), width="stretch")
                way = ("the price went up" if direction == "NONE" else "the price moved the way the pattern pointed")
                st.caption(f"The grey line is {ticker}'s {tf.label.lower()} closing price since "
                           f"{prices.index[0]:%Y} (adjusted for splits and dividends). Each dot is a {tf.unit} this "
                           f"showed up. Green: over the next {HZ[main_h]} {way}. Red: it went the other way. "
                           "Tap or hover a dot for details.")

        st.markdown(f'<div class="tph-label">How this pattern played out on {esc(ticker)} ({n_past} past cases)'
                    '</div>', unsafe_allow_html=True)
        if n_past < SMALL_SAMPLE:
            st.warning(f"This showed up only {n_past} time(s) before on {ticker}: too few to read much into.")
        if n_past:
            stats = rule_stats(past, direction, HORIZONS, sfx)
            st.dataframe(pd.DataFrame({
                "Looking ahead": [HZ[s.horizon] for s in stats],
                "Cases": [s.sample for s in stats],
                ("Price went up" if direction == "NONE" else "Pattern was right"): [share(s.win_rate) for s in stats],
                "Average result": [pct(s.average) for s in stats],
                "Typical (median)": [pct(s.median) for s in stats],
                "Worst": [pct(s.worst) for s in stats],
                "Best": [pct(s.best) for s in stats],
            }), hide_index=True, width="stretch")
            st.caption("How to read this: **Pattern was right** is how often the price moved the way the pattern "
                       "pointed. **Average** and **Typical** are the usual result; **Worst** and **Best** are the "
                       "extremes. Results are measured in the pattern's direction, so for a pattern that points "
                       "down, a price drop shows as a positive number.")

            table = past.sort_values("date", ascending=False)
            res = directional(table[f"ret_{main_h}{sfx}"], direction)
            other = "went down" if direction == "NONE" else "went the other way"
            with st.expander(f"See every past case ({n_past}); {(res <= 0).sum()} {other} after {HZ[main_h]}"):
                st.caption(f"Each row is a past {tf.unit} this showed up. The columns show how much the price "
                           "actually changed afterwards (plain price change, up is positive).")
                ok, no = ("⬆ Up", "⬇ Down") if direction == "NONE" else ("✅ Pattern's way", "❌ Other way")
                st.dataframe(pd.DataFrame({
                    "Date": table["date"].dt.strftime("%Y-%m-%d"),
                    "Price": table["close"].map(lambda v: f"{v:,.2f}"),
                    **{f"After {HZ[h]}": table[f"ret_{h}{sfx}"].map(pct) for h in HORIZONS},
                    f"After {HZ[main_h]} ": ["Not known yet" if pd.isna(x) else (ok if x > 0 else no) for x in res],
                }), hide_index=True, width="stretch", height=min(38 + 35 * n_past, 380))

# ── All patterns ────────────────────────────────────────────────────────────
rows = []
col = "rule" if tf_key != "yearly" else "pattern"
catalog = YEARLY_CHECKS if tf_key == "yearly" else RULES
seen_col = f"Times seen since {prices.index[0]:%Y}"
for key, name, desc in catalog:
    fired_today = today[today[col] == key]
    past_n = int(((hist[col] == key) & (hist["date"] < last_date)).sum())
    if fired_today.empty:
        status = "— Not now"
    else:
        status = " / ".join({"LONG": "✅ Found, points up", "SHORT": "✅ Found, points down"}.get(d, "✅ Yes")
                            for d in dict.fromkeys(fired_today["direction"]))
    rows.append({"Pattern": name, WHEN[tf_key].capitalize(): status, seen_col: past_n, "What it looks for": desc})
checked = pd.DataFrame(rows)

with tabs[-2]:
    st.markdown(f"We check the same **{what}** on every stock's {tf.label.lower()} chart. Here is which ones showed "
                f"up on {ticker} {WHEN[tf_key]}, and how often each one has shown up on {ticker} in the past.")
    st.dataframe(checked, hide_index=True, width="stretch",
                 column_config={"What it looks for": st.column_config.TextColumn(width="large")})
    if tf_key != "yearly":
        st.caption(f"{today[col].nunique()} of {n_checks} patterns showed up on the latest {tf.unit}. A pattern only "
                   "counts when at least 2 of its signs agree. The last 3 come from the rule document (trend and "
                   "major levels first, then supporting signs).")

# ── Download ────────────────────────────────────────────────────────────────
evidence = hist[hist["pattern"].isin(firing)] if firing else hist
evidence = evidence.copy()
evidence.insert(0, "ticker", ticker)
evidence["rule"] = [pattern_name(p) for p in evidence["pattern"]]
evidence.insert(1, "timeframe", tf.label)
evidence["date"] = evidence["date"].dt.strftime("%Y-%m-%d")
summary_rows = []
for h in HORIZONS:
    oh = outlook(hist, last_date, h, sfx)
    summary_rows.append({
        "Looking ahead": HZS_LONG[sfx][h],
        "Outlook": "No pattern today" if oh["status"] == "unknown" else "From past cases",
        "Went up %": None if pd.isna(oh["bullish"]) else round(oh["bullish"] * 100, 1),
        "Went down %": None if pd.isna(oh["bearish"]) else round(oh["bearish"] * 100, 1),
        "Past cases": oh["sample"],
        "How it is calculated": method_text(ticker, h, oh, tf),
    })
with tabs[-1]:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({
            "Field": ["Company", "Ticker", "Market", "Timeframe", "Latest bar", "Patterns found", "Disclaimer"],
            "Value": [company, ticker, MARKETS[market]["label"], tf.label, f"{last_date:%Y-%m-%d}",
                      ", ".join(pattern_name(p) for p in firing) or "None", DISCLAIMER],
        }).to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name="What happened before", index=False)
        checked.to_excel(xw, sheet_name="Patterns checked", index=False)
        evidence.to_excel(xw, sheet_name="Past cases", index=False)
    st.markdown("Download everything behind the numbers on this page: what happened before at every time horizon, "
                "how it is calculated, every pattern checked, and every past case, so you can check it yourself.")
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button(
        "⬇ Excel (all evidence)", buf.getvalue(),
        file_name=f"{ticker}_{tf.key}_pattern_evidence_{last_date:%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
    )
    d2.download_button(
        "⬇ CSV (past cases)", evidence.to_csv(index=False).encode(),
        file_name=f"{ticker}_{tf.key}_past_cases_{last_date:%Y%m%d}.csv", mime="text/csv", width="stretch",
    )

footer()
