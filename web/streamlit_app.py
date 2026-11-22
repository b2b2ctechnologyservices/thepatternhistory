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
from pattern_history.factors import is_independent, label as factor_label, split as split_factors
from pattern_history.plan import MIN_CONFIDENCE, MIN_R, todays_plans
from pattern_history.history import HORIZONS, directional, outlook, replay, rule_stats

SMALL_SAMPLE = 10
DISCLAIMER = (
    "Historical statistics for education only. Not investment advice, and not a recommendation to buy or "
    "sell any security. Hypothetical trades show how a fixed set of rules would behave; nobody is told to "
    "place them. Past patterns do not guarantee future results."
)
HZ = {1: "1 day", 5: "1 week", 20: "1 month", 60: "3 months"}
HZ_LONG = {1: "the next trading day", 5: "1 week (5 trading days)", 20: "1 month (20 trading days)",
           60: "3 months (60 trading days)"}
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
def get_history(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_daily(ticker, years=25)
    return prices, replay(prices)


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


def points(direction: str) -> str:
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


def method_text(ticker: str, horizon: int, o: dict) -> str:
    if o["status"] == "unknown":
        return (f"None of our 10 patterns showed up on {ticker}'s latest daily price bar, so there is nothing "
                "to look up in history.")
    return (
        f"Step 1: today these pattern(s) showed up on {ticker}: {', '.join(friendly(p) for p in o['patterns'])}. "
        f"Step 2: we went back through ~25 years of {ticker}'s daily prices and found every past trading day "
        f"when the same pattern showed up, using only the prices known up to that day "
        f"({o['sample']} days; a day where two of them showed up counts once). "
        f"Step 3: for each of those days we compared the closing price with the close {HZ_LONG[horizon]} later. "
        "'Went up' = share of days where the price was higher; 'Went down' = share where it was lower. "
        "Today isn't counted because we don't know its outcome yet. Past patterns do not guarantee future results."
    )



# ── Look and feel: Indigo ────────────────────────────────────────────────────
LOGO_MARK = ('<svg viewBox="0 0 40 40" width="{s}" height="{s}"><path d="M20 4 A16 16 0 1 0 36 20" fill="none" '
             'stroke="var(--acc)" stroke-width="3.5" stroke-linecap="round"/><path d="M36 13 L36 20 L29 20" '
             'fill="none" stroke="var(--acc)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>'
             '<circle cx="14" cy="25" r="3" fill="var(--acc2)"/><circle cx="20" cy="20" r="3" fill="var(--acc2)"/>'
             '<circle cx="26" cy="14" r="3" fill="var(--acc2)"/></svg>')
THEMES = {  # accent, second accent, soft tint, page wash
    "US": ("#6D5EF5", "#FF7A59", "#F0EEFF", "#E9E6FF", "#F7F6FF"),
    "India": ("#E8590C", "#6D5EF5", "#FFF0E6", "#FFE8D9", "#FFF9F5"),
}


def tip(text: str) -> str:
    """A small ⓘ that opens on tap or click (works on phones, no hover needed)."""
    return f'<details class="tip"><summary>i</summary><div>{text}</div></details>'


def donut(up: float, down: float, size: int = 190) -> str:
    r = size / 2 - 14
    c = 2 * 3.14159 * r
    big, lab, col = (up, "went up", "var(--up)") if up >= down else (down, "went down", "var(--down)")
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="display:block;margin:auto">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="var(--tint)" stroke-width="18"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="var(--up)" stroke-width="18" '
            f'stroke-dasharray="{c*up:.1f} {c:.1f}" transform="rotate(-90 {size/2} {size/2})"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="var(--down)" stroke-width="18" '
            f'stroke-dasharray="{c*down:.1f} {c:.1f}" stroke-dashoffset="{-c*(up+0.01):.1f}" '
            f'transform="rotate(-90 {size/2} {size/2})"/>'
            f'<text x="50%" y="48%" text-anchor="middle" font-size="{size*0.2:.0f}" font-weight="800" fill="{col}">'
            f'{big:.0%}</text><text x="50%" y="63%" text-anchor="middle" font-size="{size*0.075:.0f}" '
            f'fill="var(--muted)">{lab}</text></svg>')


def ladder(pl: dict) -> str:
    short = pl["direction"] == "SHORT"
    rows = [("Stop", pl["stop"], "var(--down)"), ("Entry", pl["entry"], "var(--acc)"),
            ("Target", pl["target"], "var(--up)")]
    if not short:
        rows = rows[::-1]
    ys = [16, 62, 118] if short else [16, 72, 118]
    y_entry = ys[1]
    out = ""
    for (lab, val, col), y in zip(rows, ys):
        dash = "0" if lab == "Entry" else "5 4"
        out += (f'<line x1="70" x2="196" y1="{y}" y2="{y}" stroke="{col}" stroke-width="2.5" stroke-dasharray="{dash}"/>'
                f'<text x="0" y="{y+4}" font-size="12" fill="var(--muted)">{lab}</text>'
                f'<text x="204" y="{y+5}" font-size="15" font-weight="700" fill="var(--text)">{val:,.2f}</text>')
    y_stop = ys[0] if short else ys[2]
    y_tgt = ys[2] if short else ys[0]
    out += (f'<rect x="120" y="{min(y_stop, y_entry)}" width="26" height="{abs(y_entry - y_stop)}" fill="var(--down)" opacity=".16"/>'
            f'<rect x="120" y="{min(y_tgt, y_entry)}" width="26" height="{abs(y_entry - y_tgt)}" fill="var(--up)" opacity=".18"/>'
            f'<path d="M133 {y_entry + (4 if short else -4)} L133 {y_tgt + (-6 if short else 6)}" stroke="var(--text)" '
            'stroke-width="1.5" marker-end="url(#ar)"/>')
    return ('<svg viewBox="0 0 270 132" width="100%" style="max-width:320px;display:block"><defs><marker id="ar" '
            'markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" '
            f'fill="var(--text)"/></marker></defs>{out}</svg>')


def conf_dots(n_ind: int, conf: int) -> str:
    return "".join(f'<span class="cd {"on" if i < n_ind else ("own" if i < conf else "")}"></span>' for i in range(5))


st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--acc:#6D5EF5;--acc2:#FF7A59;--tint:#F0EEFF;--wash:#E9E6FF;--text:#1E1B4B;--muted:#6B6A8A;
      --up:#12B886;--down:#FF6B6B;--line:#E4E1FB;--card:#FFFFFF;--page:#F7F6FF}
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded){font-family:Inter,sans-serif!important}
.stApp{background:linear-gradient(180deg,var(--wash) 0,var(--page) 380px,var(--page) 100%)!important}
.block-container{padding-top:1rem!important;padding-bottom:1rem!important;max-width:1140px}
header[data-testid="stHeader"]{display:none}
.st-key-topbar{padding-bottom:.4rem}
.st-key-topbar label p{font-size:.72rem!important;font-weight:600;color:var(--muted);letter-spacing:.06em;text-transform:uppercase}
.logo{display:flex;align-items:center;gap:8px;font-size:1.45rem;letter-spacing:-.01em;padding-top:14px}
.logo b{color:var(--text);font-weight:700} .logo span{color:var(--acc);font-weight:400}
button[data-variant="segmented_control"]{border-radius:999px!important;border:0!important;background:#fff!important;
  font-weight:600!important;color:var(--muted)!important}
button[data-variant="segmented_control"][aria-checked="true"]{background:var(--acc)!important;color:#fff!important}
button[data-variant="segmented_control"][aria-checked="true"] *{color:#fff!important}
[data-testid="stBaseButton-primaryFormSubmit"],[data-testid="stBaseButton-primary"]{background:var(--acc)!important;
  border:0!important;border-radius:999px!important;font-weight:700!important;height:2.75rem}
[data-testid="stTextInput"] input{border-radius:999px!important;height:2.75rem;font-weight:600;background:#fff!important}
[data-testid="stTextInput"] div[data-baseweb="input"]{border-radius:999px!important;border-color:var(--line)!important;background:#fff!important}
.st-key-chartcard{background:#fff;border-radius:22px;padding:20px 22px!important;box-shadow:0 10px 30px rgba(109,94,245,.10)}
.start{color:var(--muted);font-size:.95rem;padding:30px 0 0}
.chipok{display:inline-flex;align-items:center;gap:8px;font-size:.85rem;font-weight:600;color:var(--up);background:#fff;
  padding:7px 14px;border-radius:999px;box-shadow:0 2px 10px rgba(109,94,245,.08)}
.busy{display:flex;align-items:center;gap:12px;background:#fff;border-radius:18px;padding:14px 18px;box-shadow:0 10px 30px rgba(109,94,245,.10)}
.spin{width:22px;height:22px;border-radius:50%;border:3px solid var(--tint);border-top-color:var(--acc);animation:sp .8s linear infinite;flex:none}
@keyframes sp{to{transform:rotate(360deg)}}
.eyebrow{font-size:.9rem;color:var(--muted);display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:18px}
.eyebrow b{color:var(--text)}
.headline{font-size:2.9rem;line-height:1.08;font-weight:800;letter-spacing:-.025em;color:var(--text);margin:10px 0 10px}
.headline em{font-style:normal}
.sub{color:var(--muted);font-size:1.02rem}
.card{background:var(--card);border-radius:22px;padding:20px 22px;box-shadow:0 10px 30px rgba(109,94,245,.10);height:100%}
.lab{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:10px;
  display:flex;align-items:center;gap:6px}
.big{font-size:1.25rem;font-weight:800;color:var(--text);display:flex;gap:8px;align-items:center;line-height:1.25}
.small{color:var(--muted);font-size:.88rem;margin-top:6px;line-height:1.4}
.pillv{display:inline-flex;align-items:center;gap:6px;padding:7px 12px;border-radius:10px;font-weight:700;font-size:.85rem}
.pillv.pass{background:#FFE9E9;color:#E03131} .pillv.take{background:#E6FCF5;color:#0CA678} .pillv.none{background:var(--tint);color:var(--text)}
.pillv.hist{background:#fff;color:var(--text);box-shadow:0 2px 10px rgba(109,94,245,.08)}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{font-size:.8rem;font-weight:600;padding:6px 11px;border-radius:999px;background:var(--tint);color:var(--text)}
details.tip{display:inline-block;position:relative}
details.tip summary{list-style:none;cursor:pointer;width:17px;height:17px;border-radius:50%;border:1.5px solid var(--muted);
  color:var(--muted);font:700 10px Inter;display:flex;align-items:center;justify-content:center;text-transform:none;letter-spacing:0}
details.tip summary::-webkit-details-marker{display:none}
details.tip[open] summary{background:var(--acc);border-color:var(--acc);color:#fff}
details.tip div{position:absolute;z-index:20;top:24px;left:-10px;width:min(300px,78vw);background:var(--text);color:#fff;
  font-size:.82rem;font-weight:400;line-height:1.45;padding:12px 14px;border-radius:12px;text-transform:none;letter-spacing:0;
  box-shadow:0 12px 30px rgba(30,27,75,.25)}
.cd{display:inline-block;width:13px;height:13px;border-radius:50%;margin-right:4px;border:2px solid var(--acc)}
.cd.on{background:var(--acc)} .cd.own{background:var(--tint)}
.num{font-size:1.35rem;font-weight:800;color:var(--text)}
.legend{display:flex;gap:14px;font-size:.75rem;color:var(--muted);margin-top:4px;flex-wrap:wrap}
.legend i{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
.tph-conf h4{color:#C9C3F7;font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;margin:0 0 6px;padding:0}
.tph-conf ul{list-style:none;margin:6px 0;padding:0} .tph-conf li{margin:3px 0}
.tph-conf li span:last-child{color:#C9C3F7;font-size:.75rem;margin-left:4px}
.tph-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.tph-dot.ind{background:#12B886} .tph-dot.own{background:#8B86C9}
.conf .k{color:#C9C3F7;font-size:.75rem;margin-left:4px}
.foot{text-align:center;font-size:.8rem;color:var(--muted);padding:22px 8px 6px;display:flex;gap:6px;justify-content:center;align-items:center}
[data-testid="stExpander"]{background:#fff;border:0!important;border-radius:18px!important;box-shadow:0 10px 30px rgba(109,94,245,.08)}
[data-testid="stExpander"] summary p{font-weight:600;color:var(--text)}
[data-testid="stPopover"] button{border-radius:999px!important}
.st-key-pick label p{font-size:.7rem!important;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:700}
@media (max-width:640px){.headline{font-size:1.9rem}.logo{font-size:1.25rem;padding-top:0}.card{padding:16px}
  .st-key-hero [data-testid="stHorizontalBlock"]{gap:.8rem}}
</style>""", unsafe_allow_html=True)


def footer() -> None:
    st.markdown('<div class="foot">Education only · past patterns don\'t guarantee the future · not investment advice '
                + tip(esc(DISCLAIMER)) + '</div>', unsafe_allow_html=True)


# ── Top bar: logo, country, then search ─────────────────────────────────────
with st.container(key="topbar"):
    c_logo, c_mkt, c_search = st.columns([1.1, 0.8, 2.0], vertical_alignment="bottom")
    c_logo.markdown(f'<div class="logo">{LOGO_MARK.format(s=34)}<b>pattern</b><span>history</span></div>',
                    unsafe_allow_html=True)
    market = c_mkt.segmented_control("Market", list(MARKETS), default=None,
                                     format_func=lambda m: "US" if m == "US" else "India", key="market")
    query, submitted = "", False
    if market:
        with c_search.form("scan", border=False):
            f1, f2 = st.columns([4, 1], vertical_alignment="bottom", gap="small")
            query = f1.text_input("Stock", placeholder="Ticker or company, e.g. MU" if market == "US"
                                  else "NSE symbol or company, e.g. SBIN")
            submitted = f2.form_submit_button("Scan", type="primary", width="stretch")
    else:
        c_search.markdown('<div class="start">← Pick a market to start</div>', unsafe_allow_html=True)

acc, acc2, tint, wash, page = THEMES[market or "US"]
st.markdown(f"<style>:root{{--acc:{acc};--acc2:{acc2};--tint:{tint};--wash:{wash};--page:{page}}}</style>",
            unsafe_allow_html=True)

if submitted and query.strip():
    st.session_state["query"] = (query.strip(), market)

q, q_market = st.session_state.get("query", ("", market))
if not market or not q or q_market != market:
    st.markdown(
        '<div class="eyebrow">25 years of prices · 10 chart patterns · US &amp; India</div>'
        '<div class="headline">What happened every time<br>this pattern showed up before?</div>'
        '<div class="sub">Pick a market, type a stock, see the history. '
        + tip("We check a stock for 10 well-known chart patterns on the latest day, then look up every past day the "
              "same pattern showed up in ~25 years of that stock's prices, and show what the price did next. US tickers "
              "work as-is (AAPL, BRK-B). For India type the NSE symbol (SBIN) or the company name.")
        + '</div>', unsafe_allow_html=True)
    footer()
    st.stop()

# ── Scan ────────────────────────────────────────────────────────────────────
strip = st.empty()


def busy(text: str) -> None:
    strip.markdown(f'<div class="busy"><span class="spin"></span><b>{esc(text)}</b>'
                   '<span style="color:var(--muted);font-size:.85rem">first scan of a stock takes ~30–60s</span></div>',
                   unsafe_allow_html=True)


busy(f"Finding '{q}'…")
candidates, best = get_candidates(q, market)
if not candidates:
    strip.error(f"Couldn't find '{q}' in {'US' if market == 'US' else 'India (NSE)'}. Try the ticker symbol.")
    footer()
    st.stop()
pick_key = f"pick::{market}::{q}"
choice = st.session_state.get(pick_key, best)
choice = choice if 0 <= choice < len(candidates) else best
ticker, company = candidates[choice]["symbol"], candidates[choice]["name"]

busy(f"Scanning {company} ({ticker})…")
try:
    prices, hist = get_history(ticker)
except ProviderBusy as e:
    strip.error(str(e)); footer(); st.stop()
except DataError as e:
    strip.error(str(e) if market == "US" else f"No price data found for {ticker} on NSE.")
    footer(); st.stop()
except Exception as e:
    strip.error(f"Couldn't load data right now. Please try again in a minute. ({e})")
    footer(); st.stop()

last_date = prices.index[-1]
last_close = float(prices["close"].iloc[-1])
currency = "₹" if market == "India" else "$"
context, plans = get_plans(ticker, prices, last_date)
plan_by_pattern = {p["pattern"]: p for p in plans}
today = hist[hist["date"] == last_date]
firing = list(dict.fromkeys(today["pattern"]))
rule_of = dict(zip(today["pattern"], today["rule"]))
dir_of = dict(zip(today["pattern"], today["direction"]))


def pattern_name(pattern: str) -> str:
    rule = rule_of.get(pattern)
    return RULE_NAME[rule] if rule else friendly(pattern)


with strip.container():
    a, b = st.columns([4, 1.2], vertical_alignment="center") if len(candidates) > 1 else (st.container(), None)
    a.markdown(f'<span class="chipok">✓ Scanned {esc(ticker)} · {len(prices):,} days</span>', unsafe_allow_html=True)
    if b is not None:
        with b.popover("Not this company?", width="stretch"):
            st.radio("Other matches", range(len(candidates)), index=choice, key=pick_key,
                     format_func=lambda i: f"{candidates[i]['symbol']} — {candidates[i]['name']}")

# ── Hero: one sentence + donut ──────────────────────────────────────────────
horizon = st.session_state.get("horizon") or 20
o = outlook(hist, last_date, horizon)
n = o["sample"]
trend = str(context["trend"])
arrow = '<span style="color:var(--down)">▼</span>' if "DOWN" in trend else (
    '<span style="color:var(--up)">▲</span>' if "UP" in trend else "")
with st.container(key="hero"):
    h_left, h_right = st.columns([1.55, 1], gap="large", vertical_alignment="center")
    with h_left:
        if o["status"] == "unknown":
            line = f"No pattern on {esc(ticker)} today."
            sub = "Most days have none. History appears when a pattern does."
        elif not n:
            line = f"This pattern is new for {esc(ticker)}."
            sub = "It has never shown up before with a known outcome, so there's no history yet."
        else:
            upish = o["bullish"] >= o["bearish"]
            word, val, col = ("rose", o["bullish"], "var(--up)") if upish else ("fell", o["bearish"], "var(--down)")
            line = (f'After this pattern, {esc(ticker)} {word} <em style="color:{col}">{val:.0%}</em> of the time.')
            dirs = {dir_of[p] for p in o["patterns"] if p in dir_of}
            way = ("points up" if dirs == {"LONG"} else "points down" if dirs == {"SHORT"} else "is mixed")
            sub = f"{n} past cases · checked {HZ[horizon]} later · today's pattern {way}"
        st.markdown(f'<div class="eyebrow"><b>{esc(ticker)}</b>{esc(company)} · {currency}{last_close:,.2f} {arrow}</div>'
                    f'<div class="headline">{line}</div><div class="sub">{sub}</div>', unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1.3], vertical_alignment="center")
        r1.segmented_control("Look ahead", HORIZONS, default=20, format_func=lambda h: HZ[h].replace(" ", "")
                             .replace("1day", "1D").replace("1week", "1W").replace("1month", "1M").replace("3months", "3M"),
                             key="horizon", label_visibility="collapsed")
        if n and o["status"] != "unknown":
            dirs = {dir_of[p] for p in o["patterns"] if p in dir_of}
            if len(dirs) == 1:
                agree = o["bullish"] if dirs == {"LONG"} else o["bearish"]
                verdict = ("History mostly agreed with it" if agree >= 0.55 else
                           "History mostly went against it" if agree <= 0.45 else "History is a coin flip here")
                k_up, k_dn = round(o["bullish"] * n), round(o["bearish"] * n)
                r2.markdown(f'<div class="pillv hist">{verdict} ' + tip(
                    f"In {k_up} of {n} past cases {esc(ticker)}'s price was higher {HZ[horizon]} later, and in {k_dn} it "
                    f"was lower. Today's pattern {('points up' if dirs == {'LONG'} else 'points down')}, so that is how "
                    "often history agreed with it.") + "</div>", unsafe_allow_html=True)
        if n and n < SMALL_SAMPLE:
            st.caption(f"Only {n} past cases: too few to read much into.")
    with h_right:
        if n and o["status"] != "unknown":
            st.markdown(f'<div class="card" style="padding:24px">{donut(o["bullish"], o["bearish"])}'
                        f'<div class="legend" style="justify-content:center;margin-top:10px">'
                        f'<span><i style="background:var(--up)"></i>up {o["bullish"]:.0%}</span>'
                        f'<span><i style="background:var(--down)"></i>down {o["bearish"]:.0%}</span>'
                        f'<span>{tip(esc(method_text(ticker, horizon, o)))}</span></div></div>',
                        unsafe_allow_html=True)

# ── Three cards ─────────────────────────────────────────────────────────────
st.write("")
k1, k2, k3 = st.columns(3)
if firing:
    p0 = firing[0]
    more = f" +{len(firing) - 1} more" if len(firing) > 1 else ""
    sign_n = len(split_factors(p0, today[today["pattern"] == p0]["factors"].iloc[0])[1])
    arrow0 = '<span style="color:var(--up)">↑</span>' if dir_of[p0] == "LONG" else '<span style="color:var(--down)">↓</span>'
    k1.markdown(f'<div class="card"><div class="lab">Today\'s pattern {tip(esc(RULE_DESC[rule_of[p0]]))}</div>'
                f'<div class="big">{arrow0} {esc(pattern_name(p0))}</div>'
                f'<div class="small">{"Points up" if dir_of[p0] == "LONG" else "Points down"} · {sign_n} supporting '
                f'sign{"s" if sign_n != 1 else ""}{esc(more)}</div></div>', unsafe_allow_html=True)
else:
    k1.markdown('<div class="card"><div class="lab">Today\'s pattern</div><div class="big">None today</div>'
                '<div class="small">None of the 10 patterns showed up on the latest day.</div></div>',
                unsafe_allow_html=True)
if plans:
    top = plans[0]
    cls, lab = ("take", "✓ Would act") if top["decision"] == "TAKE" else ("pass", "✕ Would pass")
    short = (f"Gain is {top['r']:.2f}× the risk; rules need {MIN_R}×" if "reward:risk" in top["reason"]
             else f"{top['confidence']} signs line up" if top["decision"] == "TAKE"
             else f"Needs {MIN_CONFIDENCE}+ signs" if top["confidence"] < MIN_CONFIDENCE else "No sensible exit prices")
    k2.markdown(f'<div class="card"><div class="lab">Would the rules act? '
                f'{tip("We test every pattern against one fixed set of trading rules. " + esc(plain_reason(top, currency)) + " This shows how those rules behave; it is not a recommendation.")}</div>'
                f'<span class="pillv {cls}">{lab}</span><div class="small">{esc(short)}</div></div>',
                unsafe_allow_html=True)
else:
    k2.markdown('<div class="card"><div class="lab">Would the rules act?</div><span class="pillv none">— No setup</span>'
                '<div class="small">No pattern, so nothing to act on.</div></div>', unsafe_allow_html=True)
tech, bow = str(context["technical"]), str(context["bow_tie"])
mood = []
if trend in TREND_TEXT:
    mood.append(f"<b>Trend {esc(trend.replace('_', ' ').lower())}:</b> {esc(TREND_TEXT[trend])}.")
if tech in POSITION_TEXT:
    mood.append(f"<b>{esc(tech.title())}:</b> {esc(POSITION_TEXT[tech])}.")
if bow in BOWTIE_TEXT:
    mood.append(f"<b>Bow tie:</b> {esc(BOWTIE_TEXT[bow])}.")
t_arrow = "↑" if "UP" in trend else "↓" if "DOWN" in trend else "→"
k3.markdown(f'<div class="card"><div class="lab">Market mood {tip("<br><br>".join(mood))}</div><div class="chips">'
            f'<span class="chip">Trend {t_arrow}</span><span class="chip">{esc(tech.title())} vs averages</span>'
            f'<span class="chip">{"No bow tie" if bow == "none" else "Bow tie " + esc(bow.lower())}</span></div></div>',
            unsafe_allow_html=True)

# ── Detail for one pattern: trade example + chart ───────────────────────────
pattern = None
if firing:
    st.write("")
    if len(firing) > 1:
        with st.container(key="pick"):
            pattern = st.segmented_control("Pattern", firing, default=firing[0], format_func=pattern_name,
                                           key="pattern") or firing[0]
    else:
        pattern = firing[0]
    occ_all = hist[hist["pattern"] == pattern].sort_values("date")
    direction = occ_all["direction"].iloc[0]
    past = occ_all[occ_all["date"] < last_date]
    n_past = len(past)
    d1, d2 = st.columns([1, 1.6])
    pl = plan_by_pattern.get(pattern)
    with d1:
        if pl:
            ind_n = len(split_factors(pattern, pl["factors"])[1])
            bet = "falls (a 'short')" if pl["direction"] == "SHORT" else "rises (a 'long')"
            risk, reward = abs(pl["entry"] - pl["stop"]), abs(pl["target"] - pl["entry"])
            st.markdown(
                f'<div class="card"><div class="lab">Trade example · not advice '
                + tip(f"To make the pattern easier to picture: what a trade could look like if someone followed these "
                      f"rules exactly, betting the price {bet}. <b>Entry</b> is where it would start, <b>Stop</b> where it "
                      "would get out to keep a loss small, <b>Target</b> where it would take its gain. An example for "
                      "learning, not a suggestion to trade.") + '</div>'
                + ladder(pl)
                + '<div style="display:flex;justify-content:space-between;gap:12px;margin-top:12px;flex-wrap:wrap">'
                f'<div><div class="lab" style="margin:0">Reward : risk '
                + tip(f"Possible gain ÷ possible loss: risking {currency}{risk:,.2f} per share to possibly gain "
                      f"{currency}{reward:,.2f}. The rules want {MIN_R}× or more.")
                + f'</div><div class="num">{pl["r"]:.2f}×</div></div>'
                f'<div><div class="lab" style="margin:0">Confidence '
                + tip(confidence_html(pattern, pl["factors"], pl["confidence"]))
                + f'</div><div style="margin-top:6px">{conf_dots(min(ind_n, pl["confidence"]), pl["confidence"])}</div>'
                f'<div class="small" style="margin-top:2px;font-size:.75rem">{pl["confidence"]}/5 · {ind_n} independent</div>'
                '</div></div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="lab">Trade example</div><div class="small">No trade levels for '
                        'this pattern today.</div></div>', unsafe_allow_html=True)
    with d2:
        with st.container(key="chartcard", border=False):
            st.markdown(f'<div class="lab" style="margin:0 0 4px">25 years of {esc(ticker)} · every time this pattern '
                        'showed up ' + tip("Grey line: the price, adjusted for splits and dividends. Each dot is a past "
                                           "day this pattern showed up. Green: over the next month the price moved the "
                                           "way the pattern pointed. Red: it moved the other way. Tap a dot for details.")
                        + '</div>', unsafe_allow_html=True)
            chart_px = prices[["close"]].reset_index().rename(columns={"ts": "date"})
            accent = THEMES[market][0]
            line = alt.Chart(chart_px).mark_area(
                line={"color": "#8B86C9", "strokeWidth": 1.4}, color=alt.Gradient(
                    gradient="linear", stops=[alt.GradientStop(color="#ECEAFE", offset=0),
                                              alt.GradientStop(color="#FFFFFF", offset=1)], x1=1, x2=1, y1=1, y2=0)
            ).encode(x=alt.X("date:T", title=None), y=alt.Y("close:Q", title=None, scale=alt.Scale(zero=False)))
            marks = occ_all[["date", "close", "ret_20d"]].copy()
            marks["After 1 month"] = ["Not known yet" if pd.isna(x) else ("Went its way" if x > 0 else "Went against")
                                      for x in directional(marks["ret_20d"], direction)]
            marks["Move"] = [pct(x) for x in marks["ret_20d"]]
            pts = alt.Chart(marks).mark_circle(size=55, stroke="white", strokeWidth=1.2, opacity=1).encode(
                x="date:T", y="close:Q",
                color=alt.Color("After 1 month:N", legend=None, scale=alt.Scale(
                    domain=["Went its way", "Went against", "Not known yet"], range=["#12B886", "#FF6B6B", accent])),
                tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("close:Q", format=",.2f", title="Price"),
                         alt.Tooltip("Move", title="1-month move"), "After 1 month"])
            st.altair_chart((line + pts).properties(height=230).configure_view(strokeWidth=0)
                            .configure_axis(grid=False, labelColor="#6B6A8A", domain=False, tickColor="#E4E1FB",
                                            labelFont="Inter"), width="stretch")
            st.markdown('<div class="legend"><span><i style="background:var(--up)"></i>went its way</span>'
                        '<span><i style="background:var(--down)"></i>went against</span>'
                        f'<span><i style="background:{accent}"></i>not known yet</span></div>', unsafe_allow_html=True)

# ── Fold-out details ────────────────────────────────────────────────────────
st.write("")
if pattern is not None and n_past:
    with st.expander(f"Results by time ahead · {pattern_name(pattern)}"):
        stats = rule_stats(past, direction)
        st.dataframe(pd.DataFrame({
            "Looking ahead": [HZ[s.horizon] for s in stats], "Cases": [s.sample for s in stats],
            "Pattern was right": [share(s.win_rate) for s in stats], "Average": [pct(s.average) for s in stats],
            "Typical": [pct(s.median) for s in stats], "Worst": [pct(s.worst) for s in stats],
            "Best": [pct(s.best) for s in stats]}), hide_index=True, width="stretch")
        st.caption("Measured in the pattern's direction: for a pattern that points down, a price drop shows as positive.")
    with st.expander(f"Every past case ({n_past})"):
        table = past.sort_values("date", ascending=False)
        res = directional(table["ret_20d"], direction)
        st.dataframe(pd.DataFrame({
            "Date": table["date"].dt.strftime("%Y-%m-%d"), "Price": table["close"].map(lambda v: f"{v:,.2f}"),
            **{f"After {HZ[h]}": table[f"ret_{h}d"].map(pct) for h in HORIZONS},
            "After 1 month": ["Not known yet" if pd.isna(x) else ("✅ Its way" if x > 0 else "❌ Against") for x in res],
        }), hide_index=True, width="stretch", height=min(38 + 35 * n_past, 380))

rows = []
for key, name, desc in RULES:
    fired_today = today[today["rule"] == key]
    past_n = int(((hist["rule"] == key) & (hist["date"] < last_date)).sum())
    status = "—" if fired_today.empty else " / ".join(
        "✅ up" if d == "LONG" else "✅ down" for d in dict.fromkeys(fired_today["direction"]))
    rows.append({"Pattern": name, "Today": status, "Seen in ~25 yrs": past_n, "What it looks for": desc})
checked = pd.DataFrame(rows)
with st.expander(f"All 10 patterns checked · {today['rule'].nunique()} today"):
    st.dataframe(checked, hide_index=True, width="stretch",
                 column_config={"What it looks for": st.column_config.TextColumn(width="large")})

evidence = (hist[hist["pattern"].isin(firing)] if firing else hist).copy()
evidence.insert(0, "ticker", ticker)
evidence["rule"] = evidence["rule"].map(RULE_NAME)
evidence["date"] = evidence["date"].dt.strftime("%Y-%m-%d")
with st.expander("Download Excel / CSV"):
    summary_rows = []
    for h in HORIZONS:
        oh = outlook(hist, last_date, h)
        summary_rows.append({
            "Looking ahead": HZ_LONG[h],
            "Went up %": None if pd.isna(oh["bullish"]) else round(oh["bullish"] * 100, 1),
            "Went down %": None if pd.isna(oh["bearish"]) else round(oh["bearish"] * 100, 1),
            "Past cases": oh["sample"], "How it is calculated": method_text(ticker, h, oh)})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Field": ["Company", "Ticker", "Market", "Latest day", "Patterns found today", "Disclaimer"],
                      "Value": [company, ticker, MARKETS[market]["label"], f"{last_date:%Y-%m-%d}",
                                ", ".join(pattern_name(p) for p in firing) or "None", DISCLAIMER]}
                     ).to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(summary_rows).to_excel(xw, sheet_name="What happened before", index=False)
        checked.to_excel(xw, sheet_name="Patterns checked", index=False)
        evidence.to_excel(xw, sheet_name="Past cases", index=False)
    e1, e2, _ = st.columns([1, 1, 2])
    e1.download_button("⬇ Excel", buf.getvalue(), file_name=f"{ticker}_pattern_evidence_{last_date:%Y%m%d}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    e2.download_button("⬇ CSV", evidence.to_csv(index=False).encode(),
                       file_name=f"{ticker}_past_cases_{last_date:%Y%m%d}.csv", mime="text/csv", width="stretch")

footer()
