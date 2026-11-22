"""The "All NSE stocks today" view: every stock that showed a daily pattern on the latest trading day."""

from __future__ import annotations

import html
import json
import os
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from pattern_history.catalog import RULE_NAME
from pattern_history.market_scan import VOLUME_SHOCK

REPO = Path(__file__).resolve().parent.parent
RAW = "https://raw.githubusercontent.com/b2b2ctechnologyservices/thepatternhistory/scan-data/"
SCAN_URL = os.getenv("SCAN_DATA_URL", RAW)


def _fetch(name: str) -> str:
    local = REPO / "scan" / "out" / name
    if local.exists():  # running locally after `python scan/run_india.py`
        return local.read_text()
    with urllib.request.urlopen(SCAN_URL + name, timeout=20) as r:
        return r.read().decode()


@st.cache_data(ttl=1800, show_spinner=False)
def load_scan() -> tuple[pd.DataFrame, dict]:
    meta = json.loads(_fetch("india_meta.json"))
    df = pd.read_csv(StringIO(_fetch("india_latest.csv")), keep_default_na=False, na_values=[""])
    df["pattern"] = df["pattern"].fillna("")
    return df, meta


def _size(rank: int) -> str:
    # SEBI's grouping by market value: top 100 large, next 150 mid, the rest small.
    return "Large" if rank <= 100 else "Mid" if rank <= 250 else "Small"


def render(esc, tip, footer, pick_stock) -> None:
    try:
        df, meta = load_scan()
    except Exception:
        st.info("Today's list of all NSE stocks isn't ready yet. It's made once a day after the market closes. "
                "You can still look up one stock with the search box above.")
        footer()
        return

    day = pd.Timestamp(meta["trading_day"])
    rank = dict(zip(df["symbol"], df["size_rank"]))  # rank in the whole NSE market, from the scan
    df["Size"] = [_size(rank[s]) for s in df["symbol"]]
    pats = df[df["pattern"] != ""]
    n_pat, n_vol = pats["symbol"].nunique(), df.loc[df["volume_x"] >= VOLUME_SHOCK, "symbol"].nunique()
    n_ok = pats.loc[pats["meets_rules"].astype(str) == "True", "symbol"].nunique()

    st.markdown(
        f'<div class="eyebrow"><b>All NSE stocks</b> {day:%a %d %b %Y} · {meta["stocks_current"]:,} stocks checked '
        + tip("Once a day, after the NSE closes, we check the latest daily price bar of every NSE stock Yahoo Finance "
              "lists for the same patterns as the single-stock page. Stocks with no trade that day are left out.")
        + f'</div><div class="headline">{n_pat:,} stocks showed a daily pattern on {day:%d %b}.</div>'
        f'<div class="sub">{n_ok:,} of them also meet the rule checklist, and {n_vol:,} stocks traded at least '
        f'{VOLUME_SHOCK:g}× their usual volume. Tap a row to see what happened every time before. This is a list of '
        'what the patterns found, not a list of stocks to buy or sell.</div>', unsafe_allow_html=True)
    st.write("")

    with st.container(key="mfilters"):
        f1, f2, f3, f4 = st.columns([1.3, 1.3, 1, 1.4], vertical_alignment="bottom")
        way = f1.segmented_control("Pattern points", ["All", "Up", "Down"], default="All", key="m_way") or "All"
        size = f2.segmented_control("Company size", ["All", "Large", "Mid", "Small"], default="All",
                                    key="m_size") or "All"
        only_vol = f3.toggle("Volume shockers only", key="m_vol",
                             help=f"Today's volume is at least {VOLUME_SHOCK:g}× the average of the 20 trading days "
                                  "before. Unusual volume often comes with news or big buyers/sellers.")
        only_ok = f4.toggle("Meets the rule checklist", value=True, key="m_ok",
                            help="Only patterns at a meaningful price level, with at least 2 supporting factors, "
                                 "and not fighting the bigger (weekly) trend.")

    view = df.copy()
    if not only_vol:
        view = view[view["pattern"] != ""]
    else:
        view = view[view["volume_x"] >= VOLUME_SHOCK]
    if way != "All":
        view = view[view["direction"] == ("LONG" if way == "Up" else "SHORT")]
    if size != "All":
        view = view[view["Size"] == size]
    if only_ok:
        view = view[view["meets_rules"].astype(str) == "True"]
    view = view.assign(_r=view["symbol"].map(rank)).sort_values(["_r", "pattern"])

    table = pd.DataFrame({
        "Stock": view["symbol"].str.replace(".NS", "", regex=False),
        "Company": view["name"],
        "Size": view["Size"],
        "Pattern": [RULE_NAME.get(r, "—") for r in view["rule"].fillna("")],
        "Points": view["direction"].map({"LONG": "↑ up", "SHORT": "↓ down"}).fillna("—"),
        "Checklist": ["—" if not p else ("✓ meets" if str(m) == "True" else "✕ misses")
                      for p, m in zip(view["pattern"], view["meets_rules"])],
        "Volume vs usual": view["volume_x"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}×"),
        "Price ₹": view["close"],
        "Day change": view["change_pct"].map(lambda x: "" if pd.isna(x) else f"{x:+.1f}%"),
    }).reset_index(drop=True)

    st.caption(f"{table['Stock'].nunique():,} stocks · {len(table):,} rows (a stock can show more than one pattern)")
    symbols = list(view["symbol"])

    def _open():
        rows = st.session_state["m_table"].selection.rows
        if rows:
            pick_stock(symbols[rows[0]].replace(".NS", ""))

    st.dataframe(table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 560),
                 on_select=_open, selection_mode="single-row", key="m_table",
                 column_config={"Price ₹": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("⬇ Download this list (CSV)", table.to_csv(index=False).encode(),
                       file_name=f"nse_patterns_{day:%Y%m%d}.csv", mime="text/csv")
    st.caption(f"Made {html.escape(meta['made_at_utc'])} from Yahoo Finance prices (adjusted). Daily patterns only. "
               "Past patterns do not guarantee future results.")
    footer()
