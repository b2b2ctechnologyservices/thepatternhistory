"""
Daily scan of every NSE stock. Writes scan/out/india_latest.csv and scan/out/india_meta.json.

    python scan/run_india.py            # all NSE stocks (~2,500)
    python scan/run_india.py --limit 50 # biggest 50 only, for a quick test

The GitHub workflow .github/workflows/india-scan.yml runs this after the NSE
close and publishes the two files to the `scan-data` branch, which the site reads.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pattern_history.market_scan import download_many, nse_universe, scan_one

OUT = Path(__file__).resolve().parent / "out"


def _scan(args):
    symbol, df = args
    return scan_one(symbol, df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    t0 = time.time()
    uni = nse_universe()
    if a.limit:
        uni = uni.head(a.limit)
    print(f"{len(uni)} NSE stocks", flush=True)
    names = dict(zip(uni["symbol"], uni["name"]))
    caps = dict(zip(uni["symbol"], uni["market_cap"]))

    rows, got, latest = [], 0, {}
    with ProcessPoolExecutor(a.workers) as pool:
        futures = []
        for symbol, df in download_many(list(uni["symbol"])):
            got += 1
            latest[symbol] = df.index[-1]
            futures.append(pool.submit(_scan, (symbol, df)))
        for f in futures:
            try:
                rows += f.result()
            except Exception as e:
                print("scan failed:", e, flush=True)

    # The market's latest trading day = the most common last date; stale stocks (suspended, delisted) drop out.
    day = pd.Series(latest).mode().iloc[0] if latest else None
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[out["date"] == day.strftime("%Y-%m-%d")]
        out.insert(1, "name", out["symbol"].map(names))
        out.insert(2, "market_cap", out["symbol"].map(caps))
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "india_latest.csv", index=False)
    meta = {"market": "India", "trading_day": day.strftime("%Y-%m-%d") if day is not None else None,
            "stocks_listed": int(len(uni)), "stocks_with_prices": got,
            "stocks_current": int(sum(1 for d in latest.values() if d == day)),
            "with_pattern": int(out.loc[out["pattern"] != "", "symbol"].nunique()) if not out.empty else 0,
            "volume_shockers": int(out.loc[out["volume_x"] >= 2, "symbol"].nunique()) if not out.empty else 0,
            "made_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "seconds": round(time.time() - t0)}
    (OUT / "india_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
