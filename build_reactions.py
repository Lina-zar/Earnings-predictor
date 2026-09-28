#!/usr/bin/env python3
"""
build_reactions.py - the stock's reaction to every earnings report (Part 3).

Same definitions as the database view v_reaction:
  day 0     = the report date for reports before the open (or during/unknown time),
              the next trading day for reports after the close
  car_0_1   = stock return minus S&P 500 (SPY) return, from the close before day 0
              to the close of day +1  (2-day reaction)
  car_0_2   = same, to the close of day +2  (3-day reaction)
Uses adjusted closes (dividends and splits included).

These are OUTCOMES, never model inputs - they are only used to test whether the model's
risk scores line up with how the stock actually moved.

Input : data/processed/earnings_events.csv, data/raw/yfinance_prices/
Output: data/processed/reactions.csv
Run   : python build_reactions.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "yfinance_prices"
PROC = ROOT / "data" / "processed"


def adj_close(ticker):
    files = sorted((RAW / str(ticker)).glob("*.csv"))
    if not files:
        return None
    p = pd.read_csv(files[-1])
    p["date"] = pd.to_datetime(p.iloc[:, 0]).dt.tz_localize(None).dt.normalize()
    return p.dropna(subset=["Adj Close"]).drop_duplicates("date").set_index("date")["Adj Close"].sort_index()


def main():
    ev = pd.read_csv(PROC / "earnings_events.csv", parse_dates=["report_date"])
    spy = adj_close("SPY")
    days = spy.index.values
    rows = []
    for ticker, g in ev.groupby("ticker_used"):
        px = adj_close(ticker)
        if px is None:
            continue
        px = px.reindex(spy.index)                       # align on NYSE trading days
        for _, e in g.iterrows():
            d = np.datetime64(e["report_date"])
            side = "right" if e["report_time"] == "AMC" else "left"
            i0 = np.searchsorted(days, d, side=side)     # index of day 0
            if i0 < 1 or i0 + 2 >= len(days):
                continue
            b, s = px.iloc[i0 - 1], spy.iloc[i0 - 1]
            r = {"company_key": e["company_key"], "report_date": e["report_date"].date(),
                 "day0": pd.Timestamp(days[i0]).date()}
            for k, col in ((1, "car_0_1"), (2, "car_0_2")):
                a, m = px.iloc[i0 + k], spy.iloc[i0 + k]
                r[col] = (a / b - 1) - (m / s - 1) if pd.notna(a) and pd.notna(b) else np.nan
            rows.append(r)
    out = pd.DataFrame(rows).dropna(subset=["car_0_1"])
    out.to_csv(PROC / "reactions.csv", index=False)
    print(f"Saved {len(out):,} reactions to data/processed/reactions.csv")


if __name__ == "__main__":
    main()
