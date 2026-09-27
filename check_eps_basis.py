#!/usr/bin/env python3
"""
check_eps_basis.py - do Alpha Vantage and yfinance report adjusted ("Street") EPS or GAAP EPS?

Test 1: Microsoft's quarter ending Dec 2017 (fiscal Q2 2018). The US tax reform charge made
        GAAP EPS a loss of about -$0.82, while adjusted EPS was about $0.96.
        ~ $0.96 -> the source uses adjusted EPS.   ~ -$0.82 -> it uses GAAP.
Test 2: For every quarter both sources have, how often do they disagree by more than $0.02?

Reads only the files probe_sources.py already saved under data/raw/probe/ -
no downloads, no API calls. Run it from the Earnings_predictor folder:
    python check_eps_basis.py
"""

import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw" / "probe"
TICKERS = ["AAPL", "MSFT", "WMT", "NVDA"]
TOLERANCE = 0.02


def latest(folder, ext):
    files = sorted((RAW / folder).glob(f"*.{ext}"))
    return files[-1] if files else None


def load_av(ticker):
    f = latest(f"alphavantage_earnings/{ticker}", "json")
    if not f:
        return None
    df = pd.DataFrame(json.loads(f.read_text()).get("quarterlyEarnings", []))
    df = df.replace({"None": None})
    df["report_date"] = pd.to_datetime(df["reportedDate"])
    df["av_actual"] = pd.to_numeric(df["reportedEPS"], errors="coerce")
    df["av_estimate"] = pd.to_numeric(df["estimatedEPS"], errors="coerce")
    return df[["fiscalDateEnding", "report_date", "av_estimate", "av_actual"]]


def load_yf(ticker):
    f = latest(f"yfinance_earnings/{ticker}", "csv")
    if not f:
        return None
    df = pd.read_csv(f)
    ts = pd.to_datetime(df.iloc[:, 0], utc=True).dt.tz_convert("America/New_York")
    df["report_date"] = pd.to_datetime(ts.dt.date)
    df["yf_actual"] = pd.to_numeric(df["Reported EPS"], errors="coerce")
    df["yf_estimate"] = pd.to_numeric(df["EPS Estimate"], errors="coerce")
    return df[["report_date", "yf_estimate", "yf_actual"]].dropna(subset=["yf_actual"])


def verdict(value):
    if pd.isna(value):
        return "missing"
    if value > 0.5:
        return "ADJUSTED (matches ~$0.96)"
    if value < 0:
        return "GAAP (matches ~-$0.82) - problem!"
    return "unclear - check by hand"


def test_msft():
    print("TEST 1  Microsoft, quarter ending 2017-12-31 (GAAP ~ -$0.82, adjusted ~ $0.96)")
    av = load_av("MSFT")
    if av is None:
        print("  Alpha Vantage: no saved MSFT file - run probe_sources.py first")
    else:
        row = av[av["fiscalDateEnding"] == "2017-12-31"]
        if row.empty:
            print("  Alpha Vantage: quarter not found")
        else:
            r = row.iloc[0]
            print(f"  Alpha Vantage: reported {r.av_actual}, estimate {r.av_estimate}, "
                  f"reported on {r.report_date.date()}  ->  {verdict(r.av_actual)}")
    yf = load_yf("MSFT")
    if yf is None:
        print("  yfinance:      no saved MSFT file - run probe_sources.py first")
    else:
        row = yf[yf["report_date"].between("2018-01-15", "2018-02-28")]
        if row.empty:
            print("  yfinance:      quarter not found")
        else:
            r = row.iloc[0]
            print(f"  yfinance:      reported {r.yf_actual}, estimate {r.yf_estimate}, "
                  f"reported on {r.report_date.date()}  ->  {verdict(r.yf_actual)}")


def test_agreement():
    print(f"\nTEST 2  Alpha Vantage vs yfinance, all quarters 2018-2025 (flag if > ${TOLERANCE:.2f} apart)")
    flagged = []
    for t in TICKERS:
        av, yf = load_av(t), load_yf(t)
        if av is None or yf is None:
            print(f"  {t:5} skipped - missing saved file")
            continue
        av = av[av["report_date"].between("2018-01-01", "2025-12-31")].sort_values("report_date")
        yf = yf.sort_values("report_date")
        av = av.assign(report_date=av["report_date"].astype("datetime64[ns]"))
        yf = yf.assign(report_date=yf["report_date"].astype("datetime64[ns]"))
        # match each quarter by report date, allowing the two sources to differ by up to 3 days
        m = pd.merge_asof(av, yf, on="report_date", direction="nearest",
                          tolerance=pd.Timedelta(days=3))
        matched = m.dropna(subset=["yf_actual"])
        m_act = (matched["av_actual"] - matched["yf_actual"]).abs() > TOLERANCE
        m_est = (matched["av_estimate"] - matched["yf_estimate"]).abs() > TOLERANCE
        print(f"  {t:5} {len(matched)}/{len(av)} quarters matched by date; "
              f"actual differs in {int(m_act.sum())}, estimate differs in {int(m_est.sum())}")
        bad = matched[m_act | m_est].assign(ticker=t)
        flagged.append(bad)
    flagged = [f for f in flagged if not f.empty]
    if flagged:
        out = pd.concat(flagged)[["ticker", "fiscalDateEnding", "report_date",
                                  "av_estimate", "yf_estimate", "av_actual", "yf_actual"]]
        print("\n  Quarters where the sources disagree (look these up by hand):")
        print(out.to_string(index=False))
    else:
        print("\n  No disagreements above the tolerance.")


if __name__ == "__main__":
    test_msft()
    test_agreement()
