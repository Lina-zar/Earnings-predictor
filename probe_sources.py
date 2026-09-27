#!/usr/bin/env python3
"""
probe_sources.py - the Part 1 "afternoon test" for the Earnings Predictor project.

One run answers five questions:
  Q1  Alpha Vantage EARNINGS: how far back does it go, and does it give report time?
  Q2  yfinance get_earnings_dates: 30+ quarters per ticker for 2018-2025?
  Q3  SEC companyfacts: can Q4 be derived (FY minus 9-month YTD), and do the pieces reconcile?
  Q4  What share of earnings events have unknown timing (before open / after close)?
  Q5  Do acquired or delisted companies still have prices and earnings data?

Setup (once):
  pip install yfinance pandas requests lxml
  Get a free Alpha Vantage key: https://www.alphavantage.co/support/#api-key

Run (Mac/Linux):
  export ALPHAVANTAGE_API_KEY=your_key
  export SEC_USER_AGENT="Your Name your@email.com"
  python probe_sources.py

Run (Windows PowerShell):
  $env:ALPHAVANTAGE_API_KEY="your_key"
  $env:SEC_USER_AGENT="Your Name your@email.com"
  python probe_sources.py

Cost: 7 Alpha Vantage calls (the free limit is 25 a day), about 30 SEC requests, 14 Yahoo requests.
Every raw response is saved under data/raw/probe/<source>/<ticker>/<date>.* and reused on
later runs, so rerunning the script does not spend your Alpha Vantage allowance again.
Any source that is missing a key or fails is skipped and reported; the rest still run.
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:  # the script still runs the other sources
    yf = None

# ----------------------------------------------------------------------------- settings
# AAPL, MSFT, WMT and NVDA all have fiscal years that are not the calendar year,
# which is exactly where quarterly data tends to go wrong.
ACTIVE = ["AAPL", "MSFT", "WMT", "NVDA"]

# Companies that were S&P 500 members in 2018 and later disappeared.
# They are no longer in the SEC's current ticker file, so their CIKs are listed here.
# (CIKs from memory - confirm at https://www.sec.gov/cgi-bin/browse-edgar?company=<name>)
DELISTED = {
    "TWTR": "0001418091",  # Twitter, taken private October 2022
    "XLNX": "0000743988",  # Xilinx, acquired by AMD February 2022
    "ATVI": "0000718877",  # Activision Blizzard, acquired by Microsoft October 2023
}

WINDOW_START, WINDOW_END = "2018-01-01", "2025-12-31"  # the project's study window
PULL_START = "2016-01-01"                               # extra history for lag features
EXPECTED_QUARTERS = 32                                  # 8 years x 4 quarters
PASS_QUARTERS = 30

AV_PAUSE = 15     # seconds after each fresh Alpha Vantage call
YF_PAUSE = 3      # seconds after each fresh Yahoo call
SEC_PAUSE = 0.15  # SEC allows at most 10 requests a second

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "probe"
TODAY = date.today().isoformat()

REVENUE_TAGS = [  # same line item, different XBRL names over time; first = preferred
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]


# ----------------------------------------------------------------------------- caching
def _latest(source, key, ext):
    folder = RAW / source / key
    files = sorted(folder.glob(f"*.{ext}")) if folder.exists() else []
    return files[-1] if files else None


def _new_path(source, key, ext):
    path = RAW / source / key / f"{TODAY}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cached_json(source, key, fetch, is_valid=lambda d: True):
    """Reuse the newest saved copy if there is one; otherwise download and save it."""
    hit = _latest(source, key, "json")
    if hit:
        return json.loads(hit.read_text())
    data = fetch()
    if is_valid(data):  # never cache error or rate-limit responses
        _new_path(source, key, "json").write_text(json.dumps(data))
    return data


def cached_frame(source, key, fetch):
    hit = _latest(source, key, "csv")
    if hit:
        return pd.read_csv(hit)
    df = fetch()
    if df is not None and not df.empty:
        df.to_csv(_new_path(source, key, "csv"), index=False)
    return df


def col(df, name, numeric=True):
    if name not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce") if numeric else df[name]


def short(err):
    return f"error: {type(err).__name__}: {err}"[:140]


# ----------------------------------------------------------------------------- Alpha Vantage
def probe_alpha_vantage(ticker):
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        return {"av_note": "skipped: ALPHAVANTAGE_API_KEY not set"}

    def fetch():
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "EARNINGS", "symbol": ticker, "apikey": key},
            timeout=30,
        )
        r.raise_for_status()
        time.sleep(AV_PAUSE)
        return r.json()

    def ok(d):
        return bool(d.get("quarterlyEarnings"))

    try:
        data = cached_json("alphavantage_earnings", ticker, fetch, ok)
    except Exception as e:
        return {"av_note": short(e)}
    if not ok(data):
        msg = next((str(data[k]) for k in ("Note", "Information", "Error Message") if k in data),
                   "no quarterly data returned")
        return {"av_note": msg[:140]}

    df = pd.DataFrame(data["quarterlyEarnings"]).replace({"None": None})
    reported = pd.to_datetime(col(df, "reportedDate", numeric=False), errors="coerce")
    in_window = reported.between(WINDOW_START, WINDOW_END)
    complete = col(df, "estimatedEPS").notna() & col(df, "reportedEPS").notna()
    out = {
        "av_rows": len(df),
        "av_earliest_quarter": df["fiscalDateEnding"].min(),
        "av_q_2018_2025": int((in_window & complete).sum()),
        "av_has_report_time": "reportTime" in df.columns,
        "av_fields": ",".join(df.columns),
    }
    if "reportTime" in df.columns:
        out["av_timing"] = df.loc[in_window, "reportTime"].fillna("UNKNOWN").value_counts().to_dict()
    return out


# ----------------------------------------------------------------------------- yfinance
def classify_time(ts):
    """Yahoo stamps each report with a New York time; midnight means 'time not known'."""
    if ts.hour == 0 and ts.minute == 0:
        return "UNKNOWN"
    minutes = ts.hour * 60 + ts.minute
    if minutes < 9 * 60 + 30:
        return "BMO"   # before market open
    if minutes >= 16 * 60:
        return "AMC"   # after market close
    return "DMH"       # during market hours


def probe_yf_earnings(ticker):
    if yf is None:
        return {"yf_note": "skipped: yfinance not installed"}

    def fetch():
        df = yf.Ticker(ticker).get_earnings_dates(limit=100)
        time.sleep(YF_PAUSE)
        return None if df is None else df.reset_index()

    try:
        df = cached_frame("yfinance_earnings", ticker, fetch)
    except Exception as e:
        return {"yf_note": short(e)}
    if df is None or df.empty:
        return {"yf_note": "no earnings dates returned"}

    ts = pd.to_datetime(df.iloc[:, 0], utc=True).dt.tz_convert("America/New_York")
    day = ts.dt.tz_localize(None)
    in_window = day.between(WINDOW_START, WINDOW_END + " 23:59")
    complete = col(df, "EPS Estimate").notna() & col(df, "Reported EPS").notna()
    used = in_window & complete
    return {
        "yf_rows": len(df),
        "yf_earliest": str(day.min().date()),
        "yf_q_2018_2025": int(used.sum()),
        "yf_timing": ts[used].map(classify_time).value_counts().to_dict(),
    }


def probe_yf_prices(ticker):
    if yf is None:
        return {"px_note": "skipped: yfinance not installed"}

    def fetch():
        df = yf.download(ticker, start=PULL_START, auto_adjust=False, actions=True,
                         progress=False, threads=False)
        time.sleep(YF_PAUSE)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.reset_index()

    try:
        df = cached_frame("yfinance_prices", ticker, fetch)
    except Exception as e:
        return {"px_note": short(e)}
    if df is None or df.empty:
        return {"px_rows": 0, "px_note": "no price history returned"}
    dates = pd.to_datetime(df.iloc[:, 0])
    return {
        "px_rows": len(df),
        "px_first": str(dates.min().date()),
        "px_last": str(dates.max().date()),
        "px_has_adj_close": "Adj Close" in df.columns,
    }


# ----------------------------------------------------------------------------- SEC EDGAR
def sec_get(url):
    r = requests.get(url, timeout=60, headers={
        "User-Agent": os.environ["SEC_USER_AGENT"],
        "Accept-Encoding": "gzip, deflate",
    })
    time.sleep(SEC_PAUSE)
    r.raise_for_status()
    return r.json()


def load_cik_map():
    data = cached_json("sec", "company_tickers",
                       lambda: sec_get("https://www.sec.gov/files/company_tickers.json"))
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def duration_kind(days):
    if 80 <= days <= 100:
        return "Q"    # a single quarter (13 or 14 weeks)
    if 170 <= days <= 190:
        return "6M"   # year-to-date through Q2
    if 260 <= days <= 290:
        return "9M"   # year-to-date through Q3
    if 350 <= days <= 380:
        return "FY"   # full fiscal year (52 or 53 weeks)
    return None


def probe_sec_revenue(cik):
    facts = cached_json("sec_companyfacts", cik,
                        lambda: sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"))
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for prio, tag in enumerate(REVENUE_TAGS):
        for r in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if "start" in r and str(r.get("form", "")).startswith(("10-Q", "10-K")):
                rows.append({**r, "tag": tag, "prio": prio})
    if not rows:
        return {"sec_note": "no revenue facts under the usual tags"}

    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df = df[df["end"] >= PULL_START].copy()
    df["days"] = (df["end"] - df["start"]).dt.days
    df["kind"] = df["days"].map(duration_kind)

    # Same period reported with different values in different filings = a restatement
    # (grouped by tag too: different tags can legitimately differ, e.g. around ASC 606 in 2018)
    restated = int(df.groupby(["tag", "start", "end"])["val"].nunique().gt(1).sum())
    fp_q4 = int((df["fp"] == "Q4").sum())

    # Point-in-time: preferred tag, then the earliest filing of each period
    pit = df.sort_values(["prio", "filed"]).drop_duplicates(["start", "end"])
    q = pit[pit["kind"] == "Q"]
    m9 = pit[pit["kind"] == "9M"]
    fy = pit[pit["kind"] == "FY"]

    q4_derived, q4_checked, q4_ok, q4_ends = 0, 0, 0, []
    for _, year in fy.iterrows():
        nine = m9[m9["start"] == year["start"]]
        if nine.empty:
            continue
        nine = nine.iloc[0]
        q4_derived += 1
        q4_ends.append(year["end"])
        first3 = q[(q["start"] >= year["start"]) & (q["end"] <= nine["end"])]
        if len(first3) == 3:  # Q1+Q2+Q3 should equal the 9-month YTD figure
            q4_checked += 1
            if abs(first3["val"].sum() - nine["val"]) <= 0.005 * abs(nine["val"]):
                q4_ok += 1

    # The study window is about REPORT dates, and a quarter is typically reported
    # 3-8 weeks after it ends, so approximate the report date as period end + 45 days.
    all_q_ends = set(q["end"]) | set(q4_ends)
    lag = pd.Timedelta(days=45)
    in_window = [e for e in all_q_ends
                 if pd.Timestamp(WINDOW_START) <= e + lag <= pd.Timestamp(WINDOW_END)]
    return {
        "sec_tags_used": ",".join(sorted(pit["tag"].unique())),
        "sec_q_2018_2025": len(in_window),
        "sec_q4_derived": q4_derived,
        "sec_q4_reconciled": f"{q4_ok}/{q4_checked}",
        "sec_rows_labelled_fp_Q4": fp_q4,
        "sec_restated_periods": restated,
    }


def probe_sec_8k(cik):
    """Count 8-K filings with Item 2.02 (results of operations) = earnings announcements."""
    sub = cached_json("sec_submissions", cik,
                      lambda: sec_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for f in sub["filings"].get("files", []):
        if f.get("filingTo", "") >= PULL_START:
            name = f["name"]
            extra = cached_json("sec_submissions", name.replace(".json", ""),
                                lambda n=name: sec_get(f"https://data.sec.gov/submissions/{n}"))
            frames.append(pd.DataFrame(extra))
    df = pd.concat(frames, ignore_index=True)
    hits = (df["form"].eq("8-K")
            & df["items"].fillna("").str.contains("2.02", regex=False)
            & df["filingDate"].between(WINDOW_START, WINDOW_END))
    return {"sec_8k_2_02": int(hits.sum())}


def probe_sec(cik):
    out = {"cik": cik}
    for fn in (probe_sec_revenue, probe_sec_8k):
        try:
            out.update(fn(cik))
        except Exception as e:
            out[f"{fn.__name__}_note"] = short(e)
    return out


# ----------------------------------------------------------------------------- report
def g(row, name, default="-"):
    v = row.get(name, default)
    return default if v is None or (isinstance(v, float) and pd.isna(v)) else v


def print_answers(df):
    rows = df.to_dict("records")
    line = "=" * 78
    print(f"\n{line}\nRESULTS  (study window {WINDOW_START} to {WINDOW_END}, "
          f"{EXPECTED_QUARTERS} quarters expected)\n{line}")

    print("\nQ1  Alpha Vantage EARNINGS - how far back, and is there a report-time field?")
    for r in rows:
        if "av_rows" in r and not pd.isna(r.get("av_rows")):
            print(f"  {r['ticker']:5} earliest quarter {g(r, 'av_earliest_quarter')}, "
                  f"{g(r, 'av_q_2018_2025')} complete quarters in window, "
                  f"report-time field: {'yes' if g(r, 'av_has_report_time') is True else 'no'}")
        else:
            print(f"  {r['ticker']:5} {g(r, 'av_note')}")
    fields = next((r.get("av_fields") for r in rows if isinstance(r.get("av_fields"), str)), None)
    if fields:
        print(f"  fields returned: {fields}")

    print(f"\nQ2  yfinance get_earnings_dates - {PASS_QUARTERS}+ quarters per ticker?")
    for r in rows:
        n = r.get("yf_q_2018_2025")
        if n is not None and not pd.isna(n):
            verdict = "PASS" if n >= PASS_QUARTERS else "SHORT"
            print(f"  {r['ticker']:5} {int(n):>3} quarters (earliest {g(r, 'yf_earliest')})  {verdict}")
        else:
            print(f"  {r['ticker']:5} {g(r, 'yf_note')}")

    print("\nQ3  SEC companyfacts - can Q4 be derived and does it reconcile?")
    for r in rows:
        if "sec_q4_derived" in r and not pd.isna(r.get("sec_q4_derived")):
            print(f"  {r['ticker']:5} {g(r, 'sec_q_2018_2025')} revenue quarters in window, "
                  f"{g(r, 'sec_q4_derived')} Q4s derived, Q1-Q3 reconcile {g(r, 'sec_q4_reconciled')}, "
                  f"{g(r, 'sec_rows_labelled_fp_Q4')} rows labelled fp=Q4, "
                  f"{g(r, 'sec_restated_periods')} restated periods, "
                  f"~{g(r, 'sec_8k_2_02')} earnings 8-Ks (rough: can exceed 32)")
        else:
            note = g(r, "probe_sec_revenue_note", g(r, "sec_note"))
            print(f"  {r['ticker']:5} {note}")

    print("\nQ4  Share of events with unknown timing (yfinance timestamps)")
    totals = {}
    for r in rows:
        t = r.get("yf_timing")
        if isinstance(t, dict):
            for k, v in t.items():
                totals[k] = totals.get(k, 0) + v
    n = sum(totals.values())
    if n:
        parts = ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in sorted(totals.items()))
        print(f"  {n} events: {parts}")
        print("  (UNKNOWN = midnight timestamp, an assumption - spot-check a few on IR pages)")
    else:
        print("  no yfinance timing data")
    if any(r.get("av_has_report_time") is True for r in rows):
        print("  Alpha Vantage also returns a report-time field - compare the two.")

    print("\nQ5  Acquired / delisted companies - is the data still there?")
    for r in rows:
        if r["status"] != "delisted":
            continue
        prices = (f"{g(r, 'px_rows', 0)} rows (last {g(r, 'px_last')})"
                  if g(r, "px_rows", 0) else g(r, "px_note", "none"))
        print(f"  {r['ticker']:5} prices: {prices}; "
              f"AV quarters: {g(r, 'av_q_2018_2025', g(r, 'av_note'))}; "
              f"yfinance quarters: {g(r, 'yf_q_2018_2025', g(r, 'yf_note'))}; "
              f"SEC revenue quarters: {g(r, 'sec_q_2018_2025')}")
    print(f"\n{line}")


def main():
    ua = os.getenv("SEC_USER_AGENT")
    if not ua:
        print("SEC_USER_AGENT not set - SEC checks will be skipped. "
              'Set it to "Your Name your@email.com".')
    if yf is None:
        print("yfinance not installed - run: pip install yfinance lxml")

    cik_map = {}
    if ua:
        try:
            cik_map = load_cik_map()
        except Exception as e:
            print(f"Could not load the SEC ticker file: {short(e)}")

    results = []
    for ticker in ACTIVE + list(DELISTED):
        print(f"probing {ticker} ...", flush=True)
        row = {"ticker": ticker, "status": "delisted" if ticker in DELISTED else "active"}
        row.update(probe_alpha_vantage(ticker))
        row.update(probe_yf_earnings(ticker))
        row.update(probe_yf_prices(ticker))
        cik = DELISTED.get(ticker) or cik_map.get(ticker)
        if ua and cik:
            row.update(probe_sec(cik))
        else:
            row["sec_note"] = "skipped: no SEC_USER_AGENT" if not ua else "CIK not found"
        results.append(row)

    df = pd.DataFrame(results)
    out = ROOT / "reports" / f"probe_summary_{TODAY}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print_answers(df)
    print(f"Full table saved to {out}")
    print(f"Raw responses saved under {RAW}")


if __name__ == "__main__":
    main()