#!/usr/bin/env python3
"""
pull_data.py - download and check earnings + price data for the point-in-time universe.

For every company in data/processed/universe_companies.csv (built by build_universe.py):
  1. yfinance earnings history: EPS estimate, reported EPS, report date and time
  2. yfinance daily prices from 2016 (raw and adjusted close, dividends, splits)
  3. benchmark prices: SPY and the four sector ETFs
  4. Alpha Vantage cross-check on a fixed, stratified sample of 50 companies
  5. SEC 8-K Item 2.02 filing dates, to check yfinance report dates (current members)

Membership rule (put this in the charter):
  A quarter is in the universe if the company was an S&P 500 member on its earnings
  report date. All other quarters are kept in the files (useful as history for features)
  but flagged in_universe = False.

Every raw download is saved under data/raw/<source>/<ticker>/<date>.* and reused on later
runs, so the script can be stopped and restarted at any time: it continues where it left
off. Failed or rate-limited downloads are not saved, so a rerun retries them.

Run from the project folder (inside the .venv):
    export ALPHAVANTAGE_API_KEY=your_key                  # optional: cross-check
    export SEC_USER_AGENT="Your Name your@email.com"      # optional: date check
    python pull_data.py

Takes roughly 20-40 minutes the first time (mostly polite pauses between Yahoo requests).
Alpha Vantage allows 25 calls a day, so the 50-company cross-check completes over 2-3
days of reruns; everything else completes in one run.
"""

import json
import os
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("yfinance is not installed - run: pip install yfinance lxml")

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
TODAY = date.today().isoformat()

WINDOW_START, WINDOW_END = "2018-01-01", "2025-12-31"
PRICE_START, PRICE_END = "2016-01-01", "2026-02-28"  # a little past the window for returns
BENCHMARKS = {"SPY": "S&P 500", "XLK": "Information Technology", "XLY": "Consumer Discretionary",
              "XLP": "Consumer Staples", "XLC": "Communication Services"}  # XLC starts June 2018

# Tickers that moved after the company left the index: try these if the first returns nothing
ALT_TICKERS = {
    "PARA": ["PSKY"],   # Paramount merged with Skydance, Aug 2025
    "GPS": ["GAP"],     # Gap changed its ticker to GAP in 2024
    "WYND": ["TNL"],    # Wyndham Worldwide -> Wyndham Destinations -> Travel + Leisure
}

YF_PAUSE = 2.5            # seconds between Yahoo requests
YF_RETRIES = 3            # on rate-limit errors: wait and retry this many times
AV_SAMPLE_SIZE = 50
AV_SAMPLE_SEED = 42       # fixed so the same 50 companies are picked every run
AV_PAUSE = 13
SEC_PAUSE = 0.15
TOLERANCE = 0.02          # $ difference that counts as a disagreement
DATE_GAP_DAYS = 3

LOG = []


def say(text=""):
    print(text, flush=True)
    LOG.append(str(text))


# ----------------------------------------------------------------------------- caching
def latest(source, key, ext):
    folder = RAW / source / key
    files = sorted(folder.glob(f"*.{ext}")) if folder.exists() else []
    return files[-1] if files else None


def save_path(source, key, ext):
    p = RAW / source / key / f"{TODAY}.{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- yfinance
def yf_call(fn, label):
    """Run a Yahoo request with pauses and rate-limit retries. Returns (result, error)."""
    for attempt in range(YF_RETRIES + 1):
        try:
            out = fn()
            time.sleep(YF_PAUSE)
            return out, None
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if "RateLimit" in msg or "Too Many Requests" in msg or "429" in msg:
                wait = 60 * (attempt + 1)
                say(f"    rate-limited on {label}; waiting {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            time.sleep(YF_PAUSE)
            return None, msg[:160]
    return None, "rate-limited: rerun later"


def yf_earnings(ticker):
    hit = latest("yfinance_earnings", ticker, "csv")
    if hit:
        return pd.read_csv(hit), None
    df, err = yf_call(lambda: yf.Ticker(ticker).get_earnings_dates(limit=100), f"{ticker} earnings")
    if df is None or len(df) == 0:
        return None, err or "no earnings dates returned"
    df = df.reset_index()
    df.to_csv(save_path("yfinance_earnings", ticker, "csv"), index=False)
    return df, None


def yf_prices(ticker):
    hit = latest("yfinance_prices", ticker, "csv")
    if hit:
        return pd.read_csv(hit), None

    def fetch():
        return yf.download(ticker, start=PRICE_START, end=PRICE_END, auto_adjust=False,
                           actions=True, progress=False, threads=False)
    df, err = yf_call(fetch, f"{ticker} prices")
    if df is None or len(df) == 0:
        return None, err or "no price history returned"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.to_csv(save_path("yfinance_prices", ticker, "csv"), index=False)
    return df, None


def classify_time(ts):
    if ts.hour == 0 and ts.minute == 0:
        return "UNKNOWN"          # assumption: Yahoo uses midnight when the time is not known
    m = ts.hour * 60 + ts.minute
    if m < 9 * 60 + 30:
        return "BMO"
    if m >= 16 * 60:
        return "AMC"
    return "DMH"


def tidy_earnings(df, company_key, ticker_used):
    ts = pd.to_datetime(df.iloc[:, 0], utc=True).dt.tz_convert("America/New_York")
    est = pd.to_numeric(df.get("EPS Estimate"), errors="coerce").round(2)
    act = pd.to_numeric(df.get("Reported EPS"), errors="coerce").round(2)
    out = pd.DataFrame({
        "company_key": company_key,
        "ticker_used": ticker_used,
        "report_datetime": ts.dt.strftime("%Y-%m-%d %H:%M"),
        "report_date": pd.to_datetime(ts.dt.date),
        "report_time": ts.map(classify_time),
        "eps_estimate": est,
        "eps_actual": act,
    })
    out = out.dropna(subset=["eps_actual"])          # drop future/unreported dates
    out["surprise"] = (out["eps_actual"] - out["eps_estimate"]).round(2)
    # Beat = actual strictly above estimate, after rounding both to the cent. Ties are not beats.
    out["beat"] = (out["eps_actual"] > out["eps_estimate"]).astype(float).where(
        out["eps_estimate"].notna())                   # 1.0 = beat, 0.0 = not, blank = no estimate
    return out.drop_duplicates(subset=["report_date"]).sort_values("report_date")


# ----------------------------------------------------------------------------- Alpha Vantage
def stratified_sample(companies):
    """50 companies spread across sectors, including ones that left the index."""
    rng = random.Random(AV_SAMPLE_SEED)
    picks = []
    left = companies[companies["status"] != "current member"]
    stay = companies[companies["status"] == "current member"]
    n_left = max(10, round(AV_SAMPLE_SIZE * len(left) / len(companies)))
    for pool, n in ((left, n_left), (stay, AV_SAMPLE_SIZE - n_left)):
        by_sector = pool.groupby("sector")
        for sector, grp in by_sector:
            k = max(1, round(n * len(grp) / len(pool)))
            keys = sorted(grp["company_key"])
            picks += rng.sample(keys, min(k, len(keys)))
    return sorted(set(picks))[:AV_SAMPLE_SIZE]


def av_earnings(ticker, key, budget):
    hit = latest("alphavantage_earnings", ticker, "json")
    if hit:
        return json.loads(hit.read_text()), None, budget
    if budget["left"] <= 0:
        return None, "daily budget used - rerun tomorrow", budget
    r = requests.get("https://www.alphavantage.co/query",
                     params={"function": "EARNINGS", "symbol": ticker, "apikey": key}, timeout=30)
    time.sleep(AV_PAUSE)
    data = r.json()
    if not data.get("quarterlyEarnings"):
        note = next((str(data[k]) for k in ("Information", "Note", "Error Message") if k in data),
                    "no data returned")
        if "apikey" in note.lower() or "api key" in note.lower() and "invalid" in note.lower():
            budget["left"] = 0
            return None, "key rejected - check the export ALPHAVANTAGE_API_KEY line", budget
        if "rate limit" in note.lower() or "requests per day" in note.lower():
            budget["left"] = 0
            return None, "daily limit reached - rerun tomorrow", budget
        return None, note[:120], budget
    budget["left"] -= 1
    save_path("alphavantage_earnings", ticker, "json").write_text(json.dumps(data))
    return data, None, budget


def compare_av(av_json, yf_events, company_key):
    av = pd.DataFrame(av_json["quarterlyEarnings"]).replace({"None": None})
    av["report_date"] = pd.to_datetime(av["reportedDate"]).astype("datetime64[ns]")
    av["av_actual"] = pd.to_numeric(av["reportedEPS"], errors="coerce").round(2)
    av["av_estimate"] = pd.to_numeric(av["estimatedEPS"], errors="coerce").round(2)
    av = av[av["report_date"].between(WINDOW_START, WINDOW_END)].sort_values("report_date")
    yfe = yf_events[["report_date", "eps_estimate", "eps_actual"]].copy()
    yfe["report_date"] = yfe["report_date"].astype("datetime64[ns]")
    yfe["yf_date"] = yfe["report_date"]
    m = pd.merge_asof(av, yfe.sort_values("report_date"), on="report_date", direction="nearest")
    m["date_gap_days"] = (m["report_date"] - m["yf_date"]).dt.days.abs()
    m["company_key"] = company_key
    m["issue"] = ""
    m.loc[m["date_gap_days"] > DATE_GAP_DAYS, "issue"] = "report date differs"
    ok_date = m["date_gap_days"] <= DATE_GAP_DAYS
    m.loc[ok_date & ((m["av_actual"] - m["eps_actual"]).abs() > TOLERANCE), "issue"] += " actual differs"
    m.loc[ok_date & ((m["av_estimate"] - m["eps_estimate"]).abs() > TOLERANCE), "issue"] += " estimate differs"
    m["issue"] = m["issue"].str.strip()
    return m[["company_key", "fiscalDateEnding", "report_date", "yf_date", "date_gap_days",
              "av_estimate", "eps_estimate", "av_actual", "eps_actual", "issue"]]


# ----------------------------------------------------------------------------- SEC
def sec_8k_dates(cik, ua):
    def get(url):
        r = requests.get(url, headers={"User-Agent": ua}, timeout=60)
        time.sleep(SEC_PAUSE)
        r.raise_for_status()
        return r.json()

    def cached(key, url):
        hit = latest("sec_submissions", key, "json")
        if hit:
            return json.loads(hit.read_text())
        data = get(url)
        save_path("sec_submissions", key, "json").write_text(json.dumps(data))
        return data

    sub = cached(cik, f"https://data.sec.gov/submissions/CIK{cik}.json")
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for f in sub["filings"].get("files", []):
        if f.get("filingTo", "") >= PRICE_START:
            frames.append(pd.DataFrame(cached(f["name"].replace(".json", ""),
                                              f"https://data.sec.gov/submissions/{f['name']}")))
    df = pd.concat(frames, ignore_index=True)
    hits = df[df["form"].eq("8-K") & df["items"].fillna("").str.contains("2.02", regex=False)]
    return set(pd.to_datetime(hits["filingDate"]))


# ----------------------------------------------------------------------------- main
def main():
    companies = pd.read_csv(PROC / "universe_companies.csv", dtype={"cik": str})
    spells = pd.read_csv(PROC / "universe_membership.csv", parse_dates=["valid_from", "valid_to"])
    say(f"Pulling data for {len(companies)} companies - started {pd.Timestamp.now():%H:%M}")

    # ---- 1-2: yfinance earnings and prices for every company
    events, coverage = [], []
    for i, c in companies.iterrows():
        key, yticker = c["company_key"], c["yahoo_ticker"]
        say(f"[{i + 1}/{len(companies)}] {key}")
        tried, earn, err, used = [], None, None, None
        for t in [yticker] + ALT_TICKERS.get(key, []):
            tried.append(t)
            earn, err = yf_earnings(t)
            if earn is not None:
                used = t
                break
        prices, perr = None, None
        for t in dict.fromkeys([used or yticker] + ALT_TICKERS.get(key, [])):
            prices, perr = yf_prices(t)          # e.g. Gap's prices live under GAP, not GPS
            if prices is not None:
                break
        row = {"company_key": key, "sector": c["sector"], "status": c["status"],
               "tickers_tried": ",".join(tried), "ticker_used": used,
               "expected_quarters": c["approx_quarters"],
               "earnings_error": err if earn is None else "",
               "price_rows": 0 if prices is None else len(prices),
               "price_error": perr or ""}
        if earn is not None:
            ev = tidy_earnings(earn, key, used)
            mine = spells[spells["company_key"] == key]
            end = mine["valid_to"].fillna(pd.Timestamp(WINDOW_END) + pd.Timedelta(days=1))
            ev["in_universe"] = pd.Series(
                [bool(((mine["valid_from"] <= d) & (d < end)).any())
                 and pd.Timestamp(WINDOW_START) <= d <= pd.Timestamp(WINDOW_END)
                 for d in ev["report_date"]], index=ev.index, dtype=bool)
            events.append(ev)
            row["events_in_universe"] = int(ev["in_universe"].sum())
            row["events_with_estimate"] = int((ev["in_universe"] & ev["eps_estimate"].notna()).sum())
            if ev.empty:
                row["earnings_error"] = "earnings table has no reported quarters"
        else:
            row["events_in_universe"] = 0
            row["events_with_estimate"] = 0
        coverage.append(row)

    for t in BENCHMARKS:
        say(f"benchmark {t}")
        yf_prices(t)

    # full split history per company (prices are adjusted for ALL splits, including ones
    # after the price window; needed to rebuild the real share price for market value)
    say("split histories ...")
    for t in dict.fromkeys(r["ticker_used"] for r in coverage if r["ticker_used"]):
        if latest("yfinance_splits", t, "csv"):
            continue
        sp, err = yf_call(lambda t=t: yf.Ticker(t).splits, f"{t} splits")
        if err is None and sp is not None:
            sp = sp.reset_index()
            sp.columns = ["date", "ratio"]
            sp.to_csv(save_path("yfinance_splits", t, "csv"), index=False)

    cov = pd.DataFrame(coverage)
    ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    PROC.mkdir(parents=True, exist_ok=True)
    ev.to_csv(PROC / "earnings_events.csv", index=False, date_format="%Y-%m-%d")
    cov.to_csv(PROC / "pull_coverage.csv", index=False)

    # ---- 4: Alpha Vantage cross-check on a stratified sample
    av_rows, av_status = [], {}
    key_av = os.getenv("ALPHAVANTAGE_API_KEY")
    sample = stratified_sample(companies)
    pd.Series(sample, name="company_key").to_csv(PROC / "av_sample.csv", index=False)
    if key_av and not ev.empty:
        budget = {"left": 25}
        for k in sample:
            r = cov.loc[cov["company_key"] == k].iloc[0]
            t = r["ticker_used"]
            if pd.isna(t) or r["events_with_estimate"] == 0:
                av_status[k] = "no yfinance data to compare"
                continue
            data, note, budget = av_earnings(t, key_av, budget)
            av_status[k] = note or "ok"
            if data:
                av_rows.append(compare_av(data, ev[ev["company_key"] == k], k))
    av = pd.concat(av_rows, ignore_index=True) if av_rows else pd.DataFrame()
    if not av.empty:
        av.to_csv(PROC / "av_crosscheck.csv", index=False, date_format="%Y-%m-%d")

    # ---- 5: SEC 8-K date check (current members have CIKs)
    ua = os.getenv("SEC_USER_AGENT")
    sec_rows = []
    if ua and not ev.empty:
        say("checking report dates against SEC 8-K filings ...")
        for _, c in companies.dropna(subset=["cik"]).iterrows():
            try:
                dates = sec_8k_dates(str(c["cik"]).split(".")[0].zfill(10), ua)
            except Exception as e:
                sec_rows.append({"company_key": c["company_key"], "error": str(e)[:100]})
                continue
            mine = ev[(ev["company_key"] == c["company_key"]) & ev["in_universe"]]
            matched = pd.Series([any(abs((d - x).days) <= 1 for x in dates)
                                 for d in mine["report_date"]], index=mine.index, dtype=bool)
            sec_rows.append({"company_key": c["company_key"], "events": len(mine),
                             "matched_8k": int(matched.sum()),
                             "unmatched_dates": ",".join(mine.loc[~matched, "report_date"]
                                                         .dt.strftime("%Y-%m-%d"))})
    sec = pd.DataFrame(sec_rows)
    if not sec.empty:
        sec.to_csv(PROC / "sec_date_check.csv", index=False)

    report(cov, ev, av, av_status, sample, sec)


def report(cov, ev, av, av_status, sample, sec):
    line = "=" * 78
    say(f"\n{line}\nDATA PULL SUMMARY  ({TODAY})\n{line}")

    has = cov["events_with_estimate"] > 0
    say(f"\n1. COVERAGE  {int(has.sum())} of {len(cov)} companies have usable earnings data "
        f"in the window")
    exp, got = cov["expected_quarters"].sum(), cov["events_with_estimate"].sum()
    say(f"   Quarters in the universe with estimate + actual: {got} of ~{exp} expected "
        f"({got / exp:.0%})")
    t = cov.assign(has_data=has).pivot_table(index="status", columns="has_data",
                                             values="company_key", aggfunc="count", fill_value=0)
    say(t.rename(columns={True: "has data", False: "no data"}).to_string())
    none = cov[~has & (cov["expected_quarters"] > 0)]   # skip companies that joined after 2025
    if len(none):
        lost = none["expected_quarters"].sum()
        say(f"\n   No data for {len(none)} companies (~{lost} quarters, {lost / exp:.0%} of the "
            f"universe) - this is the survivorship gap for the charter:")
        for _, r in none.sort_values("status").iterrows():
            say(f"     {r['company_key']:6} {r['sector'][:22]:22} {r['status']:20} "
                f"~{r['expected_quarters']} qtrs  ({r['earnings_error'][:50]})")
    thin = cov[has & (cov["events_with_estimate"] < 0.8 * cov["expected_quarters"])]
    if len(thin):
        say(f"\n   Partial coverage (<80% of expected quarters): {len(thin)} companies")
        for _, r in thin.iterrows():
            say(f"     {r['company_key']:6} {r['events_with_estimate']}/{r['expected_quarters']}")
    no_px = cov[has & (cov["price_rows"] == 0)]
    if len(no_px):
        say(f"\n   Earnings but no prices: {', '.join(no_px['company_key'])}")

    if not ev.empty:
        u = ev[ev["in_universe"] & ev["beat"].notna()].copy()
        u["beat"] = u["beat"].astype(float)
        u["year"] = u["report_date"].dt.year
        u = u.merge(cov[["company_key", "sector"]], on="company_key")
        say(f"\n2. BEAT RATE SANITY CHECK  (expect roughly mid-70s to 80s %; "
            f"far lower suggests GAAP mixed in)")
        bt = u.pivot_table(index="sector", columns="year", values="beat", aggfunc="mean")
        bt["all"] = u.groupby("sector")["beat"].mean()
        say((bt * 100).round(0).astype("Int64").to_string())
        ties = (u["eps_actual"] == u["eps_estimate"]).mean()
        say(f"   Overall beat rate {u['beat'].mean():.1%} on {len(u)} quarters; "
            f"exact ties (counted as not a beat): {ties:.1%}")

        say("\n3. REPORT TIMING (in-universe quarters)")
        tt = ev[ev["in_universe"]]["report_time"].value_counts()
        say("   " + ", ".join(f"{k} {v} ({v / tt.sum():.0%})" for k, v in tt.items()))

    say(f"\n4. ALPHA VANTAGE CROSS-CHECK  (stratified sample of {len(sample)})")
    done = [k for k, v in av_status.items() if v == "ok"]
    no_yf = [k for k, v in av_status.items() if v == "no yfinance data to compare"]
    pending = [k for k, v in av_status.items() if v not in ("ok", "no yfinance data to compare")]
    if not av_status:
        say("   skipped (ALPHAVANTAGE_API_KEY not set)")
    else:
        reasons = pd.Series([v for v in av_status.values()
                             if v not in ("ok", "no yfinance data to compare")]).value_counts()
        for reason, n in reasons.items():
            say(f"   Alpha Vantage refused {n} call(s): {reason}")
        say(f"   compared {len(done)} companies so far; {len(pending)} still to do "
            f"(rerun on later days - saved results are reused); "
            f"{len(no_yf)} skipped (no yfinance data)")
    if not av.empty:
        issues = av[av["issue"] != ""]
        say(f"   {len(av)} quarters compared; {len(issues)} with an issue:")
        say("   " + ", ".join(f"{k}: {v}" for k, v in issues["issue"].value_counts().items()))
        say("   details in data/processed/av_crosscheck.csv")

    say("\n5. SEC 8-K DATE CHECK  (yfinance report date within 1 day of an earnings 8-K)")
    if sec.empty:
        say("   skipped (SEC_USER_AGENT not set)")
    elif "events" in sec:
        s = sec.dropna(subset=["events"])
        say(f"   {int(s['matched_8k'].sum())} of {int(s['events'].sum())} report dates matched "
            f"({s['matched_8k'].sum() / max(s['events'].sum(), 1):.0%})")
        low = s[s["matched_8k"] < 0.9 * s["events"]]
        if len(low):
            say(f"   Companies below 90%: {', '.join(low['company_key'])} "
                f"(see data/processed/sec_date_check.csv)")

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"data_pull_summary_{TODAY}.txt"
    out.write_text("\n".join(LOG))
    say(f"\nSaved: data/processed/earnings_events.csv, pull_coverage.csv, and this summary "
        f"to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
