#!/usr/bin/env python3
"""
pull_sec.py - download SEC "company facts" (every number from every 10-Q/10-K) for each
company in the universe that has earnings data. Part 2, step 2.

  1. finds each company's SEC ID (CIK):
       - from the S&P 500 list (current members),
       - else from the SEC's current ticker file (companies still trading),
       - else from a short hand-checked list (acquired / renamed companies),
     and checks the SEC's company name against the expected name
  2. downloads https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json for each,
     saved compressed under data/raw/sec_companyfacts/<cik>/<date>.json.gz
     (already-downloaded companies are skipped, so it is safe to rerun)
  3. writes data/processed/company_ciks.csv and prints a summary

Run from the project folder (inside the .venv):
    export SEC_USER_AGENT="Your Name your@email.com"
    python pull_sec.py
About 213 companies, ~60 MB, roughly 2-4 minutes (the SEC allows 10 requests a second).
"""

import gzip
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TODAY = date.today().isoformat()
PAUSE = 0.15

# Companies no longer in the SEC's current ticker file (acquired, merged, renamed).
# CIK and a word that must appear in the SEC's company name, as a safety check.
KNOWN = {
    "ANSS": ("0001013462", "ANSYS"),
    "JNPR": ("0001043604", "JUNIPER"),
    "WBA": ("0001618921", "WALGREEN"),
    "K": ("0000055067", "KELL"),
    "IPG": ("0000051644", "INTERPUBLIC"),
    "FL": ("0000850209", "FOOT LOCKER"),
    "JWN": ("0000072333", "NORDSTROM"),
    "HBI": ("0001359841", "HANESBRANDS"),
    "GPS": ("0000039911", "GAP"),
    "WYND": ("0001361658", "TRAVEL"),
}
ALIASES = {"GPS": ["GAP"], "WYND": ["TNL"], "BF.B": ["BF-B", "BF.B"], "PSKY": ["PSKY", "PARA"]}


def get(url, ua):
    for attempt in range(4):
        r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
                         timeout=60)
        time.sleep(PAUSE)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def main():
    ua = os.getenv("SEC_USER_AGENT")
    if not ua or "@" not in ua:
        sys.exit('Set your name and email first:  export SEC_USER_AGENT="Your Name your@email.com"')

    comp = pd.read_csv(PROC / "universe_companies.csv", dtype={"cik": str})
    cov = pd.read_csv(PROC / "pull_coverage.csv")
    comp = comp.merge(cov[["company_key", "ticker_used", "events_with_estimate"]], on="company_key")
    comp = comp[comp["events_with_estimate"] > 0].copy()

    print("loading the SEC ticker list ...")
    tickers = get("https://www.sec.gov/files/company_tickers.json", ua)
    by_ticker = {v["ticker"].upper(): (str(v["cik_str"]).zfill(10), v["title"]) for v in tickers.values()}

    rows = []
    for _, c in comp.iterrows():
        key, cik, how, expect = c["company_key"], None, "", None
        if isinstance(c["cik"], str) and c["cik"].strip() and c["cik"] != "nan":
            cik, how = c["cik"].split(".")[0].zfill(10), "S&P 500 list"
        if cik is None:
            candidates = [key, str(c["ticker_used"])] + ALIASES.get(key, [])
            variants = [v for t in candidates for v in
                        (t.upper(), t.upper().replace(".", "-"), t.upper().replace("-", "."))]
            hit = next((v for v in variants if v in by_ticker), None)
            if hit:
                cik, how = by_ticker[hit][0], f"SEC ticker file ({hit})"
        if cik is None and key in KNOWN:
            cik, expect = KNOWN[key]
            how = "hand-checked list"
        rows.append({"company_key": key, "cik": cik, "found_by": how, "expect": expect})
    ciks = pd.DataFrame(rows)

    print(f"downloading company facts for {ciks['cik'].notna().sum()} companies ...")
    names, status = [], []
    for i, r in ciks.iterrows():
        if not isinstance(r["cik"], str):
            names.append(None); status.append("no CIK found"); continue
        folder = RAW / "sec_companyfacts" / r["cik"]
        files = sorted(folder.glob("*.json.gz")) if folder.exists() else []
        try:
            if files:
                data = json.load(gzip.open(files[-1]))
                st = "cached"
            else:
                data = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{r['cik']}.json", ua)
                folder.mkdir(parents=True, exist_ok=True)
                with gzip.open(folder / f"{TODAY}.json.gz", "wt") as f:
                    json.dump(data, f)
                st = "downloaded"
            name = data.get("entityName", "")
            if isinstance(r["expect"], str) and r["expect"] not in name.upper():
                st = f"NAME MISMATCH (expected {r['expect']})"
            names.append(name); status.append(st)
        except Exception as e:
            names.append(None); status.append(f"error: {str(e)[:80]}")
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ciks)}")
    ciks["sec_name"], ciks["status"] = names, status
    ciks.drop(columns=["expect"]).to_csv(PROC / "company_ciks.csv", index=False)

    ok = ciks["status"].isin(["cached", "downloaded"])
    print(f"\nSEC company facts available for {ok.sum()} of {len(ciks)} companies")
    print(ciks["found_by"].str.split(" \\(").str[0].value_counts().to_string())
    bad = ciks[~ok]
    if len(bad):
        print("\nProblems (paste these to Claude):")
        print(bad[["company_key", "cik", "found_by", "status"]].to_string(index=False))
    print("\nSaved data/processed/company_ciks.csv")


if __name__ == "__main__":
    main()
