#!/usr/bin/env python3
"""
build_universe.py - point-in-time S&P 500 universe for the Earnings Predictor project.

Each quarter uses whichever companies were in the S&P 500 at that date, limited to four
sectors (today's GICS scheme, applied throughout):
    Information Technology, Consumer Discretionary, Consumer Staples, Communication Services

Inputs
  - S&P 500 membership history from github.com/fja05680/sp500 (downloaded once, cached)
  - exited_tickers_reference.csv (next to this script): sector and successor ticker for
    every ticker that left the index. Rows marked confidence=verify should be checked.

Outputs (in data/processed/)
  - universe_membership.csv  one row per membership spell: who was in, from when, to when
  - universe_companies.csv   one row per company, with status and rough quarter count

Run from the project folder:
    python build_universe.py
"""

import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "sp500_membership"
OUT = ROOT / "data" / "processed"
REFERENCE = ROOT / "exited_tickers_reference.csv"

WINDOW_START = pd.Timestamp("2018-01-01")
WINDOW_END = pd.Timestamp("2025-12-31")
SECTORS = ["Information Technology", "Consumer Discretionary",
           "Consumer Staples", "Communication Services"]

REPO = "https://raw.githubusercontent.com/fja05680/sp500/master/"
FILES = {
    "history": "S&P 500 Historical Components & Changes (Updated).csv",
    "current": "sp500.csv",
}


def download(name):
    """Download a file from the membership repo once; reuse the saved copy afterwards."""
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        r = requests.get(REPO + quote(name), timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"downloaded {name}")
    return path


def membership_spells(history):
    """Turn daily snapshots into (ticker, valid_from, valid_to) spells inside the window.
    valid_to is the first snapshot date the ticker is missing (exclusive), or blank."""
    history = history.sort_values("date")
    before = history[history["date"] < WINDOW_START]
    inside = history[(history["date"] >= WINDOW_START) & (history["date"] <= WINDOW_END)]
    rows = pd.concat([before.tail(1), inside])  # state on 2018-01-01, then every change

    spells, open_since = [], {}
    for _, snap in rows.iterrows():
        members = set(snap["tickers"].split(","))
        start = max(snap["date"], WINDOW_START)
        for t in members - open_since.keys():
            open_since[t] = start
        for t in list(open_since.keys() - members):
            spells.append((t, open_since.pop(t), snap["date"]))
    for t, since in open_since.items():
        spells.append((t, since, pd.NaT))  # still a member at the end of the window
    return pd.DataFrame(spells, columns=["ticker", "valid_from", "valid_to"])


# Current members with two share classes in the index: count each company once,
# under the class Yahoo and the vendors usually report earnings for.
DUAL_CLASS = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}


def union_days(group):
    """Days of membership, counting overlapping spells (e.g. two share classes) once."""
    end = group["valid_to"].fillna(WINDOW_END + pd.Timedelta(days=1))
    spans = sorted(zip(group["valid_from"], end))
    total, cur_start, cur_end = 0, None, None
    for s, e in spans:
        if cur_end is None or s > cur_end:
            if cur_end is not None:
                total += (cur_end - cur_start).days
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    if cur_end is not None:
        total += (cur_end - cur_start).days
    return total


def resolve(ticker, successor):
    """Follow renames/mergers to the ticker that holds the data today (e.g. FB -> META)."""
    seen = set()
    while successor.get(ticker) and ticker not in seen:
        seen.add(ticker)
        ticker = successor[ticker]
    return ticker


def main():
    history = pd.read_csv(download(FILES["history"]), parse_dates=["date"])
    current = pd.read_csv(download(FILES["current"]))
    if not REFERENCE.exists():
        sys.exit(f"Missing {REFERENCE.name} - put it next to this script.")
    ref = pd.read_csv(REFERENCE, dtype=str).fillna("")

    last_snapshot = history["date"].max()
    if last_snapshot < WINDOW_END - pd.Timedelta(days=30):
        print(f"WARNING: membership history ends {last_snapshot.date()}, before the window ends")

    sector = dict(zip(current["Symbol"], current["GICS Sector"]))
    cik = {s: str(int(c)).zfill(10) for s, c in zip(current["Symbol"], current["CIK"]) if pd.notna(c)}
    current_set = set(current["Symbol"])
    for _, r in ref.iterrows():
        sector.setdefault(r["ticker"], r["sector"])
    successor = {r["ticker"]: r["successor_ticker"] for _, r in ref.iterrows() if r["successor_ticker"]}
    successor.update(DUAL_CLASS)
    verify = set(ref.loc[ref["confidence"] == "verify", "ticker"])

    spells = membership_spells(history)
    spells["company_key"] = spells["ticker"].map(lambda t: resolve(t, successor))
    # A renamed company takes the sector of the ticker that carries it today (today's scheme)
    spells["sector"] = spells["company_key"].map(sector).fillna(spells["ticker"].map(sector))
    spells["sector"] = spells["sector"].fillna("UNKNOWN")

    unknown = sorted(spells.loc[spells["sector"] == "UNKNOWN", "ticker"].unique())
    if unknown:
        print(f"\nWARNING: no sector for {len(unknown)} tickers - add them to "
              f"{REFERENCE.name}: {', '.join(unknown)}")

    scope = spells[spells["sector"].isin(SECTORS)].copy()
    scope["yahoo_ticker"] = scope["company_key"].str.replace(".", "-", regex=False)
    scope["needs_check"] = scope["ticker"].isin(verify) | scope["company_key"].isin(verify)

    # ---- one row per company
    g = scope.groupby("company_key")
    companies = pd.DataFrame({
        "sector": g["sector"].first(),
        "yahoo_ticker": g["yahoo_ticker"].first(),
        "tickers_used": g["ticker"].agg(lambda s: ",".join(sorted(set(s)))),
        "first_in": g["valid_from"].min(),
        "last_out": g["valid_to"].max(),
        "open_spell": g["valid_to"].agg(lambda s: s.isna().any()),
        "member_days": g[["valid_from", "valid_to"]].apply(union_days),
        "needs_check": g["needs_check"].any(),
    })
    companies.loc[companies["open_spell"], "last_out"] = pd.NaT
    companies["member_on_2018_01_01"] = g["valid_from"].min() == WINDOW_START
    companies["approx_quarters"] = (companies["member_days"] / 91.3).round().astype(int)
    companies["cik"] = companies.index.map(cik)

    def status(row):
        if row.name in current_set:
            return "current member"
        if pd.isna(row["last_out"]):
            return "left after 2025"      # member through the whole window, gone since
        return "left during window"
    companies["status"] = companies.apply(status, axis=1)
    # A company can also still exist but have been demoted from the index (e.g. Gap, Macy's):
    # it is "left during window" here; whether data exists is tested in the data pull.
    companies = companies.drop(columns=["open_spell"]).reset_index()

    OUT.mkdir(parents=True, exist_ok=True)
    scope.sort_values(["company_key", "valid_from"]).to_csv(
        OUT / "universe_membership.csv", index=False, date_format="%Y-%m-%d")
    companies.sort_values(["sector", "company_key"]).to_csv(
        OUT / "universe_companies.csv", index=False, date_format="%Y-%m-%d")

    # ---- summary
    line = "=" * 78
    print(f"\n{line}\nPOINT-IN-TIME UNIVERSE  {WINDOW_START.date()} to {WINDOW_END.date()}"
          f"  (membership data to {last_snapshot.date()})\n{line}")
    print(f"Companies ever in scope: {len(companies)}   "
          f"(members on 2018-01-01: {int(companies['member_on_2018_01_01'].sum())}, "
          f"added later: {int((~companies['member_on_2018_01_01']).sum())})")
    print(f"Approx. company-quarters in scope: {companies['approx_quarters'].sum()}\n")
    table = companies.pivot_table(index="sector", columns="status", values="company_key",
                                  aggfunc="count", fill_value=0, margins=True, margins_name="Total")
    print(table.to_string())

    gone = companies[companies["status"] != "current member"].sort_values("last_out")
    q_gone = gone["approx_quarters"].sum()
    print(f"\nNot in the index today: {len(gone)} companies, ~{q_gone} company-quarters "
          f"({q_gone / companies['approx_quarters'].sum():.0%} of the total).")
    print("These are the ones most likely to have no estimate data (next step tests this):")
    for _, r in gone.iterrows():
        out = r["last_out"].date() if pd.notna(r["last_out"]) else "after 2025"
        flag = "  <- check sector" if r["needs_check"] else ""
        print(f"  {r['company_key']:6} {r['sector'][:22]:22} left {out}  "
              f"~{r['approx_quarters']:>2} qtrs  (as {r['tickers_used']}){flag}")

    renamed = companies[companies["tickers_used"] != companies["company_key"]]
    if len(renamed):
        print(f"\nRenamed companies merged under today's ticker ({len(renamed)}):")
        for _, r in renamed.iterrows():
            print(f"  {r['tickers_used']} -> {r['company_key']}")

    checks = companies[companies["needs_check"]]
    if len(checks):
        print(f"\nSector labels marked 'verify' in scope ({len(checks)}): "
              + ", ".join(checks["company_key"]))
    print(f"\nSaved {OUT / 'universe_membership.csv'}\n      {OUT / 'universe_companies.csv'}")


if __name__ == "__main__":
    main()
