#!/usr/bin/env python3
"""
build_fundamentals.py - Part 2, step 2: SEC fundamentals (group C) and the true reporting
lag (group E), added to the feature table.

How the SEC data is turned into quarters (the pitfalls from the research brief):
  - Periods are keyed by their start and end dates, never by the fiscal-year/quarter labels.
  - Every value is the FIRST-FILED version, so later restatements never leak into history.
  - 10-Qs often report year-to-date totals; quarters are derived by differencing
    (Q2 = 6-month - 3-month, Q3 = 9-month - 6-month, Q4 = full year - 9-month).
    A derived quarter counts as filed when the LATER of its two filings was filed.
  - Several tag names mean the same line item (e.g. revenue); the preferred tag is used,
    and growth rates only compare a tag with itself.

TIMING RULE: a feature for a report may only use SEC values filed BEFORE the report date.
In practice that is the previous quarter's 10-Q/10-K, because the current quarter's filing
comes after the earnings announcement. Checked by a leakage test at the end.

Inputs : data/processed/features_v1.csv, company_ciks.csv (from pull_sec.py), the raw
         company facts, and the yfinance prices (for market value).
Outputs: data/processed/fundamentals_quarterly.csv  (one row per company-quarter-metric)
         data/processed/features_v2.csv             (features_v1 + the new columns)
         data/processed/feature_dictionary.csv       (updated)
         reports/feature_quintiles_v2.txt

Run from the project folder:  python build_fundamentals.py
"""

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FLOWS = {  # duration items: preferred tag first
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
}
STOCKS = {  # point-in-time (balance sheet) items
    "inventory": ["InventoryNet", "InventoryGross"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
Q_DAYS, H_DAYS, N_DAYS, Y_DAYS = (80, 100), (170, 190), (260, 290), (350, 380)
DICT_NEW = []


# ----------------------------------------------------------------------------- parsing
def load_facts(cik):
    files = sorted((RAW / "sec_companyfacts" / cik).glob("*.json*"))
    if not files:
        return None
    f = files[-1]
    return json.load(gzip.open(f) if f.suffix == ".gz" else open(f))


def first_filed(rows, instant):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["form"].astype(str).str.startswith(("10-Q", "10-K"))].copy()
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    if instant:
        return df.sort_values("filed").drop_duplicates("end")[["end", "val", "filed"]]
    df["start"] = pd.to_datetime(df["start"])
    df["days"] = (df["end"] - df["start"]).dt.days
    return df.sort_values("filed").drop_duplicates(["start", "end"])[["start", "end", "days", "val", "filed"]]


def quarters_from_flows(df):
    """Single-quarter values from direct 3-month facts and from differencing YTD totals."""
    if df.empty:
        return pd.DataFrame(columns=["end", "val", "filed"])
    inr = lambda d, r: (d >= r[0]) & (d <= r[1])
    direct = df[inr(df["days"], Q_DAYS)][["end", "val", "filed"]]
    cum = df[inr(df["days"], Q_DAYS) | inr(df["days"], H_DAYS) |
             inr(df["days"], N_DAYS) | inr(df["days"], Y_DAYS)]
    derived = []
    for _, grp in cum.groupby("start"):
        grp = grp.sort_values("end")
        rows = grp.to_dict("records")
        for a, b in zip(rows, rows[1:]):
            gap = (b["end"] - a["end"]).days
            if Q_DAYS[0] <= gap <= Q_DAYS[1]:
                derived.append({"end": b["end"], "val": b["val"] - a["val"],
                                "filed": max(a["filed"], b["filed"])})
    derived = pd.DataFrame(derived, columns=["end", "val", "filed"])
    parts = [x for x in (direct.assign(p=0), derived.assign(p=1)) if not x.empty]
    if not parts:
        return pd.DataFrame(columns=["end", "val", "filed"])
    both = pd.concat(parts)
    # prefer a directly reported quarter; otherwise the derived one
    return both.sort_values(["end", "p", "filed"]).drop_duplicates("end")[["end", "val", "filed"]]


def company_quarters(cik):
    facts = load_facts(cik)
    if not facts:
        return None
    gaap = facts.get("facts", {}).get("us-gaap", {})
    out = []
    for metric, tags in FLOWS.items():
        for prio, tag in enumerate(tags):
            rows = gaap.get(tag, {}).get("units", {}).get("USD", [])
            q = quarters_from_flows(first_filed(rows, instant=False))
            out.append(q.assign(metric=metric, tag=tag, prio=prio))
    for metric, tags in STOCKS.items():
        for prio, tag in enumerate(tags):
            rows = gaap.get(tag, {}).get("units", {}).get("USD", [])
            s = first_filed(rows, instant=True)
            if not s.empty:
                out.append(s.assign(metric=metric, tag=tag, prio=prio))
    # diluted weighted-average shares (all share classes) from the income statement;
    # averages cannot be differenced, so only directly reported 3-month values are used
    for tag in ["WeightedAverageNumberOfDilutedSharesOutstanding",
                "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"]:
        rows = gaap.get(tag, {}).get("units", {}).get("shares", [])
        d = first_filed(rows, instant=False)
        if not d.empty:
            d = d[(d["days"] >= Q_DAYS[0]) & (d["days"] <= Q_DAYS[1]) & (d["val"] > 0)]
            if not d.empty:
                out.append(d[["end", "val", "filed"]].assign(metric="diluted_shares", tag=tag, prio=0))
                break
    shares = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {})
    rows = shares.get("units", {}).get("shares", [])
    if rows:
        s = pd.DataFrame(rows)
        s = s[s["form"].astype(str).str.startswith(("10-Q", "10-K"))].copy()
        s["end"] = pd.to_datetime(s["end"]); s["filed"] = pd.to_datetime(s["filed"])
        # several share classes can be listed for the same date: add them up
        s = s.groupby(["end", "filed"], as_index=False)["val"].sum()
        s = s.sort_values("filed").drop_duplicates("end")
        out.append(s.assign(metric="shares_outstanding", tag="dei", prio=0))
    q = pd.concat([o for o in out if not o.empty], ignore_index=True)
    return q


# ----------------------------------------------------------------------------- features
def value_at(q, metric, end, cutoff, tag=None, tol=6):
    """Value of `metric` for the period ending near `end`, only if filed before `cutoff`."""
    m = q[(q["metric"] == metric) & ((q["end"] - end).abs().dt.days <= tol) & (q["filed"] < cutoff)]
    if tag is not None:
        m = m[m["tag"] == tag]
    if m.empty:
        return np.nan, None, None
    r = m.sort_values("prio").iloc[0]
    return r["val"], r["tag"], r["filed"]


def growth(q, metric, end, cutoff):
    v1, tag, f1 = value_at(q, metric, end, cutoff)
    if tag is None:
        return np.nan, None
    v0, _, _ = value_at(q, metric, end - pd.Timedelta(days=364), cutoff, tag=tag, tol=10)
    if np.isnan(v0) or v0 <= 0:
        return np.nan, f1
    return v1 / v0 - 1, f1


def margin(q, num, end, cutoff):
    rev, rtag, f = value_at(q, "revenue", end, cutoff)
    if np.isnan(rev) or rev <= 0:
        return np.nan
    if num == "gross":
        gp, _, _ = value_at(q, "gross_profit", end, cutoff)
        if np.isnan(gp):
            cost, _, _ = value_at(q, "cost_of_revenue", end, cutoff)
            gp = rev - cost if not np.isnan(cost) else np.nan
        return gp / rev
    oi, _, _ = value_at(q, "operating_income", end, cutoff)
    return oi / rev


def features_for_company(q, events):
    """events: this company's rows (report_date, close_before). Returns new feature columns."""
    rev = q[q["metric"] == "revenue"]
    period_ends = np.sort(rev["end"].unique())          # quarter-end dates (known in advance)
    res = []
    for i, e in events.iterrows():
        d = e["report_date"]
        r = {"idx": i}
        # fiscal period this report covers: latest quarter end before the report
        prior = period_ends[period_ends < np.datetime64(d)]
        if len(prior) and (d - pd.Timestamp(prior[-1])).days <= 120:
            r["reporting_lag_days"] = (d - pd.Timestamp(prior[-1])).days
            r["period_end"] = pd.Timestamp(prior[-1])
        # latest quarter whose revenue was FILED before the report
        known = rev[rev["filed"] < d]
        if known.empty:
            res.append(r); continue
        end = known["end"].max()
        used = []
        g_rev, f = growth(q, "revenue", end, d); used.append(f)
        g_inv, f2 = growth(q, "inventory", end, d); used.append(f2)
        g_rec, f3 = growth(q, "receivables", end, d); used.append(f3)
        r["revenue_growth_yoy"] = g_rev
        r["inventory_minus_sales_growth"] = g_inv - g_rev if not np.isnan(g_inv) else np.nan
        r["receivables_minus_sales_growth"] = g_rec - g_rev if not np.isnan(g_rec) else np.nan
        ly = end - pd.Timedelta(days=364)
        gm, gm0 = margin(q, "gross", end, d), margin(q, "gross", ly, d)
        om, om0 = margin(q, "op", end, d), margin(q, "op", ly, d)
        r["gross_margin_chg_yoy"] = gm - gm0
        r["op_margin_chg_yoy"] = om - om0
        r["op_margin"] = om
        ni, _, f4 = value_at(q, "net_income", end, d); used.append(f4)
        cfo, _, f5 = value_at(q, "cfo", end, d); used.append(f5)
        assets, _, f6 = value_at(q, "assets", end, d, tol=10); used.append(f6)
        equity, _, f7 = value_at(q, "equity", end, d, tol=10); used.append(f7)
        if assets and not np.isnan(assets) and assets > 0:
            r["accruals_to_assets"] = (ni - cfo) / assets if not (np.isnan(ni) or np.isnan(cfo)) else np.nan
            r["leverage"] = 1 - equity / assets if not np.isnan(equity) else np.nan
        for metric in ("diluted_shares", "shares_outstanding"):
            sh = q[(q["metric"] == metric) & (q["filed"] < d) & (q["val"] > 0)]
            if sh.empty or np.isnan(e["close_before"]):
                continue
            last = sh.sort_values("end").iloc[-1]
            mcap = last["val"] * e["close_before"]
            if 5e8 <= mcap <= 6e12:            # outside $0.5bn-$6trn = share-count error
                r["log_market_cap"] = np.log(mcap)
                used.append(last["filed"])
                break
        r["fundamentals_age_days"] = (d - end).days
        r["max_filed_used"] = max([u for u in used if u is not None])
        res.append(r)
    return pd.DataFrame(res).set_index("idx")


def main():
    feats = pd.read_csv(PROC / "features_v1.csv", parse_dates=["report_date"])
    ciks = pd.read_csv(PROC / "company_ciks.csv", dtype={"cik": str}) \
        if (PROC / "company_ciks.csv").exists() else None
    if ciks is None:
        comp = pd.read_csv(PROC / "universe_companies.csv", dtype={"cik": str})
        ciks = comp[["company_key", "cik"]].dropna()
        ciks["cik"] = ciks["cik"].str.split(".").str[0].str.zfill(10)
        print("company_ciks.csv not found - using S&P 500 CIKs only (run pull_sec.py first)")
    cikmap = dict(zip(ciks["company_key"], ciks["cik"]))

    # the price before each report (for market value), recomputed the same way as step 1
    ev = pd.read_csv(PROC / "earnings_events.csv", parse_dates=["report_date"])
    feats = feats.merge(ev[["company_key", "report_date", "ticker_used"]].drop_duplicates(
        ["company_key", "report_date"]), on=["company_key", "report_date"], how="left")
    feats["close_before"] = np.nan
    for t, grp in feats.groupby("ticker_used"):
        files = sorted((RAW / "yfinance_prices" / str(t)).glob("*.csv"))
        if not files:
            continue
        p = pd.read_csv(files[-1])
        p["date"] = pd.to_datetime(p.iloc[:, 0]).dt.tz_localize(None).dt.normalize()
        p = p.dropna(subset=["Close"]).sort_values("date").reset_index(drop=True)
        # yfinance prices are split-adjusted for EVERY split, including ones after the price
        # window; SEC share counts are not. Undo all splits after each date, using the full
        # split history when pull_data.py has saved it, else the splits inside the window.
        sfiles = sorted((RAW / "yfinance_splits" / str(t)).glob("*.csv"))
        if sfiles:
            sp = pd.read_csv(sfiles[-1])
            sp["date"] = pd.to_datetime(sp["date"], utc=True).dt.tz_localize(None).dt.normalize()
            sp = sp[sp["ratio"] > 0]
            sp = sp.sort_values("date")
            # product of all split ratios strictly after each date (vectorised)
            tail = np.append(np.cumprod(sp["ratio"].values[::-1])[::-1], 1.0)
            idx = np.searchsorted(sp["date"].values, p["date"].values, side="right")
            later = pd.Series(tail[idx], index=p.index)
        else:
            split = p.get("Stock Splits", pd.Series(0, index=p.index)).fillna(0)
            split = split.where(split > 0, 1.0)
            later = split[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
        raw_close = pd.to_numeric(p["Close"], errors="coerce") * later.astype(float)
        k = np.searchsorted(p["date"].values, grp["report_date"].values, side="left") - 1
        vals = np.where(k >= 0, raw_close.values[np.clip(k, 0, None)], np.nan).astype(float)
        feats.loc[grp.index, "close_before"] = vals

    all_q, parts, missing = [], [], []
    for key, grp in feats.groupby("company_key"):
        cik = cikmap.get(key)
        q = company_quarters(cik) if isinstance(cik, str) else None
        if q is None or q.empty:
            missing.append(key); continue
        all_q.append(q.assign(company_key=key))
        parts.append(features_for_company(q, grp))
    new = pd.concat(parts)
    out = feats.join(new)
    ev_sorted = out.sort_values(["company_key", "report_date"])
    lag_ly = ev_sorted.groupby("company_key")["reporting_lag_days"].shift(4)
    gap_ok = (ev_sorted["report_date"] - ev_sorted.groupby("company_key")["report_date"].shift(4)).dt.days.between(300, 430)
    out["reporting_lag_change_vs_ly"] = (ev_sorted["reporting_lag_days"] - lag_ly).where(gap_ok)

    # ---- leakage tests
    print("\nLEAKAGE TESTS (each must find 0 problems)")
    tests = {
        "SEC value filed on/after the report date": int((out["max_filed_used"] >= out["report_date"]).sum()),
        "row count differs from features_v1": abs(len(out) - len(feats)),
    }
    for k, v in tests.items():
        print(f"  {v:>5}  {k}")
    same_q = int((out["fundamentals_age_days"] < 45).sum())
    print(f"  (info) {same_q} reports where the current quarter's 10-Q was already filed "
          f"before the announcement - allowed, since it was public")

    new_cols = [
        ("revenue_growth_yoy", "C", "Revenue growth vs the same quarter a year earlier (latest quarter filed before the report)"),
        ("gross_margin_chg_yoy", "C", "Change in gross margin vs the same quarter a year earlier"),
        ("op_margin_chg_yoy", "C", "Change in operating margin vs the same quarter a year earlier"),
        ("op_margin", "C", "Operating margin in the latest filed quarter"),
        ("inventory_minus_sales_growth", "C", "Inventory growth minus revenue growth, year over year"),
        ("receivables_minus_sales_growth", "C", "Receivables growth minus revenue growth, year over year"),
        ("accruals_to_assets", "C", "(Net income - operating cash flow) / total assets, latest filed quarter"),
        ("leverage", "C", "1 - shareholders' equity / total assets"),
        ("log_market_cap", "C", "Log of shares outstanding (latest filing) x share price before the report"),
        ("fundamentals_age_days", "C", "Days between the end of the latest filed quarter and the report"),
        ("reporting_lag_days", "E", "Days from the end of the quarter being reported to the report date"),
        ("reporting_lag_change_vs_ly", "E", "Change in reporting lag vs the same quarter last year (later than usual > 0)"),
    ]
    dic = pd.read_csv(PROC / "feature_dictionary.csv")
    dic = dic[~dic["feature"].isin([c[0] for c in new_cols])]
    add = pd.DataFrame([{"feature": n, "group": g, "definition": d,
                         "available": "SEC filings dated before the report" if g == "C"
                         else "quarter-end date and report schedule"} for n, g, d in new_cols])
    pd.concat([dic, add]).to_csv(PROC / "feature_dictionary.csv", index=False)

    names = [c[0] for c in new_cols]
    table = out.drop(columns=["ticker_used", "close_before", "max_filed_used", "period_end"], errors="ignore")
    table.to_csv(PROC / "features_v2.csv", index=False, date_format="%Y-%m-%d")
    pd.concat(all_q).to_csv(PROC / "fundamentals_quarterly.csv", index=False, date_format="%Y-%m-%d")

    base = table["not_beat"].mean()
    lines = [f"New features - not-beat rate by quintile (overall {base:.1%})\n",
             f"{'feature':32} {'Q1 (low)':>9} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'Q5 (high)':>9} {'spread':>7} {'coverage':>9}"]
    rows = []
    for n in names:
        x = table[n]
        if x.notna().sum() < 100 or x.nunique() < 5:
            continue
        qn = pd.qcut(x.rank(method="first"), 5, labels=False)
        r = table.groupby(qn)["not_beat"].mean()
        rows.append((abs(r.iloc[-1] - r.iloc[0]), n, r, x.notna().mean()))
    for _, n, r, cov in sorted(rows, reverse=True):
        lines.append(f"{n:32} " + " ".join(f"{v:>7.1%}" for v in r.values)
                     + f" {r.iloc[-1] - r.iloc[0]:>+8.1%} {cov:>8.0%}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "feature_quintiles_v2.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nCompanies with SEC data: {feats['company_key'].nunique() - len(missing)} of "
          f"{feats['company_key'].nunique()}" + (f"  (missing: {len(missing)})" if missing else ""))
    print(f"Saved {len(table)} rows to data/processed/features_v2.csv")


if __name__ == "__main__":
    main()
