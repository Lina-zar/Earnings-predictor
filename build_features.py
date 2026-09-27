#!/usr/bin/env python3
"""
build_features.py - Part 2, step 1: one row per labelled quarter, features built only from
information available before the earnings report.

TIMING RULES (every feature obeys all three; the leakage tests at the end check them)
  1. Prices: only closes up to the trading day BEFORE the report date.
  2. Earnings history: only this company's earlier reports (report date < this report date).
  3. Peers: only companies that reported strictly earlier than this company.
  (SEC fundamentals come in step 2 and will follow the same rule: filed before the report.)

Target
  not_beat = 1 if reported EPS <= consensus estimate (rounded to the cent), else 0.
  outcome  = 'beat' / 'in line' / 'miss', kept for analysis.

Feature groups built here
  A. Track record         beat rates, streak, shrunk beat rate, past surprises (scaled by price)
  B. Estimate demand      estimate vs same quarter last year and vs last quarter
  D. Market signals       20/60-day return vs SPY, volatility, distance from 52-week high,
                          trading-volume size proxy, reaction to the previous report
  E. Timing and context   before/after market, report month, days since the last report
  F. Earnings season      share of same-sector (and all) companies that beat earlier this season

Outputs
  data/processed/features_v1.csv          feature table (5,380 rows expected)
  data/processed/feature_dictionary.csv   name, group, definition, available when
  reports/feature_quintiles.txt           not-beat rate by quintile for every feature

Run from the project folder:  python build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

SHRINK_K = 4           # pseudo-quarters of sector history mixed into a company's beat rate
SEASON_DAYS = 45       # "this season" = reports in the 45 days before this report
EXPECTED_ROWS = 5380

DICTIONARY = []


def feature(name, group, definition, available):
    DICTIONARY.append({"feature": name, "group": group, "definition": definition,
                       "available": available})


# ----------------------------------------------------------------------------- prices
def load_prices(ticker):
    files = sorted((RAW / "yfinance_prices" / str(ticker)).glob("*.csv"))
    if not files:
        return None
    p = pd.read_csv(files[-1])
    p["date"] = pd.to_datetime(p.iloc[:, 0]).dt.tz_localize(None).dt.normalize()
    p = p.dropna(subset=["Adj Close"]).sort_values("date").drop_duplicates("date")
    return p.set_index("date")[["Adj Close", "Close", "Volume"]]


def market_features(ev, spy):
    """Group D for every event of one company. Uses closes strictly before the report date."""
    ticker = ev["ticker_used"].iloc[0]
    px = load_prices(ticker)
    out = pd.DataFrame(index=ev.index)
    cols = ["px_cutoff_date", "ret_20d_vs_spy", "ret_60d_vs_spy", "vol_60d",
            "dist_52w_high", "log_dollar_volume_60d", "close_before", "reaction_2d"]
    for c in cols:
        out[c] = np.nan
    out["px_cutoff_date"] = pd.NaT
    if px is None or len(px) < 30:
        return out
    dates = px.index.values
    spy_adj = spy.reindex(px.index)
    for i, r in ev.iterrows():
        k = np.searchsorted(dates, np.datetime64(r["report_date"]), side="left") - 1
        if k < 0:
            continue
        cut = px.index[k]
        if (r["report_date"] - cut).days > 7:        # no recent price (e.g. not yet listed)
            continue
        out.at[i, "px_cutoff_date"] = cut
        out.at[i, "close_before"] = px["Close"].iloc[k]
        a = px["Adj Close"].values
        s = spy_adj.values
        for n, col in ((20, "ret_20d_vs_spy"), (60, "ret_60d_vs_spy")):
            if k - n >= 0 and s[k - n] > 0:
                out.at[i, col] = (a[k] / a[k - n] - 1) - (s[k] / s[k - n] - 1)
        if k >= 60:
            rets = np.diff(np.log(a[k - 60:k + 1]))
            out.at[i, "vol_60d"] = rets.std() * np.sqrt(252)
            dv = (px["Close"].values[k - 59:k + 1] * px["Volume"].values[k - 59:k + 1]).mean()
            out.at[i, "log_dollar_volume_60d"] = np.log(dv) if dv > 0 else np.nan
        if k >= 20:
            hi = a[max(0, k - 251):k + 1].max()
            out.at[i, "dist_52w_high"] = a[k] / hi - 1
        # this event's own 2-day reaction (used ONLY lagged, as last quarter's reaction):
        # day 0 = report day (k+1) for before-open, next trading day (k+2) for after-close;
        # reaction = close before day 0 -> close of day +1, minus SPY over the same days
        d0 = k + 2 if r["report_time"] == "AMC" else k + 1
        if d0 + 1 < len(a) and not np.isnan(s[d0 - 1]) and not np.isnan(s[d0 + 1]):
            out.at[i, "reaction_2d"] = (a[d0 + 1] / a[d0 - 1] - 1) - (s[d0 + 1] / s[d0 - 1] - 1)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ev = pd.read_csv(PROC / "earnings_events.csv", parse_dates=["report_date"])
    comp = pd.read_csv(PROC / "universe_companies.csv")
    ev = ev.merge(comp[["company_key", "sector"]], on="company_key", how="left")
    ev = ev.sort_values(["company_key", "report_date"]).reset_index(drop=True)

    has_label = ev["eps_estimate"].notna() & ev["eps_actual"].notna()
    ev["beat"] = np.where(has_label, (ev["eps_actual"] > ev["eps_estimate"]).astype(float), np.nan)
    ev["inline"] = np.where(has_label, (ev["eps_actual"] == ev["eps_estimate"]).astype(float), np.nan)

    # ---- D first: we need the price before each report to scale surprises
    spy = load_prices("SPY")["Adj Close"]
    print("computing market features ...")
    mk = pd.concat([market_features(g, spy) for _, g in ev.groupby("company_key")])
    ev = ev.join(mk)
    ev["surprise_ps"] = (ev["eps_actual"] - ev["eps_estimate"]) / ev["close_before"]

    g = ev.groupby("company_key")

    # ---- A. Track record (shift(1) = only earlier reports)
    prev_beat = g["beat"].shift(1)
    ev["beat_rate_4q"] = prev_beat.groupby(ev["company_key"]).transform(
        lambda s: s.rolling(4, min_periods=2).mean())
    ev["beat_rate_8q"] = prev_beat.groupby(ev["company_key"]).transform(
        lambda s: s.rolling(8, min_periods=4).mean())
    ev["inline_rate_8q"] = g["inline"].shift(1).groupby(ev["company_key"]).transform(
        lambda s: s.rolling(8, min_periods=4).mean())
    n8 = prev_beat.groupby(ev["company_key"]).transform(lambda s: s.rolling(8, min_periods=1).count())
    b8 = prev_beat.groupby(ev["company_key"]).transform(lambda s: s.rolling(8, min_periods=1).sum())

    def streak(s):
        out, run = [], 0
        for v in s:
            out.append(run)
            run = run + 1 if v == 1 else 0
        return pd.Series(out, index=s.index)
    ev["beat_streak"] = g["beat"].transform(streak)

    # sector prior: beat rate of all same-sector reports strictly before this date
    ev = ev.sort_values("report_date")
    sec_rate = []
    for sector, grp in ev.groupby("sector"):
        daily = grp.dropna(subset=["beat"]).groupby("report_date")["beat"].agg(["sum", "count"])
        cum = daily.cumsum().shift(1)                  # strictly earlier dates only
        rate = (cum["sum"] / cum["count"]).reindex(grp["report_date"].values, method="ffill")
        sec_rate.append(pd.Series(rate.values, index=grp.index))
    ev["sector_prior"] = pd.concat(sec_rate)
    ev = ev.sort_values(["company_key", "report_date"])
    ev["beat_rate_shrunk"] = (b8 + SHRINK_K * ev["sector_prior"]) / (n8 + SHRINK_K)

    ev["last_surprise_ps"] = g["surprise_ps"].shift(1)
    prev_sp = g["surprise_ps"].shift(1)
    ev["mean_surprise_ps_4q"] = prev_sp.groupby(ev["company_key"]).transform(
        lambda s: s.rolling(4, min_periods=2).mean())
    ev["std_surprise_ps_8q"] = prev_sp.groupby(ev["company_key"]).transform(
        lambda s: s.rolling(8, min_periods=4).std())

    # ---- B. How demanding the estimate is
    lag4_act = g["eps_actual"].shift(4)
    lag4_date = g["report_date"].shift(4)
    ok4 = (ev["report_date"] - lag4_date).dt.days.between(300, 430)
    ev["est_vs_yoy_actual_ps"] = np.where(ok4, (ev["eps_estimate"] - lag4_act) / ev["close_before"], np.nan)
    ev["est_growth_yoy"] = np.where(ok4 & (lag4_act.abs() >= 0.05),
                                    ((ev["eps_estimate"] - lag4_act) / lag4_act.abs()).clip(-2, 2), np.nan)
    lag1_act = g["eps_actual"].shift(1)
    ev["est_vs_last_actual_ps"] = (ev["eps_estimate"] - lag1_act) / ev["close_before"]
    ev["est_negative"] = (ev["eps_estimate"] < 0).astype(float)

    # ---- D. lagged reaction
    ev["prev_reaction_2d"] = g["reaction_2d"].shift(1)

    # ---- E. Timing and context
    ev["is_bmo"] = (ev["report_time"] == "BMO").astype(float)
    ev["report_month"] = ev["report_date"].dt.month
    ev["days_since_last_report"] = (ev["report_date"] - g["report_date"].shift(1)).dt.days
    gap_lastyear = (g["report_date"].shift(4) - g["report_date"].shift(5)).dt.days
    ev["report_gap_change_vs_ly"] = np.where(ok4, ev["days_since_last_report"] - gap_lastyear, np.nan)

    # ---- F. Earnings season so far (strictly earlier reports, last SEASON_DAYS days)
    lab = ev.dropna(subset=["beat"])[["report_date", "sector", "beat"]]
    by_day_sec = lab.groupby(["sector", "report_date"])["beat"].agg(["sum", "count"])
    by_day_all = lab.groupby("report_date")["beat"].agg(["sum", "count"])

    def season(dates, table):
        t = table.sort_index()
        cs = t.cumsum()
        idx = t.index.values
        res_rate, res_n, res_last = [], [], []
        for d in dates:
            hi = np.searchsorted(idx, np.datetime64(d), side="left") - 1          # < d
            lo = np.searchsorted(idx, np.datetime64(d - pd.Timedelta(days=SEASON_DAYS)), side="left") - 1
            if hi < 0 or hi <= lo:
                res_rate.append(np.nan); res_n.append(0); res_last.append(pd.NaT); continue
            s = cs["sum"].iloc[hi] - (cs["sum"].iloc[lo] if lo >= 0 else 0)
            n = cs["count"].iloc[hi] - (cs["count"].iloc[lo] if lo >= 0 else 0)
            res_rate.append(s / n if n else np.nan); res_n.append(n); res_last.append(t.index[hi])
        return res_rate, res_n, res_last

    ev["season_sector_beat_rate"] = np.nan
    ev["season_sector_n"] = 0
    ev["season_last_peer_date"] = pd.NaT
    for sector, grp in ev.groupby("sector"):
        rate, n, last = season(grp["report_date"], by_day_sec.loc[sector])
        ev.loc[grp.index, "season_sector_beat_rate"] = rate
        ev.loc[grp.index, "season_sector_n"] = n
        ev.loc[grp.index, "season_last_peer_date"] = last
    rate, n, _ = season(ev["report_date"], by_day_all)
    ev["season_all_beat_rate"] = rate
    ev["season_all_n"] = n

    # ---- keep labelled, in-universe quarters
    out = ev[ev["in_universe"].astype(bool) & ev["beat"].notna()].copy()
    out["not_beat"] = (1 - out["beat"]).astype(int)
    out["outcome"] = np.select([out["eps_actual"] > out["eps_estimate"],
                                out["eps_actual"] == out["eps_estimate"]], ["beat", "in line"], "miss")

    features = [
        ("beat_rate_4q", "A", "Share of the previous 4 reports that beat consensus", "prior reports"),
        ("beat_rate_8q", "A", "Share of the previous 8 reports that beat consensus", "prior reports"),
        ("beat_rate_shrunk", "A", f"8-quarter beat rate blended with the sector's prior beat rate ({SHRINK_K} pseudo-quarters)", "prior reports"),
        ("inline_rate_8q", "A", "Share of the previous 8 reports that exactly matched consensus", "prior reports"),
        ("beat_streak", "A", "Number of consecutive beats immediately before this report", "prior reports"),
        ("last_surprise_ps", "A", "Last quarter's surprise (actual - estimate) divided by the share price before that report", "prior reports"),
        ("mean_surprise_ps_4q", "A", "Average price-scaled surprise over the previous 4 reports", "prior reports"),
        ("std_surprise_ps_8q", "A", "Volatility of price-scaled surprises over the previous 8 reports", "prior reports"),
        ("est_vs_yoy_actual_ps", "B", "Consensus estimate minus actual EPS of the same quarter last year, divided by price", "consensus + prior reports"),
        ("est_growth_yoy", "B", "Implied EPS growth vs the same quarter last year (capped at +/-200%)", "consensus + prior reports"),
        ("est_vs_last_actual_ps", "B", "Consensus estimate minus last quarter's actual EPS, divided by price", "consensus + prior reports"),
        ("est_negative", "B", "1 if analysts expect a loss", "consensus"),
        ("ret_20d_vs_spy", "D", "Stock return minus S&P 500 return over the 20 trading days before the report", "close of the day before the report"),
        ("ret_60d_vs_spy", "D", "Same over 60 trading days", "close of the day before the report"),
        ("vol_60d", "D", "Annualised volatility of daily returns over the previous 60 trading days", "close of the day before the report"),
        ("dist_52w_high", "D", "Price relative to its 52-week high (0 = at the high)", "close of the day before the report"),
        ("log_dollar_volume_60d", "D", "Log of average daily dollar trading volume over 60 days (size proxy)", "close of the day before the report"),
        ("prev_reaction_2d", "D", "Stock reaction vs S&P 500 to the previous earnings report (2 days)", "prior reports"),
        ("is_bmo", "E", "1 if reported before the market opens", "report schedule"),
        ("report_month", "E", "Calendar month of the report", "report schedule"),
        ("days_since_last_report", "E", "Days since the company's previous earnings report", "report schedule"),
        ("report_gap_change_vs_ly", "E", "Change in that gap vs the same report last year (later than usual > 0)", "report schedule"),
        ("season_sector_beat_rate", "F", f"Share of same-sector companies that beat in the {SEASON_DAYS} days before this report", "peers' earlier reports"),
        ("season_sector_n", "F", "Number of same-sector reports counted above", "peers' earlier reports"),
        ("season_all_beat_rate", "F", f"Share of all universe companies that beat in the {SEASON_DAYS} days before", "peers' earlier reports"),
        ("season_all_n", "F", "Number of reports counted above", "peers' earlier reports"),
        ("sector_prior", "A", "Beat rate of all earlier same-sector reports since 2000", "peers' earlier reports"),
    ]
    for f in features:
        feature(*f)
    names = [f[0] for f in features]
    keep = ["company_key", "sector", "report_date", "report_time", "eps_estimate",
            "not_beat", "outcome"] + names
    table = out[keep]

    # ---- leakage tests
    print("\nLEAKAGE TESTS (each must find 0 problems)")
    problems = {}
    problems["price cutoff on/after report date"] = int((out["px_cutoff_date"] >= out["report_date"]).sum())
    problems["peer counted on/after report date"] = int((out["season_last_peer_date"] >= out["report_date"]).sum())
    banned = {"eps_actual", "surprise", "surprise_ps", "beat", "inline", "reaction_2d"}
    problems["outcome columns among features"] = len(banned & set(names))
    # recompute beat_rate_4q the slow, obvious way for 300 random rows
    sample = out.sample(min(300, len(out)), random_state=0)
    bad = 0
    for _, r in sample.iterrows():
        hist = ev[(ev["company_key"] == r["company_key"]) & (ev["report_date"] < r["report_date"])]
        last4 = hist["beat"].tail(4)
        exp = last4.mean() if last4.notna().sum() >= 2 else np.nan
        got = r["beat_rate_4q"]
        if not ((np.isnan(exp) and np.isnan(got)) or np.isclose(exp, got)):
            bad += 1
    problems["beat_rate_4q uses a later report (300-row recheck)"] = bad
    problems[f"row count differs from {EXPECTED_ROWS}"] = abs(len(table) - EXPECTED_ROWS)
    for k, v in problems.items():
        print(f"  {v:>5}  {k}")

    # ---- save
    table.to_csv(PROC / "features_v1.csv", index=False, date_format="%Y-%m-%d")
    pd.DataFrame(DICTIONARY).to_csv(PROC / "feature_dictionary.csv", index=False)

    # ---- first look at the signal: not-beat rate by quintile
    base = table["not_beat"].mean()
    lines = [f"Not-beat rate by feature quintile (overall {base:.1%}, {len(table)} quarters)\n",
             f"{'feature':28} {'Q1 (low)':>9} {'Q2':>7} {'Q3':>7} {'Q4':>7} {'Q5 (high)':>9} {'spread':>7} {'coverage':>9}"]
    spreads = []
    for n in names:
        x = table[n]
        if x.nunique() < 5:
            continue
        q = pd.qcut(x.rank(method="first"), 5, labels=False)
        r = table.groupby(q)["not_beat"].mean()
        if len(r) < 5:
            continue
        spreads.append((abs(r.iloc[4] - r.iloc[0]), n, r, x.notna().mean()))
    for spread, n, r, cov in sorted(spreads, reverse=True):
        lines.append(f"{n:28} " + " ".join(f"{v:>7.1%}" for v in r.values)
                     + f" {r.iloc[4] - r.iloc[0]:>+8.1%} {cov:>8.0%}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "feature_quintiles.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSaved {len(table)} rows x {len(names)} features to data/processed/features_v1.csv")
    print("Saved data/processed/feature_dictionary.csv and reports/feature_quintiles.txt")


if __name__ == "__main__":
    main()
