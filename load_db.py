#!/usr/bin/env python3
"""
load_db.py - load the pulled data into PostgreSQL, check it, and run the Step 6 queries.

What it does
  1. creates the database "earnings" if it doesn't exist (Postgres.app must be running)
  2. creates the tables (sql/001_schema.sql) and views (sql/010_views.sql)
  3. loads companies, membership, prices, benchmarks, earnings events, vendor cross-check
     - each table is loaded into a temporary staging table, then upserted
       (INSERT ... ON CONFLICT DO UPDATE), so running it again never creates duplicates
  4. runs the data-quality checks (sql/900_checks.sql) - each should find 0 problems
  5. runs the Step 6 exploration queries (sql/step6_exploration.sql) and prints results

Setup (once, inside the .venv):
    pip install "psycopg[binary]"
Run from the project folder:
    python load_db.py

Connection: by default it connects to Postgres.app on this Mac as your user, with no
password. To use another server, set DATABASE_URL, e.g.
    export DATABASE_URL="postgresql://user:password@host:5432/earnings"
"""

import io
import os
import re
from pathlib import Path

import pandas as pd

try:
    import psycopg
except ImportError:
    raise SystemExit('psycopg is not installed - run: pip install "psycopg[binary]"')

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SQL = ROOT / "sql"
DB_NAME = "earnings"
BENCHMARKS = ["SPY", "XLK", "XLY", "XLP", "XLC"]
ALT_TICKERS = {"PARA": ["PSKY"], "GPS": ["GAP"], "WYND": ["TNL"]}   # same as pull_data.py


# ----------------------------------------------------------------------------- connection
def connect():
    url = os.getenv("DATABASE_URL")
    if url:
        return psycopg.connect(url)
    try:
        admin = psycopg.connect("host=localhost dbname=postgres", autocommit=True)
    except psycopg.OperationalError as e:
        raise SystemExit("Could not connect to PostgreSQL. Is Postgres.app open, with the "
                         f"elephant icon in the menu bar?\n  ({e})")
    exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,)).fetchone()
    if not exists:
        admin.execute(f'CREATE DATABASE "{DB_NAME}"')
        print(f'created database "{DB_NAME}"')
    admin.close()
    return psycopg.connect(f"host=localhost dbname={DB_NAME}")


def run_file(cur, name):
    cur.execute((SQL / name).read_text())


# ----------------------------------------------------------------------------- loading
def upsert(cur, table, df, key_cols):
    """Copy df into a temporary staging table, then insert-or-update into `table`."""
    if df.empty:
        return 0
    cols = list(df.columns)
    stage = f"stage_{table}"
    cur.execute(f"DROP TABLE IF EXISTS {stage}")
    cur.execute(f"CREATE TEMP TABLE {stage} (LIKE {table} INCLUDING DEFAULTS)")
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    col_list = ", ".join(cols)
    with cur.copy(f"COPY {stage} ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
        cp.write(buf.read())
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key_cols)
    action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
    cur.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {stage} "
                f"ON CONFLICT ({', '.join(key_cols)}) {action}")
    return len(df)


def latest_file(source, ticker, ext="csv"):
    folder = RAW / source / str(ticker)
    files = sorted(folder.glob(f"*.{ext}")) if folder.exists() else []
    return files[-1] if files else None


def read_prices(path):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "trade_date", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Adj Close": "adj_close",
                            "Volume": "volume", "Dividends": "dividend",
                            "Stock Splits": "split_ratio"})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["retrieved_at"] = path.stem                          # file name is the download date
    for c in ["open", "high", "low", "close", "adj_close", "volume", "dividend", "split_ratio"]:
        if c not in df:
            df[c] = None
    df = df.dropna(subset=["adj_close"])
    df["volume"] = df["volume"].astype("Int64")
    return df


def main():
    comp = pd.read_csv(PROC / "universe_companies.csv", dtype={"cik": str})
    memb = pd.read_csv(PROC / "universe_membership.csv")
    cov = pd.read_csv(PROC / "pull_coverage.csv")
    events = pd.read_csv(PROC / "earnings_events.csv")

    with connect() as conn, conn.cursor() as cur:
        run_file(cur, "001_schema.sql")
        run_id = cur.execute("INSERT INTO etl_runs (notes) VALUES ('load_db.py') "
                             "RETURNING run_id").fetchone()[0]

        # --- companies
        c = comp.merge(cov[["company_key", "ticker_used", "events_with_estimate"]],
                       on="company_key", how="left")
        c["has_data"] = c["events_with_estimate"].fillna(0) > 0
        c["cik"] = c["cik"].map(lambda x: str(x).split(".")[0].zfill(10)
                                if pd.notna(x) and str(x).strip() else None)
        c["expected_quarters"] = c["approx_quarters"]
        c = c[["company_key", "sector", "status", "cik", "yahoo_ticker", "ticker_used",
               "tickers_used", "has_data", "expected_quarters"]]
        # companies that are no longer in the universe file (e.g. merged into another key)
        # are removed with all their rows, so a rebuilt universe never leaves stale data
        gone = [k for (k,) in cur.execute("SELECT company_key FROM companies").fetchall()
                if k not in set(c["company_key"])]
        for k in gone:
            cid = cur.execute("SELECT company_id FROM companies WHERE company_key = %s", (k,)).fetchone()[0]
            for t in ["vendor_crosscheck", "earnings_events", "daily_prices", "universe_membership"]:
                cur.execute(f"DELETE FROM {t} WHERE company_id = %s", (cid,))
            cur.execute("DELETE FROM companies WHERE company_id = %s", (cid,))
        if gone:
            print(f"removed companies no longer in the universe: {', '.join(gone)}")
        n = upsert(cur, "companies", c, ["company_key"])
        ids = dict(cur.execute("SELECT company_key, company_id FROM companies").fetchall())
        print(f"companies            {n:>8,}")

        # --- membership
        m = memb.assign(company_id=memb["company_key"].map(ids))
        m = m[["company_id", "ticker", "valid_from", "valid_to"]]
        print(f"universe_membership  {upsert(cur, 'universe_membership', m, ['company_id', 'ticker', 'valid_from']):>8,}")

        # --- benchmarks + trading days
        bench = []
        for sym in BENCHMARKS:
            f = latest_file("yfinance_prices", sym)
            if f:
                b = read_prices(f)
                b["symbol"] = sym
                bench.append(b[["symbol", "trade_date", "close", "adj_close", "retrieved_at"]])
        if not bench:
            raise SystemExit("No benchmark prices found - run pull_data.py first.")
        bench = pd.concat(bench)
        print(f"benchmark_prices     {upsert(cur, 'benchmark_prices', bench, ['symbol', 'trade_date']):>8,}")
        days = (bench.loc[bench["symbol"] == "SPY", ["trade_date"]].drop_duplicates()
                .sort_values("trade_date").reset_index(drop=True))
        days["td"] = days.index + 1
        cur.execute("DELETE FROM trading_days")          # renumbered from scratch each load
        print(f"trading_days         {upsert(cur, 'trading_days', days, ['trade_date']):>8,}")

        # --- company prices
        total, missing = 0, []
        for _, r in c.iterrows():
            tickers = [r["ticker_used"], r["yahoo_ticker"]] + ALT_TICKERS.get(r["company_key"], [])
            f = next((latest_file("yfinance_prices", t) for t in tickers
                      if isinstance(t, str) and latest_file("yfinance_prices", t)), None)
            if not f:
                missing.append(r["company_key"])
                continue
            p = read_prices(f)
            p["company_id"] = ids[r["company_key"]]
            p["source"] = "yfinance"
            p = p[["company_id", "trade_date", "open", "high", "low", "close", "adj_close",
                   "volume", "dividend", "split_ratio", "source", "retrieved_at"]]
            total += upsert(cur, "daily_prices", p, ["company_id", "trade_date"])
        print(f"daily_prices         {total:>8,}   (no prices for {len(missing)} companies)")

        # --- earnings events
        e = events.copy()
        e["company_id"] = e["company_key"].map(ids)
        e["source"] = "yfinance"
        e["retrieved_at"] = e["ticker_used"].map(
            lambda t: latest_file("yfinance_earnings", t).stem
            if latest_file("yfinance_earnings", t) else None)
        e = e[["company_id", "report_date", "report_datetime", "report_time", "eps_estimate",
               "eps_actual", "in_universe", "source", "retrieved_at"]]
        e = e.drop_duplicates(subset=["company_id", "report_date"])
        print(f"earnings_events      {upsert(cur, 'earnings_events', e, ['company_id', 'report_date', 'source']):>8,}")

        # --- vendor cross-check
        avf = PROC / "av_crosscheck.csv"
        if avf.exists():
            a = pd.read_csv(avf)
            a = a.rename(columns={"fiscalDateEnding": "fiscal_period_end",
                                  "report_date": "av_report_date", "yf_date": "yf_report_date",
                                  "eps_estimate": "yf_estimate", "eps_actual": "yf_actual"})
            a["company_id"] = a["company_key"].map(ids)
            a["fiscal_period_end"] = pd.to_datetime(a["fiscal_period_end"], errors="coerce").dt.date
            a["date_gap_days"] = a["date_gap_days"].astype("Int64")
            a = a.drop_duplicates(subset=["company_id", "av_report_date"])
            a = a[["company_id", "fiscal_period_end", "av_report_date", "yf_report_date",
                   "date_gap_days", "av_estimate", "yf_estimate", "av_actual", "yf_actual", "issue"]]
            print(f"vendor_crosscheck    {upsert(cur, 'vendor_crosscheck', a, ['company_id', 'av_report_date']):>8,}")

        run_file(cur, "010_views.sql")
        cur.execute("ANALYZE")                           # refresh planner statistics
        cur.execute("UPDATE etl_runs SET finished_at = now() WHERE run_id = %s", (run_id,))
        conn.commit()

        # --- data-quality checks
        line = "=" * 78
        print(f"\n{line}\nDATA-QUALITY CHECKS  (0 = clean)\n{line}")
        for title, query in split_queries((SQL / "900_checks.sql").read_text(), "check:"):
            n = cur.execute(f"SELECT COUNT(*) FROM ({query}) q").fetchone()[0]
            print(f"  {n:>6,}  {title}")

        # --- Step 6 exploration
        print(f"\n{line}\nSTEP 6 EXPLORATION\n{line}")
        for title, query in split_queries((SQL / "step6_exploration.sql").read_text(), "Q"):
            print(f"\n{title}")
            cur.execute(query)
            df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
            print(df.to_string(index=False))


def split_queries(text, marker):
    """Split a .sql file into (title, query) pairs using the comment line before each query."""
    out, title, body = [], None, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("--"):
            c = s.lstrip("-").strip()
            if c.startswith(marker):
                if title and body:
                    out.append((title, "\n".join(body).rstrip().rstrip(";")))
                title, body = c.replace("check:", "").strip(), []
            elif title and body:
                continue          # comment inside/after a query title block
            elif title:
                title += " " + c  # continuation of a multi-line title
            continue
        if title is not None and s:
            body.append(line)
    if title and body:
        out.append((title, "\n".join(body).rstrip().rstrip(";")))
    return out


if __name__ == "__main__":
    main()
