-- 001_schema.sql : tables for the Earnings Predictor database (PostgreSQL 15+)
-- Safe to run repeatedly: every object is created only if it does not exist yet.

-- One row per company in the point-in-time universe. company_key is today's ticker
-- (renamed companies are merged under it, e.g. FB -> META).
CREATE TABLE IF NOT EXISTS companies (
    company_id     SERIAL PRIMARY KEY,
    company_key    TEXT UNIQUE NOT NULL,
    sector         TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN
                     ('current member', 'left after 2025', 'left during window')),
    cik            CHAR(10),                 -- SEC company ID (current members only for now)
    yahoo_ticker   TEXT,
    ticker_used    TEXT,                     -- ticker the data was actually found under
    tickers_used   TEXT,                     -- every index ticker the company had, e.g. 'FB,META'
    has_data       BOOLEAN NOT NULL DEFAULT FALSE,
    expected_quarters INT                    -- approx. quarters in the index, 2018-2025
);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS expected_quarters INT;

-- When each company was in the S&P 500 (valid_to NULL = still a member at the end of 2025).
CREATE TABLE IF NOT EXISTS universe_membership (
    company_id  INT  NOT NULL REFERENCES companies,
    ticker      TEXT NOT NULL,               -- ticker in the index during this spell
    valid_from  DATE NOT NULL,
    valid_to    DATE,
    PRIMARY KEY (company_id, ticker, valid_from),
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);

-- NYSE trading days (taken from SPY's price history), numbered 1, 2, 3 ... so that
-- "two trading days later" is simply td + 2.
CREATE TABLE IF NOT EXISTS trading_days (
    trade_date DATE PRIMARY KEY,
    td         INT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_prices (
    company_id   INT  NOT NULL REFERENCES companies,
    trade_date   DATE NOT NULL,
    open         NUMERIC(14,4),
    high         NUMERIC(14,4),
    low          NUMERIC(14,4),
    close        NUMERIC(14,4),              -- raw close: use for per-share ratios
    adj_close    DOUBLE PRECISION,           -- split/dividend adjusted: use for returns
    volume       BIGINT CHECK (volume >= 0),
    dividend     NUMERIC(12,4) DEFAULT 0,
    split_ratio  DOUBLE PRECISION DEFAULT 0,
    source       TEXT NOT NULL,
    retrieved_at DATE NOT NULL,
    PRIMARY KEY (company_id, trade_date)
);

-- SPY and sector ETFs, used to measure market-adjusted reactions.
CREATE TABLE IF NOT EXISTS benchmark_prices (
    symbol       TEXT NOT NULL,              -- SPY, XLK, XLY, XLP, XLC
    trade_date   DATE NOT NULL,
    close        NUMERIC(14,4),
    adj_close    DOUBLE PRECISION,
    retrieved_at DATE NOT NULL,
    PRIMARY KEY (symbol, trade_date)
);

-- One row per earnings announcement per source. EPS is the vendor's adjusted ("Street")
-- EPS, rounded to the cent. Derived values (beat, surprise, day 0) live in views.
CREATE TABLE IF NOT EXISTS earnings_events (
    event_id        BIGSERIAL PRIMARY KEY,
    company_id      INT  NOT NULL REFERENCES companies,
    report_date     DATE NOT NULL,
    report_datetime TIMESTAMP,               -- New York local time, as given by the vendor
    report_time     TEXT NOT NULL DEFAULT 'UNKNOWN'
                    CHECK (report_time IN ('BMO', 'AMC', 'DMH', 'UNKNOWN')),
    eps_estimate    NUMERIC(10,4),
    eps_actual      NUMERIC(10,4),
    in_universe     BOOLEAN NOT NULL,        -- index member on the report date, 2018-2025
    source          TEXT NOT NULL,           -- 'yfinance'
    retrieved_at    DATE NOT NULL,
    UNIQUE (company_id, report_date, source)
);
CREATE INDEX IF NOT EXISTS earnings_events_date_idx ON earnings_events (report_date);

-- Alpha Vantage vs yfinance comparison, one row per compared quarter.
CREATE TABLE IF NOT EXISTS vendor_crosscheck (
    company_id        INT  NOT NULL REFERENCES companies,
    fiscal_period_end DATE,
    av_report_date    DATE NOT NULL,
    yf_report_date    DATE,
    date_gap_days     INT,
    av_estimate       NUMERIC(10,4),
    yf_estimate       NUMERIC(10,4),
    av_actual         NUMERIC(10,4),
    yf_actual         NUMERIC(10,4),
    issue             TEXT,
    PRIMARY KEY (company_id, av_report_date)
);

-- A log of every load, so you can see when the data was last refreshed.
CREATE TABLE IF NOT EXISTS etl_runs (
    run_id      BIGSERIAL PRIMARY KEY,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    notes       TEXT
);
