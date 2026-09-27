# Feature dictionary

One row per earnings report (5,380 reports, 2018–2025). **Every feature uses only information available before the report:**

1. **Prices:** closes up to the trading day *before* the report date.
2. **Earnings history:** only the company's earlier reports.
3. **Peers:** only companies that reported strictly earlier.
4. **SEC filings:** only values filed before the report date (in practice, the previous quarter's 10-Q/10-K), always the *first-filed* version so later restatements never leak in.

Automated leakage tests in `build_features.py` and `build_fundamentals.py` check these rules on every run.

**Target:** `not_beat` = 1 if reported EPS ≤ consensus estimate (rounded to the cent), else 0. An exact match counts as not beating.

## A. Company track record

| Feature | Definition | Available from |
|---|---|---|
| `beat_rate_4q` | Share of the previous 4 reports that beat consensus | prior reports |
| `beat_rate_8q` | Share of the previous 8 reports that beat consensus | prior reports |
| `beat_rate_shrunk` | 8-quarter beat rate blended with the sector's prior beat rate (4 pseudo-quarters) | prior reports |
| `inline_rate_8q` | Share of the previous 8 reports that exactly matched consensus | prior reports |
| `beat_streak` | Number of consecutive beats immediately before this report | prior reports |
| `last_surprise_ps` | Last quarter's surprise (actual - estimate) divided by the share price before that report | prior reports |
| `mean_surprise_ps_4q` | Average price-scaled surprise over the previous 4 reports | prior reports |
| `std_surprise_ps_8q` | Volatility of price-scaled surprises over the previous 8 reports | prior reports |
| `sector_prior` | Beat rate of all earlier same-sector reports since 2000 | peers' earlier reports |

## B. How demanding the estimate is

| Feature | Definition | Available from |
|---|---|---|
| `est_vs_yoy_actual_ps` | Consensus estimate minus actual EPS of the same quarter last year, divided by price | consensus + prior reports |
| `est_growth_yoy` | Implied EPS growth vs the same quarter last year (capped at +/-200%) | consensus + prior reports |
| `est_vs_last_actual_ps` | Consensus estimate minus last quarter's actual EPS, divided by price | consensus + prior reports |
| `est_negative` | 1 if analysts expect a loss | consensus |

## C. Fundamentals (SEC filings)

| Feature | Definition | Available from |
|---|---|---|
| `revenue_growth_yoy` | Revenue growth vs the same quarter a year earlier (latest quarter filed before the report) | SEC filings dated before the report |
| `gross_margin_chg_yoy` | Change in gross margin vs the same quarter a year earlier | SEC filings dated before the report |
| `op_margin_chg_yoy` | Change in operating margin vs the same quarter a year earlier | SEC filings dated before the report |
| `op_margin` | Operating margin in the latest filed quarter | SEC filings dated before the report |
| `inventory_minus_sales_growth` | Inventory growth minus revenue growth, year over year | SEC filings dated before the report |
| `receivables_minus_sales_growth` | Receivables growth minus revenue growth, year over year | SEC filings dated before the report |
| `accruals_to_assets` | (Net income - operating cash flow) / total assets, latest filed quarter | SEC filings dated before the report |
| `leverage` | 1 - shareholders' equity / total assets | SEC filings dated before the report |
| `log_market_cap` | Log of shares outstanding (latest filing) x share price before the report | SEC filings dated before the report |
| `fundamentals_age_days` | Days between the end of the latest filed quarter and the report | SEC filings dated before the report |

## D. Market signals before the report

| Feature | Definition | Available from |
|---|---|---|
| `ret_20d_vs_spy` | Stock return minus S&P 500 return over the 20 trading days before the report | close of the day before the report |
| `ret_60d_vs_spy` | Same over 60 trading days | close of the day before the report |
| `vol_60d` | Annualised volatility of daily returns over the previous 60 trading days | close of the day before the report |
| `dist_52w_high` | Price relative to its 52-week high (0 = at the high) | close of the day before the report |
| `log_dollar_volume_60d` | Log of average daily dollar trading volume over 60 days (size proxy) | close of the day before the report |
| `prev_reaction_2d` | Stock reaction vs S&P 500 to the previous earnings report (2 days) | prior reports |

## E. Timing and context

| Feature | Definition | Available from |
|---|---|---|
| `is_bmo` | 1 if reported before the market opens | report schedule |
| `report_month` | Calendar month of the report | report schedule |
| `days_since_last_report` | Days since the company's previous earnings report | report schedule |
| `report_gap_change_vs_ly` | Change in that gap vs the same report last year (later than usual > 0) | report schedule |
| `reporting_lag_days` | Days from the end of the quarter being reported to the report date | quarter-end date and report schedule |
| `reporting_lag_change_vs_ly` | Change in reporting lag vs the same quarter last year (later than usual > 0) | quarter-end date and report schedule |

## F. Earnings season so far

| Feature | Definition | Available from |
|---|---|---|
| `season_sector_beat_rate` | Share of same-sector companies that beat in the 45 days before this report | peers' earlier reports |
| `season_sector_n` | Number of same-sector reports counted above | peers' earlier reports |
| `season_all_beat_rate` | Share of all universe companies that beat in the 45 days before | peers' earlier reports |
| `season_all_n` | Number of reports counted above | peers' earlier reports |

## Known gaps

- **Estimate revisions** (how analysts' estimates changed during the quarter) are among the strongest known predictors but are not available in free data.
- Price-based features are missing for the 8 delisted companies without price history (~4% of reports).
- Fundamentals are missing where a company does not report the line item (e.g. inventory for software firms), so coverage varies by feature (72–99%).
