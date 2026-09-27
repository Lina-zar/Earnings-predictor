# Earnings Predictor — Project Charter

*Part 1 of the Earnings Predictor & Equity Research Engine · last updated 27 September 2026*

## Problem

Large US companies beat analysts' earnings-per-share (EPS) estimates most of the time: 79% of quarters in this dataset. A beat is therefore close to the default and carries little information. The information sits in the minority of quarters where a company fails to beat. Those quarters are also where the market reacts hardest: over 2018–2025, stocks fell 3.2% relative to the S&P 500 in the two days after a miss, against a gain of only 0.6% after a beat.

## Question

**Using only information available before an earnings announcement, can we identify which quarters will fail to beat the consensus EPS estimate — better than simple base rates can?**

## Hypotheses

1. **Track record matters, but only once it is weighted properly.** A company's recent beat history predicts the next quarter better when it is combined with its sector's base rate than when used raw.
2. **Demanding estimates are more likely to be missed.** Quarters where analysts expect unusually high growth over the same quarter last year miss more often.
3. **The market partly anticipates misses.** Weak relative stock performance in the weeks before the report signals a higher chance of missing.
4. **Earnings season is contagious.** When many companies in the same sector have already missed this season, the chance that the next one misses rises.
5. **Fundamentals give early warnings.** Inventory or receivables growing faster than sales, and reports filed later than usual, precede misses.

## Data

| Item | Choice |
|---|---|
| Universe | Point-in-time S&P 500 members in four sectors (Information Technology, Consumer Discretionary, Consumer Staples, Communication Services), using today's GICS sector labels throughout |
| Membership rule | A quarter is included if the company was an index member on its earnings report date |
| Period | Report dates 1 January 2018 – 31 December 2025; history from earlier years used only to build features |
| Size | 231 companies, of which 2 (Carvana, Sandisk) joined after the study window; 213 have usable data; **5,380 labelled quarters** |
| EPS estimates and actuals | yfinance (adjusted "Street" EPS, rounded to the cent) |
| Cross-checks | Alpha Vantage on a stratified sample of 50 companies; SEC 8-K earnings filings for report dates (99% match within one day) |
| Prices | yfinance daily prices, raw and adjusted; SPY and sector ETFs as benchmarks |
| Fundamentals (Part 2) | SEC EDGAR company facts, first-filed values only |
| Storage | PostgreSQL, with an append-only raw-file cache and automated data-quality checks |

**Target.** `not_beat` = 1 if reported EPS is at or below the consensus estimate, else 0. An exact match counts as not beating. The market treats it that way too: in-line quarters fell 2.0% on average, close to the 3.2% fall for misses.

## Success criteria

A model that always predicts "beat" is already 79% accurate (87% in IT), so **accuracy is not the success metric**.

| Metric | Bar to clear |
|---|---|
| **Brier score** (quality of predicted probabilities), walk-forward 2019–2025 | Below **0.164**, the sector base-rate baseline (overall base rate: 0.167) |
| **Miss detection** (precision–recall AUC for not-beat quarters) | Above the not-beat share of about 21% |
| Calibration | Predicted probabilities match observed rates by decile |

Validation is walk-forward by quarter, training only on earlier data and never shuffling across time. Results are reported by year so that 2020–21 cannot dominate.

## Key findings so far (Part 1)

- **Misses are punished about 5.6 times harder than beats are rewarded:** beat +0.57%, in line −2.01%, miss −3.20%, over two days relative to the S&P 500.
  - FactSet reports roughly 2.5x. The difference comes mainly from method: this project uses market-adjusted rather than raw returns, on a four-sector large-cap sample over 2018–2025. On FactSet's own day −2 to +2 window, the ratio here is still about 4.9x.
- **Meeting expectations is not enough:** beating by less than 10% averages −0.19%. Only beats above 10% are clearly rewarded (+1.60%).
- **Vendor data needed correcting:** a cross-check found Alpha Vantage reporting GAAP instead of adjusted EPS in some quarters, mixing stock-split bases (Walmart) and using annual-report filing dates instead of announcement dates. yfinance was chosen as the primary source after passing all three checks.

## Limitations

- **Survivorship:** 16 companies (~203 quarters, 3.6% of the universe) have no estimate history in free sources because they were acquired, taken private or delisted. The largest exclusion is **Electronic Arts**, an index member for the whole period. Its SEC financials remain available for comparison.
- **Estimate timing:** free sources do not say whether the historical consensus is the one on the day before the report or a later revision.
- **No estimate-revision history:** free data does not provide how estimates changed during the quarter, one of the strongest known predictors. A production version would add it from I/B/E/S.
- **Sectors:** today's GICS labels are used throughout. The 2018 and 2023 reclassifications are not reflected historically.
- **Prices:** 8 delisted companies have earnings data but no prices, so their quarters lack market features and reaction measures.

## Deliverables

1. **Database:** PostgreSQL with point-in-time tables, views and data-quality checks *(done)*
2. **Feature table:** one row per labelled quarter, every feature built only from pre-announcement information, with automated leakage tests and a feature dictionary *(Part 2)*
3. **Model:** logistic regression and gradient-boosted trees, evaluated walk-forward against the baselines above *(Part 3)*
4. **Dashboard:** beat rates, reactions and model predictions by sector and company
5. **Report:** a short write-up of findings, method and limitations
