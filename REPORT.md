# Earnings Surprise Predictor — Final Report

*S&P 500 · Information Technology, Consumer Discretionary, Consumer Staples, Communication Services · report dates 2018–2025*

## 1. Summary

Large US companies beat Wall Street's EPS estimate 79% of the time, so a beat is the default. This project asks whether the **21% of quarters that fail to beat** can be identified *before* the announcement, using only information available at the time.

**Answer: yes, modestly and reliably.** A walk-forward logistic regression improves on the sector base rate by **7.5% in Brier score** (0.1512 vs 0.1635), reaches a **ROC AUC of 0.71**, and beats the baseline in **all 7 test years (2019–2025)**. The 10% of quarters it rates riskiest fail to beat **46%** of the time, against 21% on average.

**But this is not a trading signal.** The market already prices the same risk: stocks the model rates risky react *less* badly to bad news and *more* positively to beats. Predicting misses is not the same as predicting returns.

## 2. Why it matters

In the two days after a report (stock return minus the S&P 500):

| Outcome | Share of quarters | Average 2-day reaction |
|---|---|---|
| Beat | 79% | **+0.57%** |
| Exactly in line | 5% | **−2.01%** |
| Miss | 16% | **−3.20%** |

Misses are punished about **5.6 times** harder than beats are rewarded, and meeting the estimate exactly is treated almost like a miss. That asymmetry is why the not-beat quarters are the ones worth predicting.

## 3. Data

| | |
|---|---|
| Universe | Point-in-time S&P 500 membership (a quarter counts if the company was in the index on its report date), 4 sectors, today's GICS labels |
| Size | 231 companies → 213 with data → **5,380 labelled quarters** |
| EPS | yfinance adjusted ("Street") EPS and consensus |
| Prices | yfinance daily prices; SPY and sector ETFs as benchmarks |
| Fundamentals | SEC EDGAR company facts, **first-filed values only** (no later restatements), Q4 derived as annual minus nine-month year-to-date |
| Storage | PostgreSQL with automated data-quality checks |

**Data-quality work that changed the result:**
- **EPS basis.** An Alpha Vantage cross-check (42 companies, 1,313 quarters) found the secondary vendor mixing GAAP with adjusted EPS, mixing stock-split bases and using 10-K filing dates instead of announcement dates. yfinance passed all three checks and became the primary source.
- **Report dates.** 99% of dates match an SEC 8-K earnings filing within one day.
- **Stock splits.** Yahoo's prices are adjusted for *all* splits, including those after the study window (for example KLA 10:1, Booking 25:1). Market caps were rebuilt from SEC diluted share counts with the full split history undone. KLA's 2018 market cap went from an impossible figure to the correct ~$18bn.
- **Tickers.** Paramount (CBS/Viacom/PARA/PSKY), Gap, Wyndham and dual-class shares (GOOG/GOOGL and others) were merged by hand so that each company is counted once.

## 4. Method

- **Target:** `not_beat` = 1 if actual EPS ≤ estimate (ties count as not beating, matching the market's reaction).
- **39 features** in 7 groups, each built only from information dated before the report, with automated leakage tests:
  - Track record: beat rate, streak, past surprises
  - Market signals: returns vs S&P 500, distance from 52-week high, volatility
  - Earnings season so far: how many companies in the sector have beaten this season
  - Estimate demand: expected growth vs last year
  - Timing: reporting lag, before-open or after-close
  - Sector
  - SEC fundamentals: margins, leverage, inventory and receivables vs sales, accruals
- **Validation:** walk-forward by year. Each test year (2019–2025) is predicted by a model trained only on earlier years. Nothing is shuffled across time.
- **Models:**
  - Two baselines: the overall base rate and the sector base rate.
  - Logistic regression: C = 0.05, median imputation fitted on the training set only, 1–99% clipping.
  - Gradient-boosted trees.
- **Metric:** Brier score, the mean squared error of the predicted probability. Accuracy is useless here because "always beat" is already 79% accurate.

## 5. Results

| Model | Brier | Skill vs sector base rate | ROC AUC | PR AUC | Not-beat rate, riskiest 10% |
|---|---|---|---|---|---|
| Overall base rate | 0.1669 | −2.1% | 0.46 | 0.20 | 22% |
| Sector base rate | 0.1635 | — | 0.57 | 0.24 | 21% |
| **Logistic regression** | **0.1512** | **+7.5%** | **0.71** | **0.39** | **46%** |
| Gradient-boosted trees | 0.1557 | +4.8% | 0.68 | 0.36 | 46% |

- Logistic regression beats the baseline in 7 of 7 years; the trees do so in 6 of 7 (they lose in 2019, the year with the least training data).
- **Calibration is good:** predicted vs actual not-beat rate by decile runs from 5% vs 4% in the safest decile to 51% vs 46% in the riskiest. The probabilities can be read literally, with slight overconfidence at the top.
- PR AUC of 0.39 against a 21% not-beat share means roughly double the precision of random selection.

## 6. What drives the model

| Group | Skill without it | Skill with it alone |
|---|---|---|
| Track record | +3.6% (−3.9 pts) | +6.0% |
| Market signals | +6.6% (−0.8) | +2.3% |
| Earnings season | +6.9% (−0.6) | +0.9% |
| Sector | +7.2% (−0.3) | — |
| SEC fundamentals | +7.5% (−0.0) | +1.4% |

- **Track record does most of the work.** A company that has beaten 8 quarters in a row rarely misses (10% not-beat rate); one that beat 5 or fewer of the last 8 misses 36% of the time.
- **SEC fundamentals are individually predictive but add nothing once track record is known.** Inventory growing faster than sales raises the not-beat rate from 16% to 26% across quintiles, but the company's recent history already contains that information.
- **Earnings season is contagious.** When the sector has beaten less often so far this season, the next company is more likely to miss (26% vs 15%).

**The five hypotheses from the charter:**

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | Track record matters, especially once weighted by sector base rate | **Partly.** Track record is the strongest signal, but shrinking it toward the sector rate adds nothing measurable |
| 2 | Demanding estimates are missed more often | **Rejected.** The opposite holds: quarters with *low* expected growth miss more (28% vs 16–19%) |
| 3 | The market partly anticipates misses | **Supported.** Stocks far below their 52-week high miss 32% of the time vs 15% |
| 4 | Earnings season is contagious | **Supported** (see above) |
| 5 | Fundamentals give early warnings | **Supported individually, redundant in the model** |

## 7. Robustness

| Test | Skill |
|---|---|
| Full model, 2019–2025 | +7.5% |
| Ties counted as beats (target = strict miss) | +7.8% |
| Without SEC fundamentals | +7.5% |
| Test years 2021–2025 only | +8.9% |
| 2020 alone (COVID) | +7.3% |
| Compact 21-feature model, chosen on 2019–21, tested on 2022–25 | +8.6% (vs +8.9% full) |

The last test matters: pruning features *with hindsight* had suggested a smaller model was better, but chosen honestly on past data only it was not. Feature selection must follow the same walk-forward rule as training.

By segment, skill is highest for Communication Services (+10.0%) and for companies at either extreme of track record (+10.6% for frequent missers, +11.0% for 8-of-8 beaters), and lowest for Consumer Staples (+4.5%).

## 8. Can you trade it? The market test

| | Safest third | Riskiest third | Difference |
|---|---|---|---|
| Reaction when the company **beats** | +0.10% | **+1.13%** | +1.03 pts (t = 3.5) |
| Reaction when **in line** | −4.46% | −1.07% | +3.39 pts (t = 2.9) |
| Reaction when it **misses** | −3.83% | −2.76% | +1.07 pts (t = 1.1) |

The riskiest decile underperforms the safest by only 0.33 points after the report, which is not significant (t = −0.6). The market expects the same risk the model sees: a risky company that beats is a real surprise and is rewarded, while a safe company that misses is punished hardest. **The model measures information the market already uses; it does not beat the market.**

## 9. Limitations

- **Survivorship:** 16 companies (3.6% of expected quarters) have no free estimate history, the largest being Electronic Arts.
- **Estimate timing and revisions:** free data gives one consensus number with no timestamp and no revision history. Estimate revisions are among the strongest known predictors; a production version would add I/B/E/S data.
- **Rounding:** EPS is rounded to the cent, so 5.4% of quarters are exact ties.
- **Sector labels:** today's GICS labels are used throughout (see spot checks).
- **Report-time labels:** 30 in-window quarters (0.6%) are tagged "during market hours" or unknown. Some are really after-close reports. The 2-day reaction window still captures the reaction day for these.

## 10. Manual spot checks (29 September 2026)

| Check | Result |
|---|---|
| **Walmart, Q3 FY23 (reported 15 Nov 2022)** | Walmart reported adjusted EPS $1.50 (GAAP $0.66). The dataset shows $0.50: the same figure after the 3-for-1 split of February 2024. ✅ Correct and split-consistent; Alpha Vantage mixed the two bases |
| **Microsoft, Q1 FY22 (reported 26 Oct 2021, quarter ending Sept 2021)** | Microsoft reported $2.71 GAAP, which includes a $0.44 one-off tax benefit, and $2.27 excluding it. The dataset uses **$2.27**. ✅ Adjusted basis confirmed; Alpha Vantage's $2.71 was GAAP |
| **Microsoft, Q2 FY23 (24 Jan 2023)** | Non-GAAP $2.32, GAAP $2.20 (severance and impairment charges). The dataset uses $2.32 ✅. With an estimate of exactly $2.32 it is labelled "in line"; other sources quote a consensus a few cents lower, so the label depends on the estimate vendor, a known limitation of ties |
| **Supermicro, Feb 2025** | Preliminary Q2 FY25 results came out on **11 Feb 2025** (non-GAAP EPS $0.58–0.60). The dataset dates the quarter **25 Feb 2025**, when the delayed annual report was filed, with EPS $0.60. ⚠️ EPS consistent, date wrong: 1 of 5,380 reactions is measured two weeks late. Documented, not changed |
| **Disney, Paramount, Bunge report dates** (below 90% match with SEC 8-Ks) | The unmatched dates all come from before a change of legal entity: Disney's new holding company after the Fox deal (2019), Paramount Skydance (2025), Bunge's move to Switzerland (2023). Each earlier filing sits under an older SEC company number. ✅ Dates are correct; the check looked up the wrong filer |
| **Sector: TripAdvisor** | Moved from Consumer Discretionary to Communication Services in the September 2018 reclassification. ✅ Communication Services |
| **Sector: Scripps Networks** | A media company, left the index in March 2018 when Discovery bought it, before the reclassification. Labelled Communication Services, consistent with where media moved. ✅ Consistent with the "today's labels" rule |
| **Sector: Wyndham Worldwide** | Hotels and timeshares; Consumer Discretionary. ✅ |
| **Sector: Vontier** | Fortive spin-off (electronic and mobility technology); Information Technology. ✅ |

## 11. What I would do next

1. Add **estimate revisions** (I/B/E/S or a paid vendor): the biggest missing signal.
2. Use **text from earnings calls and 8-K guidance** (NLP) as a new feature group.
3. Predict the **size** of the surprise, or the reaction given the surprise, rather than beat vs not-beat.
4. Test whether the model's risk score combined with options-implied volatility says anything the market does not already know.

## Sources for the spot checks

- [Walmart Q3 FY23 earnings release](https://corporate.walmart.com/news/2022/11/15/walmart-releases-q3-fy23-earnings)
- [Microsoft FY22 Q1 press release](https://www.microsoft.com/en-us/investor/earnings/fy-2022-q1/press-release-webcast)
- [Microsoft FY23 Q2 press release](https://www.microsoft.com/en-us/investor/earnings/fy-2023-q2/press-release-webcast)
- [Supermicro Q2 FY25 preliminary results (Business Wire, 11 Feb 2025)](https://www.businesswire.com/news/home/20250211669277/en/Supermicro-Announces-Second-Quarter-Fiscal-Year-2025-Preliminary-Financial-Information)
