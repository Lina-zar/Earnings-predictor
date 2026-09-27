-- 900_checks.sql : data-quality checks. Each query lists PROBLEMS, so on clean data
-- every one returns 0 rows. load_db.py runs them all and prints the counts.

-- check: duplicate events (same company, same report date)
SELECT company_id, report_date, COUNT(*) FROM earnings_events
GROUP BY 1, 2 HAVING COUNT(*) > 1;

-- check: in-universe events missing the estimate or the actual
SELECT event_id FROM earnings_events
WHERE in_universe AND (eps_estimate IS NULL OR eps_actual IS NULL);

-- check: two reports less than 45 days apart for one company (double-counted quarter)
SELECT * FROM (
  SELECT company_id, report_date,
         report_date - LAG(report_date) OVER (PARTITION BY company_id ORDER BY report_date) AS gap
  FROM earnings_events WHERE in_universe) t
WHERE gap < 45;

-- check: gaps over 150 days between reports while in the index (missing quarter;
--        year-end reports routinely take ~125-140 days, so those are not flagged)
SELECT * FROM (
  SELECT company_id, report_date,
         report_date - LAG(report_date) OVER (PARTITION BY company_id ORDER BY report_date) AS gap
  FROM earnings_events WHERE in_universe) t
WHERE gap > 150;

-- check: in-universe events with no day-0 reaction (missing prices around the report)
SELECT v.event_id, v.company_key, v.report_date FROM v_events v
LEFT JOIN v_reaction r USING (event_id)
WHERE v.in_universe AND r.car_0_1 IS NULL;

-- check: price gaps over 5 calendar days
SELECT * FROM (
  SELECT company_id, trade_date,
         trade_date - LAG(trade_date) OVER (PARTITION BY company_id ORDER BY trade_date) AS gap
  FROM daily_prices) t
WHERE gap > 5;

-- check: one-day moves over 40% - a review list, not errors (most are real news:
--        March 2020, takeover bids, earnings shocks); look for unrepaired splits
SELECT * FROM (
  SELECT company_id, trade_date,
         adj_close / LAG(adj_close) OVER (PARTITION BY company_id ORDER BY trade_date) - 1 AS r
  FROM daily_prices) t
WHERE ABS(r) > 0.40;

-- check: surprise % exploding because the estimate is near zero
SELECT event_id, company_key, eps_estimate, surprise_pct FROM v_events
WHERE in_universe AND ABS(eps_estimate) < 0.05;
