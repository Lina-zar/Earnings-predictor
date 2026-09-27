-- 010_views.sql : derived values, computed from the tables (never stored twice)

DROP VIEW IF EXISTS v_reaction;
DROP VIEW IF EXISTS v_events;
DROP VIEW IF EXISTS v_trading_days;

-- Earnings events with the beat label, surprise and day 0 (the first day the market
-- can react): the report date itself for before-open reports, the next trading day
-- for after-close reports.
CREATE VIEW v_events AS
SELECT e.event_id, e.company_id, c.company_key, c.sector,
       e.report_date, e.report_time, e.eps_estimate, e.eps_actual, e.in_universe,
       EXTRACT(YEAR FROM e.report_date)::INT                            AS report_year,
       (e.eps_actual - e.eps_estimate)                                  AS surprise,
       (e.eps_actual - e.eps_estimate) / NULLIF(ABS(e.eps_estimate), 0) AS surprise_pct,
       CASE WHEN e.eps_estimate IS NULL OR e.eps_actual IS NULL THEN NULL
            WHEN e.eps_actual > e.eps_estimate THEN 1 ELSE 0 END        AS beat,  -- ties = 0
       d0.trade_date AS day0,
       d0.td         AS day0_td
FROM earnings_events e
JOIN companies c USING (company_id)
LEFT JOIN LATERAL (
    SELECT t.trade_date, t.td FROM trading_days t
    WHERE t.trade_date >= e.report_date + CASE WHEN e.report_time = 'AMC' THEN 1 ELSE 0 END
    ORDER BY t.trade_date LIMIT 1) d0 ON TRUE
WHERE e.source = 'yfinance';

-- Market-adjusted stock reaction around each event (stock return minus SPY return):
--   car_0_1  : close before day 0 -> close of day +1   (2-day reaction)
--   car_0_2  : close before day 0 -> close of day +2   (3-day reaction)
--   car_m2_p2: close of day -2    -> close of day +2   (close to FactSet's window)
CREATE VIEW v_reaction AS
WITH w AS (
    SELECT v.event_id, v.company_id,
           dm2.trade_date AS dm2, dm1.trade_date AS dm1,
           dp1.trade_date AS dp1, dp2.trade_date AS dp2
    FROM v_events v
    JOIN trading_days dm2 ON dm2.td = v.day0_td - 2
    JOIN trading_days dm1 ON dm1.td = v.day0_td - 1
    JOIN trading_days dp1 ON dp1.td = v.day0_td + 1
    JOIN trading_days dp2 ON dp2.td = v.day0_td + 2)
SELECT w.event_id,
       (p1.adj_close / pb.adj_close - 1) - (s1.adj_close / sb.adj_close - 1) AS car_0_1,
       (p2.adj_close / pb.adj_close - 1) - (s2.adj_close / sb.adj_close - 1) AS car_0_2,
       (p2.adj_close / pa.adj_close - 1) - (s2.adj_close / sa.adj_close - 1) AS car_m2_p2
FROM w
LEFT JOIN daily_prices     pa ON pa.company_id = w.company_id AND pa.trade_date = w.dm2
LEFT JOIN daily_prices     pb ON pb.company_id = w.company_id AND pb.trade_date = w.dm1
LEFT JOIN daily_prices     p1 ON p1.company_id = w.company_id AND p1.trade_date = w.dp1
LEFT JOIN daily_prices     p2 ON p2.company_id = w.company_id AND p2.trade_date = w.dp2
LEFT JOIN benchmark_prices sa ON sa.symbol = 'SPY' AND sa.trade_date = w.dm2
LEFT JOIN benchmark_prices sb ON sb.symbol = 'SPY' AND sb.trade_date = w.dm1
LEFT JOIN benchmark_prices s1 ON s1.symbol = 'SPY' AND s1.trade_date = w.dp1
LEFT JOIN benchmark_prices s2 ON s2.symbol = 'SPY' AND s2.trade_date = w.dp2;
