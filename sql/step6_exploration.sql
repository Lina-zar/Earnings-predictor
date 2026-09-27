-- step6_exploration.sql : the Part 1, Step 6 questions. Run any query in psql,
-- DBeaver, TablePlus or Tableau; load_db.py also prints their results.

-- Q1. What % of quarters were beats, by sector and by year?
SELECT COALESCE(sector, 'ALL SECTORS')          AS sector,
       COALESCE(report_year::TEXT, 'all years')  AS year,
       ROUND(100.0 * AVG(beat), 1) AS beat_pct,
       COUNT(*)                    AS quarters
FROM v_events
WHERE in_universe AND beat IS NOT NULL
GROUP BY ROLLUP (sector, report_year)
ORDER BY GROUPING(sector), sector, GROUPING(report_year), report_year;

-- Q2. Average stock reaction (vs SPY) to a beat, an exact match and a miss
SELECT CASE WHEN v.surprise > 0 THEN 'beat'
            WHEN v.surprise = 0 THEN 'in line'
            ELSE 'miss' END                         AS outcome,
       COUNT(*)                                     AS quarters,
       ROUND(100 * AVG(r.car_0_1)::NUMERIC, 2)      AS avg_2day_pct,
       ROUND(100 * AVG(r.car_m2_p2)::NUMERIC, 2)    AS avg_factset_window_pct,
       ROUND(100 * PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.car_0_1)::NUMERIC, 2)
                                                    AS median_2day_pct
FROM v_events v JOIN v_reaction r USING (event_id)
WHERE v.in_universe AND v.beat IS NOT NULL AND r.car_0_1 IS NOT NULL
GROUP BY 1 ORDER BY 1;

-- Q3. Which companies beat most consistently? (at least 16 quarters in the index)
SELECT company_key, sector, COUNT(*) AS quarters,
       ROUND(100.0 * AVG(beat), 1) AS beat_pct,
       SUM(1 - beat)               AS misses_or_ties
FROM v_events
WHERE in_universe AND beat IS NOT NULL
GROUP BY company_key, sector
HAVING COUNT(*) >= 16
ORDER BY beat_pct DESC, quarters DESC
LIMIT 15;

-- Q4. Least consistent (most misses) - the cases the model is really about
SELECT company_key, sector, COUNT(*) AS quarters,
       ROUND(100.0 * AVG(beat), 1) AS beat_pct
FROM v_events
WHERE in_universe AND beat IS NOT NULL
GROUP BY company_key, sector
HAVING COUNT(*) >= 16
ORDER BY beat_pct ASC
LIMIT 15;

-- Q5. Missing quarters: companies with under 80% of the quarters their time in the
--     index implies (share classes are counted once)
SELECT c.company_key, c.status, c.expected_quarters AS expected,
       COUNT(e.event_id)                     AS found
FROM companies c
LEFT JOIN earnings_events e ON e.company_id = c.company_id AND e.in_universe
WHERE c.expected_quarters > 0
GROUP BY c.company_id, c.company_key, c.status, c.expected_quarters
HAVING COUNT(e.event_id) < 0.8 * c.expected_quarters
ORDER BY c.expected_quarters DESC;

-- Q6. Does the reaction to a miss depend on its size? (surprise % buckets)
SELECT CASE WHEN v.surprise_pct < -0.10 THEN '1: miss by >10%'
            WHEN v.surprise_pct < 0     THEN '2: miss by 0-10%'
            WHEN v.surprise_pct = 0     THEN '3: in line'
            WHEN v.surprise_pct <= 0.10 THEN '4: beat by 0-10%'
            ELSE                             '5: beat by >10%' END AS surprise_bucket,
       COUNT(*)                                AS quarters,
       ROUND(100 * AVG(r.car_0_1)::NUMERIC, 2) AS avg_2day_pct
FROM v_events v JOIN v_reaction r USING (event_id)
WHERE v.in_universe AND v.surprise_pct IS NOT NULL AND r.car_0_1 IS NOT NULL
GROUP BY 1 ORDER BY 1;
