-- ============================================================
-- Customer Sentiment NLP Pipeline — SQL Analysis Queries
-- ============================================================
-- Compatible with: DuckDB, SQLite, PostgreSQL
--
-- Load data (DuckDB):
--   CREATE TABLE reviews  AS SELECT * FROM read_csv_auto('data/processed/reviews.csv');
--   CREATE TABLE company  AS SELECT * FROM read_csv_auto('data/processed/company_summary.csv');
--   CREATE TABLE monthly  AS SELECT * FROM read_csv_auto('data/processed/monthly_trend.csv');
--   CREATE TABLE topics   AS SELECT * FROM read_csv_auto('data/processed/topic_distribution.csv');
--   CREATE TABLE priority AS SELECT * FROM read_csv_auto('data/processed/priority_complaints.csv');
--
-- NOTE: All queries use `true_sentiment` (ground-truth label) only.
--       Model confidence scores are NOT used as features in any query.
-- ============================================================


-- ── 1. OVERALL SENTIMENT DISTRIBUTION ─────────────────────────────────────
-- Baseline breakdown — sanity check before any analysis

SELECT
    true_sentiment,
    COUNT(*)                                      AS review_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS pct_of_total
FROM reviews
GROUP BY true_sentiment
ORDER BY review_count DESC;


-- ── 2. COMPANY SENTIMENT LEAGUE TABLE ─────────────────────────────────────
-- Which companies generate the most positive vs negative reviews?

SELECT
    company,
    category,
    total_reviews,
    ROUND(avg_rating, 2)        AS avg_star_rating,
    pct_positive                AS pct_pos,
    pct_negative                AS pct_neg,
    pct_neutral                 AS pct_neu,
    priority_count,
    CASE
        WHEN pct_positive >= 60 THEN 'Strong'
        WHEN pct_positive >= 50 THEN 'Moderate'
        WHEN pct_positive >= 40 THEN 'Weak'
        ELSE 'Poor'
    END AS sentiment_band
FROM company
ORDER BY pct_positive DESC;


-- ── 3. CATEGORY-LEVEL SENTIMENT ANALYSIS ──────────────────────────────────
-- Aggregate by product category — which sectors perform worst?

SELECT
    category,
    COUNT(*)                                          AS total_reviews,
    ROUND(AVG(rating), 2)                             AS avg_rating,
    ROUND(SUM(CASE WHEN true_sentiment='positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_positive,
    ROUND(SUM(CASE WHEN true_sentiment='negative' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_negative,
    ROUND(SUM(CASE WHEN true_sentiment='neutral'  THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_neutral,
    SUM(CASE WHEN is_priority THEN 1 ELSE 0 END)      AS priority_complaints
FROM reviews
GROUP BY category
ORDER BY pct_negative DESC;


-- ── 4. TOPIC DISTRIBUTION AND SENTIMENT BY TOPIC ──────────────────────────
-- What do customers talk about, and how does sentiment split per topic?

SELECT
    primary_topic,
    COUNT(*)                                              AS total_reviews,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1)    AS pct_of_all,
    ROUND(SUM(CASE WHEN true_sentiment='negative' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_negative,
    ROUND(SUM(CASE WHEN true_sentiment='positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_positive,
    SUM(CASE WHEN is_priority THEN 1 ELSE 0 END)          AS priority_count
FROM reviews
GROUP BY primary_topic
ORDER BY total_reviews DESC;


-- ── 5. MONTHLY SENTIMENT TREND ─────────────────────────────────────────────
-- Are things improving or getting worse over time?

SELECT
    year,
    month,
    quarter,
    total_reviews,
    ROUND(avg_rating, 2)    AS avg_rating,
    pct_positive,
    pct_negative,
    priority_count,
    -- 3-month rolling average of negative % (requires DuckDB/PostgreSQL)
    ROUND(AVG(pct_negative) OVER (
        ORDER BY year, month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 1) AS rolling_3m_neg_pct
FROM monthly
ORDER BY year, month;


-- ── 6. PRIORITY COMPLAINT ANALYSIS ────────────────────────────────────────
-- Deep dive into flagged reviews requiring immediate action

SELECT
    company,
    category,
    primary_topic,
    priority_signals,
    rating,
    LEFT(review_text, 120) || '...' AS review_extract,
    date
FROM priority
ORDER BY date DESC
LIMIT 50;


-- ── 7. PRIORITY COMPLAINT RATE BY COMPANY ─────────────────────────────────
-- Which companies have the highest proportion of flagged complaints?

SELECT
    r.company,
    r.category,
    COUNT(*)                                                  AS total_reviews,
    SUM(CASE WHEN r.is_priority THEN 1 ELSE 0 END)            AS priority_count,
    ROUND(SUM(CASE WHEN r.is_priority THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS priority_rate_pct,
    ROUND(AVG(r.rating), 2)                                   AS avg_rating
FROM reviews r
GROUP BY r.company, r.category
HAVING COUNT(*) >= 10
ORDER BY priority_rate_pct DESC;


-- ── 8. RATING VS SENTIMENT CROSS-TAB ──────────────────────────────────────
-- Validates label quality — 5-star reviews should be positive, 1-star negative
-- This is an internal consistency check, NOT used in model training

SELECT
    rating,
    COUNT(*)                                              AS total,
    ROUND(SUM(CASE WHEN true_sentiment='positive' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_positive,
    ROUND(SUM(CASE WHEN true_sentiment='negative' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_negative,
    ROUND(SUM(CASE WHEN true_sentiment='neutral'  THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_neutral
FROM reviews
GROUP BY rating
ORDER BY rating;


-- ── 9. TOP NEGATIVE SIGNAL TOPICS (priority only) ─────────────────────────
-- Which topics are most associated with safety and legal escalations?

SELECT
    primary_topic,
    COUNT(*)                                          AS priority_count,
    ROUND(AVG(rating), 2)                             AS avg_rating,
    -- Most common signals
    COUNT(CASE WHEN priority_signals LIKE '%safety%'   THEN 1 END) AS safety_signals,
    COUNT(CASE WHEN priority_signals LIKE '%legal%'    THEN 1 END) AS legal_signals,
    COUNT(CASE WHEN priority_signals LIKE '%fraud%'    THEN 1 END) AS fraud_signals,
    COUNT(CASE WHEN priority_signals LIKE '%health%'   THEN 1 END) AS health_signals
FROM priority
GROUP BY primary_topic
ORDER BY priority_count DESC;


-- ── 10. QUARTERLY PERFORMANCE SUMMARY ─────────────────────────────────────
-- Board-level quarterly view for stakeholder reporting

SELECT
    year,
    quarter,
    SUM(total_reviews)          AS quarterly_reviews,
    ROUND(AVG(avg_rating), 2)   AS avg_rating,
    ROUND(AVG(pct_positive), 1) AS avg_pct_positive,
    ROUND(AVG(pct_negative), 1) AS avg_pct_negative,
    SUM(priority_count)         AS total_priority_complaints,
    ROUND(SUM(priority_count) * 100.0 / SUM(total_reviews), 2) AS priority_rate_pct
FROM monthly
GROUP BY year, quarter
ORDER BY year, quarter;
