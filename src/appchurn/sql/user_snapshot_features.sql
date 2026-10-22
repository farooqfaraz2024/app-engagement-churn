-- One row per (snapshot_date, user): behaviour in the observation window before the
-- snapshot, and whether the user went silent in the churn window after it.
-- Works on any Firebase / GA4 BigQuery export. Placeholders in curly braces are filled
-- in by appchurn.sql from config.py; all values are validated before rendering.

WITH daily AS (
  -- The only step that reads the raw event table: collapse events to one row per
  -- user per day. Everything after this works on a small table.
  SELECT
    user_pseudo_id,
    PARSE_DATE('%Y%m%d', event_date) AS activity_date,
    COUNT(*) AS events,
    COUNTIF(event_name NOT IN ({passive_events})) AS active_events,
    COUNTIF(event_name = 'session_start') AS sessions,
    SUM(COALESCE((
      SELECT ep.value.int_value FROM UNNEST(event_params) AS ep
      WHERE ep.key = 'engagement_time_msec'
    ), 0)) AS engagement_msec,
    SUM(COALESCE(event_value_in_usd, 0)) AS revenue_usd,
{journey_daily_counts}
    MIN(user_first_touch_timestamp) AS first_touch_us,
    MAX(platform) AS platform,
    MAX(geo.country) AS country,
    MAX(device.category) AS device_category
  FROM {events_source}
  GROUP BY user_pseudo_id, activity_date
),

snapshots AS (
  SELECT snapshot_date FROM UNNEST([{snapshot_dates}]) AS snapshot_date
),

user_first_seen AS (
  SELECT
    user_pseudo_id,
    LEAST(
      MIN(activity_date),
      COALESCE(DATE(TIMESTAMP_MICROS(MIN(first_touch_us))), MIN(activity_date))
    ) AS first_seen_date
  FROM daily
  GROUP BY user_pseudo_id
),

observation AS (
  SELECT
    s.snapshot_date,
    d.user_pseudo_id,
    COUNT(DISTINCT IF(d.active_events > 0, d.activity_date, NULL)) AS active_days_{obs}d,
    COUNT(DISTINCT IF(d.active_events > 0 AND d.activity_date >= DATE_SUB(s.snapshot_date, INTERVAL 7 DAY),
                      d.activity_date, NULL)) AS active_days_7d,
    SUM(d.events) AS events_{obs}d,
    SUM(IF(d.activity_date >= DATE_SUB(s.snapshot_date, INTERVAL 7 DAY), d.events, 0)) AS events_7d,
    SUM(d.sessions) AS sessions_{obs}d,
    SUM(d.engagement_msec) / 60000.0 AS engagement_minutes_{obs}d,
    SUM(IF(d.activity_date >= DATE_SUB(s.snapshot_date, INTERVAL 7 DAY), d.engagement_msec, 0))
      / 60000.0 AS engagement_minutes_7d,
    SUM(d.revenue_usd) AS revenue_usd_{obs}d,
    DATE_DIFF(s.snapshot_date, MAX(IF(d.active_events > 0, d.activity_date, NULL)), DAY)
      AS days_since_last_active,
{journey_window_sums}
    MAX(d.platform) AS platform,
    MAX(d.country) AS country,
    MAX(d.device_category) AS device_category
  FROM snapshots AS s
  JOIN daily AS d
    ON d.activity_date >= DATE_SUB(s.snapshot_date, INTERVAL {obs} DAY)
   AND d.activity_date < s.snapshot_date
  GROUP BY s.snapshot_date, d.user_pseudo_id
  HAVING COUNT(DISTINCT IF(d.active_events > 0, d.activity_date, NULL)) > 0
){future_cte}

SELECT
  o.*,
  DATE_DIFF(o.snapshot_date, u.first_seen_date, DAY) AS tenure_days{label_column}
FROM observation AS o
JOIN user_first_seen AS u USING (user_pseudo_id){future_join}
ORDER BY o.snapshot_date, o.user_pseudo_id
