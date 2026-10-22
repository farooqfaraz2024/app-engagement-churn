-- Which events exist, how often, and for how many users. Run this first on any new
-- export to decide which events are "journeys" worth counting.
SELECT
  event_name,
  COUNT(*) AS events,
  COUNT(DISTINCT user_pseudo_id) AS users,
  MIN(PARSE_DATE('%Y%m%d', event_date)) AS first_date,
  MAX(PARSE_DATE('%Y%m%d', event_date)) AS last_date
FROM {events_source}
GROUP BY event_name
ORDER BY events DESC
