-- Optional: copy Google's public Flood-It! export into your own project as one table.
-- Run in the BigQuery console after creating a dataset named flood_it in location US
-- (the public data lives in the US multi-region; a copy must be in the same location).
-- Replace YOUR_PROJECT_ID, then point the pipeline at it:
--   EVENTS_TABLE=YOUR_PROJECT_ID.flood_it.events
--
-- Clustered, not date-partitioned, on purpose: in the free BigQuery sandbox,
-- partitions older than 60 days are deleted automatically, and this data is from
-- 2018, so a date-partitioned copy would disappear. Sandbox tables still expire
-- 60 days after creation; re-run this script if that happens.

CREATE OR REPLACE TABLE `YOUR_PROJECT_ID.flood_it.events`
CLUSTER BY event_date, user_pseudo_id
AS
SELECT *
FROM `firebase-public-project.analytics_153293282.events_*`;
