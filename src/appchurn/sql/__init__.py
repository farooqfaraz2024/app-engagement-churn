"""Render the SQL templates from config.

Values are validated and inserted as literals rather than query parameters so the
exact SQL that ran can be printed, reviewed and dry-run. Every value comes from
config.py, never from end-user input, and is checked against a strict pattern.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from importlib.resources import files

from appchurn.config import DATASET, WINDOWS, DatasetConfig, WindowConfig, snapshot_dates

_EVENT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,99}$")
_TABLE = re.compile(r"^[A-Za-z0-9_\-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+\*?$")


def _check_event(name: str) -> str:
    if not _EVENT_NAME.match(name):
        raise ValueError(f"Invalid event name: {name!r}")
    return name


def _template(name: str) -> str:
    return files("appchurn.sql").joinpath(name).read_text()


def events_source(
    dataset: DatasetConfig = DATASET,
    start: date | None = None,
    end: date | None = None,
    local_test: bool = False,
) -> str:
    """FROM clause target, restricted to the date range.

    Two kinds of table are supported:

    * a sharded GA4 export (``project.dataset.events_*``, one table per day), where
      the range is applied through ``_TABLE_SUFFIX`` so BigQuery reads only the
      needed daily tables;
    * a single table (for example your own copy of the export), filtered on
      ``event_date``.

    ``local_test=True`` leaves the name unquoted for the DuckDB tests.
    """
    table = dataset.events_table
    if not _TABLE.match(table):
        raise ValueError(f"Invalid table: {table!r}")
    start = start or dataset.data_start
    end = end or dataset.data_end
    date_filter = f"'{start:%Y%m%d}' AND '{end:%Y%m%d}'"
    if local_test:
        return f"{table.rstrip('*')}\n  WHERE event_date BETWEEN {date_filter}"
    if table.endswith("*"):
        return f"`{table}`\n  WHERE _TABLE_SUFFIX BETWEEN {date_filter}"
    return f"`{table}`\n  WHERE event_date BETWEEN {date_filter}"


_FUTURE_CTE = """,

future AS (
  SELECT
    s.snapshot_date,
    d.user_pseudo_id,
    SUM(d.active_events) AS future_active_events
  FROM snapshots AS s
  JOIN daily AS d
    ON d.activity_date >= s.snapshot_date
   AND d.activity_date < DATE_ADD(s.snapshot_date, INTERVAL {churn} DAY)
  GROUP BY s.snapshot_date, d.user_pseudo_id
)"""


def user_snapshot_sql(
    dataset: DatasetConfig = DATASET,
    windows: WindowConfig = WINDOWS,
    local_test: bool = False,
    score_date: date | None = None,
) -> str:
    """Feature SQL, one row per (snapshot, user).

    Default: every historical snapshot, each with its ``churned`` label (training).
    With ``score_date``: that single snapshot and no label, because on a real scoring
    day the next 14 days have not happened yet. Events are read up to the day before
    ``score_date`` only.
    """
    journeys = [_check_event(e) for e in dataset.journey_events]
    passive = ", ".join(f"'{_check_event(e)}'" for e in dataset.passive_events)
    obs, churn = windows.observation_days, windows.churn_days

    if score_date is None:
        snaps = snapshot_dates(dataset, windows)
        if not snaps:
            raise ValueError("Date range too short for the observation and churn windows")
        # Read from the earliest observation day to the last churn-window day only.
        start, end = snaps[0] - timedelta(days=obs), snaps[-1] + timedelta(days=churn)
        future_cte = _FUTURE_CTE.format(churn=churn)
        label_column = ",\n  IF(COALESCE(f.future_active_events, 0) = 0, 1, 0) AS churned"
        future_join = "\nLEFT JOIN future AS f USING (snapshot_date, user_pseudo_id)"
    else:
        snaps = [score_date]
        start, end = score_date - timedelta(days=obs), score_date - timedelta(days=1)
        future_cte = label_column = future_join = ""

    daily_counts = "\n".join(f"    COUNTIF(event_name = '{e}') AS n_{e.lower()}," for e in journeys)
    window_sums = "\n".join(f"    SUM(d.n_{e.lower()}) AS n_{e.lower()}_{obs}d," for e in journeys)

    return _template("user_snapshot_features.sql").format(
        passive_events=passive,
        journey_daily_counts=daily_counts,
        journey_window_sums=window_sums,
        events_source=events_source(dataset, start, end, local_test),
        snapshot_dates=", ".join(f"DATE '{d:%Y-%m-%d}'" for d in snaps),
        obs=obs,
        future_cte=future_cte,
        label_column=label_column,
        future_join=future_join,
    )


def event_profile_sql(dataset: DatasetConfig = DATASET, local_test: bool = False) -> str:
    return _template("event_profile.sql").format(
        events_source=events_source(dataset, local_test=local_test)
    )
