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
    wildcard: bool = True,
) -> str:
    """FROM clause target, restricted to the date range.

    For sharded GA4 exports (``events_*``) the range is applied through
    ``_TABLE_SUFFIX`` so BigQuery only reads the needed daily tables, which is what
    keeps cost down. ``wildcard=False`` is used by the local tests.
    """
    table = dataset.events_table
    if not _TABLE.match(table):
        raise ValueError(f"Invalid table: {table!r}")
    start = start or dataset.data_start
    end = end or dataset.data_end
    if wildcard and table.endswith("*"):
        return f"`{table}`\n  WHERE _TABLE_SUFFIX BETWEEN '{start:%Y%m%d}' AND '{end:%Y%m%d}'"
    return f"{table.rstrip('*')}\n  WHERE event_date BETWEEN '{start:%Y%m%d}' AND '{end:%Y%m%d}'"


def user_snapshot_sql(
    dataset: DatasetConfig = DATASET,
    windows: WindowConfig = WINDOWS,
    wildcard: bool = True,
) -> str:
    journeys = [_check_event(e) for e in dataset.journey_events]
    passive = ", ".join(f"'{_check_event(e)}'" for e in dataset.passive_events)
    snaps = snapshot_dates(dataset, windows)
    if not snaps:
        raise ValueError("Date range too short for the observation and churn windows")

    obs, churn = windows.observation_days, windows.churn_days
    daily_counts = "\n".join(f"    COUNTIF(event_name = '{e}') AS n_{e.lower()}," for e in journeys)
    window_sums = "\n".join(f"    SUM(d.n_{e.lower()}) AS n_{e.lower()}_{obs}d," for e in journeys)
    # Read events from the earliest observation day to the last churn-window day only.
    start = snaps[0] - timedelta(days=obs)
    end = snaps[-1] + timedelta(days=churn)

    return _template("user_snapshot_features.sql").format(
        passive_events=passive,
        journey_daily_counts=daily_counts,
        journey_window_sums=window_sums,
        events_source=events_source(dataset, start, end, wildcard),
        snapshot_dates=", ".join(f"DATE '{d:%Y-%m-%d}'" for d in snaps),
        obs=obs,
        churn=churn,
    )


def event_profile_sql(dataset: DatasetConfig = DATASET, wildcard: bool = True) -> str:
    return _template("event_profile.sql").format(
        events_source=events_source(dataset, wildcard=wildcard)
    )
