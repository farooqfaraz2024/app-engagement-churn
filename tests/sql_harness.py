"""Run the real BigQuery SQL locally against a tiny, hand-built GA4-shaped table.

The production SQL is translated from BigQuery to DuckDB with sqlglot, so the
tests exercise the same logic that runs in BigQuery: windows, labels, journey
counts and nested ``event_params`` unnesting. No Google account is needed.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone

import duckdb
import pandas as pd
import sqlglot

from appchurn.config import DATASET
from appchurn.sql import user_snapshot_sql

TEST_DATASET = replace(DATASET, events_table="proj.ds.events_*")

DDL = """
CREATE TABLE events (
  event_date VARCHAR,
  event_timestamp BIGINT,
  event_name VARCHAR,
  user_pseudo_id VARCHAR,
  user_first_touch_timestamp BIGINT,
  event_value_in_usd DOUBLE,
  platform VARCHAR,
  event_params STRUCT(
    key VARCHAR,
    value STRUCT(string_value VARCHAR, int_value BIGINT, float_value DOUBLE, double_value DOUBLE)
  )[],
  geo STRUCT(country VARCHAR),
  device STRUCT(category VARCHAR)
)
"""


def event(
    user: str,
    day: date,
    name: str = "user_engagement",
    engagement_ms: int | None = None,
    revenue: float | None = None,
    first_touch: date | None = None,
    platform: str = "ANDROID",
    country: str = "Germany",
) -> dict:
    params = []
    if engagement_ms is not None:
        params.append(
            {
                "key": "engagement_time_msec",
                "value": {
                    "string_value": None,
                    "int_value": engagement_ms,
                    "float_value": None,
                    "double_value": None,
                },
            }
        )
    touch = first_touch or day
    return {
        "event_date": f"{day:%Y%m%d}",
        "event_timestamp": int(
            datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc).timestamp() * 1e6
        ),
        "event_name": name,
        "user_pseudo_id": user,
        "user_first_touch_timestamp": int(
            datetime(touch.year, touch.month, touch.day, tzinfo=timezone.utc).timestamp() * 1e6
        ),
        "event_value_in_usd": revenue,
        "platform": platform,
        "event_params": params,
        "geo": {"country": country},
        "device": {"category": "mobile"},
    }


def duckdb_sql() -> str:
    bq = user_snapshot_sql(TEST_DATASET, local_test=True)
    sql = sqlglot.transpile(bq, read="bigquery", write="duckdb")[0]
    return sql.replace("proj.ds.events_", "events")


def run_features(events: list[dict]) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(DDL)
    con.register("staging", pd.DataFrame(events))
    con.execute("INSERT INTO events SELECT * FROM staging")
    df = con.execute(duckdb_sql()).fetchdf()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    return df


def row(df: pd.DataFrame, user: str, snapshot: date) -> pd.Series | None:
    hit = df[(df["user_pseudo_id"] == user) & (df["snapshot_date"] == snapshot)]
    return None if hit.empty else hit.iloc[0]
