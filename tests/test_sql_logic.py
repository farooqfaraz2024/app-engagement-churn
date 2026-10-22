"""Behaviour tests for the feature/label SQL, executed locally in DuckDB.

Snapshot used throughout: T = 2018-08-07 (observation window 2018-07-10 to
2018-08-06, churn window 2018-08-07 to 2018-08-20).
"""

from datetime import date, timedelta

import pytest
import sqlglot

from appchurn.sql import event_profile_sql, user_snapshot_sql
from tests.sql_harness import event, row, run_features

T = date(2018, 8, 7)
OBS_START = T - timedelta(days=28)
CHURN_END = T + timedelta(days=14)  # exclusive


def d(offset: int) -> date:
    return T + timedelta(days=offset)


def test_production_sql_is_valid_bigquery():
    sqlglot.parse_one(user_snapshot_sql(), read="bigquery")
    sqlglot.parse_one(event_profile_sql(), read="bigquery")


def test_user_who_returns_is_not_churned():
    df = run_features([event("u", d(-3)), event("u", d(+5))])
    assert row(df, "u", T)["churned"] == 0


def test_user_who_goes_silent_is_churned():
    df = run_features([event("u", d(-3))])
    assert row(df, "u", T)["churned"] == 1


def test_only_passive_events_after_snapshot_still_counts_as_churn():
    # An uninstall or OS update is not the customer choosing to use the app.
    df = run_features(
        [event("u", d(-3)), event("u", d(+2), "app_remove"), event("u", d(+4), "os_update")]
    )
    assert row(df, "u", T)["churned"] == 1


def test_users_with_only_passive_events_are_out_of_scope():
    df = run_features([event("u", d(-3), "os_update"), event("other", d(-1))])
    assert row(df, "u", T) is None
    assert row(df, "other", T) is not None


def test_window_boundaries():
    df = run_features(
        [
            event("first_day_in", OBS_START),  # inside observation window
            event("before_window", OBS_START - timedelta(days=1)),  # outside
            event("returns_last_day", d(-1)),
            event("returns_last_day", CHURN_END - timedelta(days=1)),  # inside churn window
            event("returns_too_late", d(-1)),
            event("returns_too_late", CHURN_END),  # just outside churn window
        ]
    )
    assert row(df, "first_day_in", T) is not None
    assert row(df, "before_window", T) is None
    assert row(df, "returns_last_day", T)["churned"] == 0
    assert row(df, "returns_too_late", T)["churned"] == 1


def test_event_on_snapshot_day_is_future_not_feature():
    # Guards against label leakage: day T belongs to the churn window only.
    df = run_features([event("u", d(-10)), event("u", T, "level_start_quickplay")])
    r = row(df, "u", T)
    assert r["n_level_start_quickplay_28d"] == 0
    assert r["churned"] == 0


def test_activity_recency_and_counts():
    df = run_features(
        [
            event("u", d(-20)),
            event("u", d(-6), "session_start"),
            event("u", d(-6), "level_start_quickplay"),
            event("u", d(-6), "level_start_quickplay"),
            event("u", d(-2), "spend_virtual_currency"),
        ]
    )
    r = row(df, "u", T)
    assert r["active_days_28d"] == 3
    assert r["active_days_7d"] == 2
    assert r["events_28d"] == 5
    assert r["events_7d"] == 4
    assert r["sessions_28d"] == 1
    assert r["n_level_start_quickplay_28d"] == 2
    assert r["n_spend_virtual_currency_28d"] == 1
    assert r["days_since_last_active"] == 2


def test_engagement_minutes_and_revenue():
    df = run_features(
        [
            event("u", d(-10), engagement_ms=90_000),
            event("u", d(-3), engagement_ms=30_000),
            event("u", d(-3), "in_app_purchase", revenue=4.99),
        ]
    )
    r = row(df, "u", T)
    assert r["engagement_minutes_28d"] == pytest.approx(2.0)
    assert r["engagement_minutes_7d"] == pytest.approx(0.5)
    assert r["revenue_usd_28d"] == pytest.approx(4.99)
    assert r["n_in_app_purchase_28d"] == 1


def test_tenure_uses_first_touch_timestamp():
    df = run_features([event("u", d(-3), first_touch=date(2018, 6, 20))])
    assert row(df, "u", T)["tenure_days"] == (T - date(2018, 6, 20)).days


def test_same_user_appears_in_each_snapshot_window():
    df = run_features([event("u", date(2018, 7, 20))])
    snaps = sorted(df.loc[df["user_pseudo_id"] == "u", "snapshot_date"])
    # 20 July is inside the 28-day window of the 24 Jul, 31 Jul, 7 Aug and 14 Aug snapshots
    assert snaps == [date(2018, 7, 24), date(2018, 7, 31), date(2018, 8, 7), date(2018, 8, 14)]
