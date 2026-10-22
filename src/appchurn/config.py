"""Everything that changes between datasets lives here.

To run the same pipeline on another Firebase / GA4 BigQuery export (for example a
company's own app, inside that company's own Google Cloud project), change
``EVENTS_TABLE``, the date range and ``JOURNEY_EVENTS``. Nothing else needs to move.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_PATH = DATA_DIR / "user_snapshots.parquet"
EVENT_PROFILE_PATH = REPORTS_DIR / "event_profile.csv"

RANDOM_STATE = 42


@dataclass(frozen=True)
class DatasetConfig:
    # Google's public Flood-It! demo export (obfuscated GA4 / Firebase events).
    # Override with the EVENTS_TABLE environment variable, e.g. to use your own copy:
    #   EVENTS_TABLE=my-project.flood_it.events
    events_table: str = field(
        default_factory=lambda: os.environ.get(
            "EVENTS_TABLE", "firebase-public-project.analytics_153293282.events_*"
        )
    )
    data_start: date = date(2018, 6, 12)
    data_end: date = date(2018, 10, 3)
    # First-touch dates before this are placeholders in the obfuscated export
    # (1970-01-01-style epochs, 2015-01-01); the app's real installs start in May 2016.
    # Tenures implying an earlier start are treated as unknown.
    earliest_valid_first_touch: date = date(2016, 1, 1)

    # Business-meaningful actions ("journeys"), counted per user in the observation
    # window. For a telecom app these would be e.g. login, bill_payment, recharge,
    # bundle_purchase, roaming_activation, support_chat.
    journey_events: tuple[str, ...] = (
        "level_start_quickplay",
        "level_end_quickplay",
        "level_complete_quickplay",
        "level_reset_quickplay",
        "post_score",
        "spend_virtual_currency",
        "ad_reward",
        "challenge_a_friend",
        "completed_5_levels",
        "use_extra_steps",
        "in_app_purchase",
        # Friction and frustration: failed attempts, running out, errors, crashes
        "level_fail_quickplay",
        "level_retry_quickplay",
        "no_more_extra_steps",
        "error",
        "app_exception",
        # Breadth of use: browsing content, the second game mode, progression
        "select_content",
        "level_start",
        "level_up",
    )

    # Automatic events that do not show the user actually chose to use the app.
    # They are excluded when deciding whether a user was "active" on a day.
    passive_events: tuple[str, ...] = (
        "app_remove",
        "app_update",
        "app_clear_data",
        "os_update",
        "app_exception",
        "notification_receive",
        "notification_dismiss",
        "firebase_campaign",
        "dynamic_link_app_update",
    )


@dataclass(frozen=True)
class WindowConfig:
    """Snapshot design: the standard set-up for telecom churn models.

    At each snapshot date T:
      * features describe the user's behaviour in [T - observation_days, T)
      * the user is in the SQL output if they were active at least once in that
        window, and in the model population if active on ``min_active_days`` days
      * churned = 1 if the user has no active event in [T, T + churn_days)
    """

    observation_days: int = 28
    churn_days: int = 14
    # Model population: regular users only, as telecom retention models score the
    # established base. One-off visitors (most of a casual game's traffic) churn at
    # ~87% and are an onboarding problem, not a retention one.
    min_active_days: int = 3
    snapshot_every_days: int = 7
    # Snapshots on or after this date form the out-of-time test set.
    test_from: date = date(2018, 9, 4)


@dataclass(frozen=True)
class BigQueryConfig:
    project: str | None = field(default_factory=lambda: os.environ.get("GCP_PROJECT"))
    location: str = "US"
    # Hard cap: BigQuery refuses to run the query if it would scan more than this.
    max_gb_billed: float = float(os.environ.get("MAX_GB_BILLED", "10"))


DATASET = DatasetConfig()
WINDOWS = WindowConfig()
BIGQUERY = BigQueryConfig()


def snapshot_dates(dataset: DatasetConfig = DATASET, windows: WindowConfig = WINDOWS) -> list[date]:
    """Every snapshot date with a full observation window and a full churn window."""
    first = dataset.data_start + timedelta(days=windows.observation_days)
    last = dataset.data_end - timedelta(days=windows.churn_days)
    out, d = [], first
    while d <= last:
        out.append(d)
        d += timedelta(days=windows.snapshot_every_days)
    return out
