"""Turn the BigQuery snapshot table into model inputs and split it out-of-time."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from appchurn.config import WINDOWS, WindowConfig

ID_COLUMNS = ["snapshot_date", "user_pseudo_id"]
TARGET = "churned"
CATEGORICAL = ["platform", "country", "device_category"]
TOP_COUNTRIES = 15


def add_derived_features(df: pd.DataFrame, windows: WindowConfig = WINDOWS) -> pd.DataFrame:
    """Ratios that describe *trend*, which raw counts do not.

    A customer whose activity is fading looks different from one who was always
    light, even with the same 28-day total. Telecom churn models rely on exactly
    this kind of "last week vs. usual" signal.
    """
    obs = windows.observation_days
    df = df.copy()
    weeks = obs / 7
    weekly_avg_events = df[f"events_{obs}d"] / weeks
    df["events_7d_vs_weekly_avg"] = df["events_7d"] / weekly_avg_events.replace(0, np.nan)
    df["active_day_share_7d"] = df["active_days_7d"] / 7
    df["active_day_share_obs"] = df[f"active_days_{obs}d"] / obs
    df["minutes_per_active_day"] = df[f"engagement_minutes_{obs}d"] / df[f"active_days_{obs}d"]
    df["events_per_session"] = df[f"events_{obs}d"] / df[f"sessions_{obs}d"].replace(0, np.nan)
    df["is_payer"] = (df[f"revenue_usd_{obs}d"] > 0).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").replace({"": "Unknown"}).astype(str)
    top = df["country"].value_counts().head(TOP_COUNTRIES).index
    df["country"] = df["country"].where(df["country"].isin(top), "Other")
    df[TARGET] = df[TARGET].astype(int)
    return add_derived_features(df)


def numeric_features(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS) | set(CATEGORICAL) | {TARGET}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def out_of_time_split(
    df: pd.DataFrame, windows: WindowConfig = WINDOWS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier snapshots, test on later ones, with no overlap in time.

    A training snapshot is kept only if its churn window ends before the first test
    snapshot, so no label in training "sees" the test period. This mirrors
    production, where a model trained on past months scores the current month.
    """
    test_from = windows.test_from
    gap = timedelta(days=windows.churn_days)
    train = df[df["snapshot_date"] + gap <= test_from]
    test = df[df["snapshot_date"] >= test_from]
    if train.empty or test.empty:
        raise ValueError("Out-of-time split produced an empty train or test set")
    return train.reset_index(drop=True), test.reset_index(drop=True)
