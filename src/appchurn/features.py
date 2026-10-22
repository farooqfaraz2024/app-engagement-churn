"""Turn the BigQuery snapshot table into model inputs and split it out-of-time."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from appchurn.config import DATASET, WINDOWS, DatasetConfig, WindowConfig

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


def clean(df: pd.DataFrame, dataset: DatasetConfig = DATASET) -> pd.DataFrame:
    df = df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.date
    # BigQuery integers arrive as nullable Int64; scikit-learn wants plain floats.
    for col in df.columns:
        if str(df[col].dtype) in {"Int64", "Int32", "Float64"}:
            df[col] = df[col].astype("float64")
    # Placeholder first-touch timestamps would give tenures of decades.
    max_tenure = pd.Series(
        [(d - dataset.earliest_valid_first_touch).days for d in df["snapshot_date"]],
        index=df.index,
    )
    df["tenure_days"] = df["tenure_days"].where(df["tenure_days"] <= max_tenure)
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").replace({"": "Unknown"}).astype(str)
    top = df["country"].value_counts().head(TOP_COUNTRIES).index
    df["country"] = df["country"].where(df["country"].isin(top), "Other")
    df[TARGET] = df[TARGET].astype(int)
    return add_derived_features(df)


def numeric_features(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS) | set(CATEGORICAL) | {TARGET}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def model_population(df: pd.DataFrame, windows: WindowConfig = WINDOWS) -> pd.DataFrame:
    """Regular users: active on at least ``min_active_days`` days of the window."""
    col = f"active_days_{windows.observation_days}d"
    return df[df[col] >= windows.min_active_days].reset_index(drop=True)


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


def forward_chaining_splits(
    df: pd.DataFrame, windows: WindowConfig = WINDOWS, n_validation: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validation that respects time, inside the training period.

    Each of the last ``n_validation`` snapshots is validated by a model fitted only
    on snapshots whose churn window had closed by then. Random or user-grouped folds
    mix weeks together and overstate performance on the future; this does not.
    """
    gap = timedelta(days=windows.churn_days)
    snaps = sorted(df["snapshot_date"].unique())
    splits = []
    for val_snap in snaps[-n_validation:]:
        fit = np.flatnonzero((df["snapshot_date"] + gap <= val_snap).to_numpy())
        val = np.flatnonzero((df["snapshot_date"] == val_snap).to_numpy())
        if len(fit) and len(val):
            splits.append((fit, val))
    if not splits:
        raise ValueError("Not enough snapshots for forward-chaining validation")
    return splits
