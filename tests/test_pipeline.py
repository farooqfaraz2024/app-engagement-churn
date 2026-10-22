"""End to end without Google Cloud: synthetic events -> production SQL (in DuckDB)
-> features -> out-of-time split -> training -> reports."""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from appchurn.config import WINDOWS
from appchurn.features import clean, numeric_features, out_of_time_split
from appchurn.model import gains_table
from appchurn.train import run
from tests.sql_harness import run_features
from tests.synthetic import synthetic_events


@pytest.fixture(scope="module")
def snapshots() -> pd.DataFrame:
    return run_features(synthetic_events())


def test_snapshot_table_shape(snapshots):
    assert len(snapshots) > 500
    assert snapshots["churned"].isin([0, 1]).all()
    assert not snapshots.duplicated(["snapshot_date", "user_pseudo_id"]).any()
    assert 0.05 < snapshots["churned"].mean() < 0.95


def test_out_of_time_split_has_no_overlap(snapshots):
    train, test = out_of_time_split(clean(snapshots))
    gap = timedelta(days=WINDOWS.churn_days)
    assert (train["snapshot_date"] + gap <= WINDOWS.test_from).all()
    assert (test["snapshot_date"] >= WINDOWS.test_from).all()


def test_no_label_or_id_in_features(snapshots):
    cols = numeric_features(clean(snapshots))
    assert "churned" not in cols
    assert "user_pseudo_id" not in cols


def test_gains_table_is_monotonic_and_ends_at_100pct():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    g = gains_table(y, rng.uniform(size=1000))
    assert g["share_of_churners_reached"].is_monotonic_increasing
    assert g["share_of_churners_reached"].iloc[-1] == pytest.approx(1.0)


def test_full_training_run(snapshots, tmp_path):
    metrics = run(
        snapshots, reports_dir=tmp_path / "reports", models_dir=tmp_path / "models", fast=True
    )
    # The synthetic users fade before quitting, so a working pipeline must find signal.
    assert metrics["roc_auc"] > 0.65
    assert (tmp_path / "models" / "app_churn_model.joblib").exists()
    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert saved["selected_model"] in {"Logistic regression", "Gradient boosting"}
    for fig in ("gains_curve.png", "feature_importance.png", "churn_by_recency.png"):
        assert (tmp_path / "reports" / "figures" / fig).exists()


def test_placeholder_first_touch_tenure_becomes_missing(snapshots):
    df = snapshots.copy()
    df.loc[df.index[0], "tenure_days"] = 17_747  # first touch in 1970
    cleaned = clean(df)
    assert np.isnan(cleaned.loc[df.index[0], "tenure_days"])
    assert cleaned["tenure_days"].notna().sum() == len(df) - 1


def test_forward_chaining_never_fits_on_the_future(snapshots):
    from appchurn.features import forward_chaining_splits

    train, _ = out_of_time_split(clean(snapshots))
    gap = timedelta(days=WINDOWS.churn_days)
    splits = forward_chaining_splits(train)
    assert len(splits) == 3
    for fit, val in splits:
        val_snap = train["snapshot_date"].iloc[val].unique()
        assert len(val_snap) == 1
        assert (train["snapshot_date"].iloc[fit] + gap <= val_snap[0]).all()
