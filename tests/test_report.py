"""Scoring and the HTML report, end to end on synthetic GA4 events."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from appchurn.config import WINDOWS
from appchurn.report import build, fmt, shift
from appchurn.score import EXPORT_COLUMNS, assign_bands, export, reason_codes, score
from appchurn.train import run
from tests.sql_harness import run_features
from tests.synthetic import synthetic_events


@pytest.fixture(scope="module")
def snapshots() -> pd.DataFrame:
    return run_features(synthetic_events(n_users=500, seed=3))


def test_bands_split_20_30_50_by_rank():
    p = np.linspace(0, 1, 1000)
    bands = assign_bands(p)
    assert (bands == "High").sum() == 200
    assert (bands == "Medium").sum() == 300
    assert (bands == "Low").sum() == 500
    assert set(bands[p >= 0.81]) == {"High"}  # the riskiest are High


def test_reason_codes_are_plain_language_and_capped():
    row = pd.Series(
        {
            "days_since_last_active": 12,
            "events_7d_vs_weekly_avg": 0.2,
            "active_days_28d": 3,
            "tenure_days": 10,
            "n_error_28d": 2,
            "n_app_exception_28d": 1,
        }
    )
    text = reason_codes(row)
    assert text.split("; ") == ["Silent 12 days", "Only 3 active days in 28", "New user (10 days)"]
    calm = pd.Series({"days_since_last_active": 1, "active_days_28d": 20, "tenure_days": 400})
    assert reason_codes(calm) == "No single strong signal"


def test_scoring_model_only_learns_from_outcomes_known_by_the_score_date(snapshots):
    result = score(snapshots)
    gap = timedelta(days=WINDOWS.churn_days)
    assert all(d + gap <= result.score_date for d in result.train_snapshots)
    assert result.scored["rank"].tolist() == list(range(1, len(result.scored) + 1))
    assert result.scored["churn_score"].is_monotonic_decreasing


def test_export_never_contains_outcomes(snapshots, tmp_path):
    result = score(snapshots)
    path = tmp_path / "at_risk.csv"
    export(result, path)
    out = pd.read_csv(path)
    assert "churned" not in out.columns
    assert list(out.columns) == ["score_date", *EXPORT_COLUMNS]


def test_display_shift_keeps_the_weekday():
    d = date(2018, 9, 18)
    assert shift(d, 365).weekday() == d.weekday()
    assert fmt(d, 365) == "Tue 16 Sep 2025"


def test_report_builds_with_every_section(snapshots, tmp_path):
    run(snapshots, reports_dir=tmp_path / "reports", models_dir=tmp_path / "models", fast=True)
    out = build(
        snapshots,
        out_path=tmp_path / "site" / "index.html",
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
    )
    html = out.read_text()
    assert "__REPORT_DATA__" not in html
    for section in (
        "summary", "customers", "reasons", "planner", "performance",
        "segments", "drivers", "calibration", "quality", "method",
    ):  # fmt: skip
        assert f'id="{section}"' in html
    assert "Demo dates." in html
    assert (tmp_path / "reports" / "at_risk_customers.csv").exists()
