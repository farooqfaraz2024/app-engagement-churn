"""The production monitor, on synthetic GA4 events with known behaviour."""

import numpy as np
import pandas as pd
import pytest

from appchurn import monitor
from appchurn.config import WINDOWS
from tests.sql_harness import run_features
from tests.synthetic import synthetic_events

OBS = WINDOWS.observation_days


@pytest.fixture(scope="module")
def snapshots() -> pd.DataFrame:
    return run_features(synthetic_events(n_users=500, seed=11))


@pytest.fixture(scope="module")
def result(snapshots) -> dict:
    return monitor.run(snapshots)


def test_psi_is_zero_for_identical_and_large_for_shifted_data():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert monitor.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert monitor.psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert monitor.psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_every_week_is_scored_only_with_known_outcomes(result):
    weeks = result["weeks"]
    assert len(weeks) >= 3
    for w in weeks:
        assert w["train_rows"] > 0
        assert {c["check"] for c in w["checks"]} >= {
            "Beats the recency rule",
            "Probabilities match reality",
            "Inputs match training data",
            "Event tracking healthy",
        }
        assert w["status"] in {"ok", "warning", "alert"}


def test_worst_check_sets_the_week_status(result):
    rank = {"ok": 0, "pending": 0, "warning": 1, "alert": 2}
    for w in result["weeks"]:
        worst = max((c["status"] for c in w["checks"]), key=rank.get)
        assert rank[w["status"]] == rank[worst]


def test_a_broken_event_triggers_a_tracking_alert(snapshots):
    broken = snapshots.copy()
    last = broken["snapshot_date"].max()
    col = f"n_level_start_quickplay_{OBS}d"
    broken.loc[broken["snapshot_date"] == last, col] = 0  # the app stopped sending it
    result = monitor.run(broken)
    latest = {c["check"]: c for c in result["latest"]["checks"]}
    assert latest["Event tracking healthy"]["status"] == "alert"
    assert "level_start_quickplay stopped" in latest["Event tracking healthy"]["detail"]
    assert result["latest"]["status"] == "alert"


def test_score_shift_is_a_warning_never_an_alert(result):
    for w in result["weeks"]:
        scores = next(c for c in w["checks"] if c["check"] == "Scores stable")
        assert scores["status"] in {"ok", "warning"}


def test_outputs_are_written(result, tmp_path):
    monitor.write(result, tmp_path)
    weekly = pd.read_csv(tmp_path / "monitoring_weekly.csv")
    assert len(weekly) == len(result["weeks"])
    assert "checks" not in weekly.columns
    assert (tmp_path / "monitoring.json").exists()
