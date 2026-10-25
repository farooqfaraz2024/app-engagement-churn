"""The scheduled BigQuery ML scripts.

BigQuery-only parts (scripting, ML.PREDICT, EXPORT DATA, FARM_FINGERPRINT) cannot run
locally. Everything that decides a number can: run-time token substitution must
reproduce the tested demo SQL exactly, and the SQL ROC-AUC, outcome labels and
monitoring metrics are executed in DuckDB and compared with Python.
"""

import re
from dataclasses import replace
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from appchurn import bqml, bqml_scheduled
from appchurn.bqml import Target
from appchurn.config import DATASET, WINDOWS, snapshot_dates
from appchurn.sql import user_snapshot_sql
from tests.sql_harness import TEST_DATASET, events_db, run_features, to_duckdb
from tests.synthetic import synthetic_events

PUBLIC = replace(DATASET, events_table=bqml.PUBLIC_EVENTS_TABLE)
OBS, CHURN = WINDOWS.observation_days, WINDOWS.churn_days
SCRIPTS = bqml.OUT_DIR / "scheduled"


def _sub(sql: str, values: dict[str, str]) -> str:
    """What BigQuery's nested REPLACE() calls do at run time."""
    for token, value in values.items():
        sql = sql.replace(token, value)
    return sql


def _date_list(dates) -> str:
    return "[" + ", ".join(f"DATE '{d:%Y-%m-%d}'" for d in dates) + "]"


def _code(script: str) -> list[str]:
    return [ln for ln in script.splitlines() if ln.strip() and not ln.lstrip().startswith("--")]


def test_committed_scripts_match_the_generator():
    for name, sql in bqml_scheduled.render_all(dataset=PUBLIC).items():
        assert (SCRIPTS / name).read_text() == sql, f"{name} is stale: run python -m appchurn.bqml"


def test_declarations_come_first_and_no_statement_ends_in_a_comment():
    for path in SCRIPTS.glob("*.sql"):
        script = path.read_text()
        top_level = [ln for ln in _code(script) if not ln.startswith((" ", ")"))]
        kinds = ["DECLARE" if ln.startswith("DECLARE") else "other" for ln in top_level]
        if "DECLARE" in kinds:
            assert kinds.index("other") > max(i for i, k in enumerate(kinds) if k == "DECLARE"), (
                f"{path.name}: DECLARE after another statement"
            )
        for line in script.splitlines():
            if "--" in line and not line.lstrip().startswith("--"):
                assert ";" not in line.split("--", 1)[1], f"{path.name}: {line}"


def test_every_token_in_a_dynamic_block_gets_a_value():
    for path in SCRIPTS.glob("*.sql"):
        script = path.read_text()
        for block in bqml_scheduled.raw_blocks(script):
            for token in bqml_scheduled.TOKENS:
                if token in block:
                    assert f"'{token}'," in script, f"{path.name}: {token} never replaced"


def test_execute_refuses_a_token_without_a_value():
    with pytest.raises(ValueError, match="without a value"):
        bqml_scheduled.execute("SELECT '__DATE__', '__MODEL__'", [("__DATE__", "x")])


def test_scoring_block_equals_the_tested_demo_sql_at_run_time():
    run_date = date(2018, 9, 18)
    script = bqml_scheduled.score_daily_sql(dataset=PUBLIC, run_date=run_date)
    block = next(b for b in bqml_scheduled.raw_blocks(script) if "scoring_features" in b)
    resolved = _sub(
        block,
        {
            "__SNAPSHOTS__": _date_list([run_date]),
            "__FIRST_SUFFIX__": f"{run_date - timedelta(days=OBS):%Y%m%d}",
            "__LAST_SUFFIX__": f"{run_date - timedelta(days=1):%Y%m%d}",
        },
    )
    expected = bqml._model_input_sql(
        user_snapshot_sql(PUBLIC, score_date=run_date), PUBLIC, WINDOWS
    )
    assert resolved == f"CREATE OR REPLACE TABLE {Target().ref('scoring_features')} AS\n{expected}"


def test_training_block_equals_the_tested_demo_sql_at_run_time():
    snaps = snapshot_dates(PUBLIC, WINDOWS)
    test_from = WINDOWS.test_from
    block = next(
        b
        for b in bqml_scheduled.raw_blocks(bqml_scheduled.retrain_sql(dataset=PUBLIC))
        if "training_snapshots` AS" in b
    )
    resolved = _sub(
        block,
        {
            "__SNAPSHOTS__": _date_list(snaps),
            "__FIRST_SUFFIX__": f"{snaps[0] - timedelta(days=OBS):%Y%m%d}",
            "__LAST_SUFFIX__": f"{snaps[-1] + timedelta(days=CHURN):%Y%m%d}",
            "__FIRST_TEST__": f"{test_from:%Y-%m-%d}",
        },
    )
    demo = bqml.train_sql(dataset=PUBLIC)
    demo_block = demo.split("CREATE OR REPLACE TABLE ", 1)[1].split(";\n", 1)[0]
    assert resolved == "CREATE OR REPLACE TABLE " + demo_block


def test_sql_roc_auc_matches_scikit_learn_with_ties():
    rng = np.random.default_rng(3)
    n = 900
    df = pd.DataFrame(
        {
            "score_date": np.repeat(["a", "b", "c"], n // 3),
            "y": rng.integers(0, 2, n),
        }
    )
    df["s"] = np.round(df["y"] * 0.3 + rng.uniform(0, 1, n), 1)  # rounding creates ties
    rank, agg = bqml_scheduled.auc_sql("y", "s", "score_date")
    sql = (
        f"WITH r AS (SELECT score_date, y, {rank} AS rk FROM df) "
        f"SELECT score_date, {agg.format(rank='rk')} AS auc FROM r GROUP BY score_date"
    )
    con = duckdb.connect()
    con.register("df", df)
    got = con.execute(to_duckdb(sql)).fetchdf().set_index("score_date")["auc"]
    for key, part in df.groupby("score_date"):
        assert got[key] == pytest.approx(roc_auc_score(part["y"], part["s"]), abs=1e-9)


@pytest.fixture(scope="module")
def events():
    return synthetic_events(n_users=500, seed=21)


def _outcomes(events, at_risk: pd.DataFrame, earliest: date, latest: date) -> pd.DataFrame:
    target = Target(project="proj", dataset="ds")
    sql = _sub(
        bqml_scheduled.outcomes_sql(target, TEST_DATASET, WINDOWS, local_test=True),
        {
            "__FIRST_SUFFIX__": f"{earliest:%Y%m%d}",
            "__LAST_SUFFIX__": f"{latest + timedelta(days=CHURN - 1):%Y%m%d}",
            "__EARLIEST__": f"{earliest:%Y-%m-%d}",
            "__LATEST__": f"{latest:%Y-%m-%d}",
        },
    )
    select = sql.split(" AS\n", 1)[1]  # drop CREATE OR REPLACE TABLE for the local run
    con = events_db(events)
    con.register("at_risk", at_risk)
    duck = to_duckdb(select).replace('"proj"."ds"."at_risk_customers"', "at_risk")
    return con.execute(duck).fetchdf()


def test_outcome_sql_reproduces_the_training_label(events):
    t = date(2018, 9, 4)
    labelled = run_features(events)
    week = labelled[labelled["snapshot_date"] == t]
    at_risk = pd.DataFrame(
        {
            "score_date": pd.to_datetime([t] * len(week)),
            "user_pseudo_id": week["user_pseudo_id"].to_numpy(),
            "model_name": "m",
            "churn_probability": 0.5,
            "risk_band": "High",
            "in_holdout": False,
            "days_since_last_active": week["days_since_last_active"].to_numpy(),
            f"active_days_{OBS}d": week[f"active_days_{OBS}d"].to_numpy(),
        }
    )
    got = _outcomes(events, at_risk, t, t).set_index("user_pseudo_id")["churned"]
    expected = week.set_index("user_pseudo_id")["churned"]
    assert len(got) == len(expected) > 50
    assert (got.loc[expected.index].to_numpy() == expected.to_numpy()).all()


def test_monitoring_metrics_match_python():
    rng = np.random.default_rng(5)
    rows = []
    for d in ("2018-08-07", "2018-08-14"):
        y = rng.integers(0, 2, 400)
        p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 400), 0, 1)
        rows.append(
            pd.DataFrame(
                {
                    "score_date": pd.to_datetime(d),
                    "user_pseudo_id": [f"u{i}" for i in range(400)],
                    "model_name": "m1",
                    "churn_probability": p,
                    "risk_band": np.where(p > 0.7, "High", np.where(p > 0.4, "Medium", "Low")),
                    "in_holdout": rng.uniform(size=400) < 0.1,
                    "days_since_last_active": rng.integers(1, 28, 400),
                    f"active_days_{OBS}d": rng.integers(3, 28, 400),
                    "churned": y,
                }
            )
        )
    outcomes = pd.concat(rows, ignore_index=True)
    sql = bqml_scheduled.monitoring_metrics_sql(Target(project="proj", dataset="ds"))
    con = duckdb.connect()
    con.register("score_outcomes", outcomes)
    duck = to_duckdb(sql).replace('"proj"."ds"."score_outcomes"', "score_outcomes")
    result = con.execute(duck).fetchdf()
    got = result.set_index(pd.to_datetime(result["score_date"]))
    for d, part in outcomes.groupby("score_date"):
        row = got.loc[d]
        rule = part["days_since_last_active"] - 0.01 * part[f"active_days_{OBS}d"]
        assert row["roc_auc"] == pytest.approx(
            roc_auc_score(part["churned"], part["churn_probability"])
        )
        assert row["rule_roc_auc"] == pytest.approx(roc_auc_score(part["churned"], rule))
        assert row["mean_predicted"] == pytest.approx(part["churn_probability"].mean())
        assert row["actual_churn"] == pytest.approx(part["churned"].mean())
        hm = part[part["risk_band"].isin(["High", "Medium"])]
        assert row["holdout_churn"] == pytest.approx(hm[hm["in_holdout"]]["churned"].mean())
        assert row["status"] in {"ok", "warning", "alert"}


def test_retraining_window_logic():
    script = bqml_scheduled.retrain_sql(dataset=PUBLIC)
    assert "SET last_snapshot = DATE_SUB(run_date, INTERVAL 14 DAY);" in script
    assert (
        "candidate_auc > rule_auc AND (current_auc IS NULL OR candidate_auc >= current_auc)"
        in script
    )
    assert "DROP MODEL IF EXISTS" in script  # rejected candidates are removed
    assert re.search(r"data_split_method = 'NO_SPLIT'", script)
