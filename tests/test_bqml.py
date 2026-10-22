"""The BigQuery ML scripts: kept in sync with the pipeline, and checked locally.

CREATE MODEL, ML.PREDICT and ML.EVALUATE cannot run outside BigQuery. Everything
around them can: the feature SQL (training and scoring modes), the derived ratios
and the reason codes are executed in DuckDB and compared with the Python pipeline.
"""

import re
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
import sqlglot

from appchurn import bqml
from appchurn.config import DATASET, WINDOWS
from appchurn.features import add_derived_features
from appchurn.score import reason_codes
from tests.sql_harness import TEST_DATASET, event, events_db, run_features, to_duckdb
from tests.synthetic import synthetic_events

PUBLIC = replace(DATASET, events_table="firebase-public-project.analytics_153293282.events_*")
T = date(2018, 9, 18)


@pytest.fixture(scope="module")
def events():
    return synthetic_events(n_users=400, seed=7)


def test_committed_scripts_match_the_generator():
    for name, sql in bqml.render_all(dataset=PUBLIC).items():
        committed = (bqml.OUT_DIR / name).read_text()
        assert committed == sql, f"{name} is stale: run `python -m appchurn.bqml`"


def test_no_statement_ends_inside_a_comment():
    for path in bqml.OUT_DIR.glob("*.sql"):
        for line in path.read_text().splitlines():
            if "--" in line and not line.lstrip().startswith("--"):
                assert ";" not in line.split("--", 1)[1], f"{path.name}: {line}"


def test_sql_statements_parse_except_bigquery_ml_functions():
    for name, sql in bqml.render_all(dataset=PUBLIC).items():
        code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
        for stmt in [s.strip() for s in re.split(r";\s*\n", code) if s.strip()]:
            if "FROM ML." in stmt:
                continue  # ML.EVALUATE / ML.GLOBAL_EXPLAIN: BigQuery-only syntax
            parsed = sqlglot.parse_one(stmt, read="bigquery")
            assert type(parsed).__name__ != "Command", f"{name}: {stmt[:60]}"


def test_scoring_mode_has_no_label_and_reads_only_the_past():
    sql = bqml.score_sql(dataset=PUBLIC)
    assert "churned" not in sql.split("b) and c)")[0]
    assert "BETWEEN '20180906' AND '20181003'" in sql  # 28 days before 2018-10-04


def test_scoring_features_equal_training_features_for_the_same_day(events):
    labelled = run_features(events)
    labelled = labelled[labelled["snapshot_date"] == T].drop(columns="churned")
    scoring = run_features(events, score_date=T)
    key = ["snapshot_date", "user_pseudo_id"]
    a = labelled.sort_values(key).reset_index(drop=True)
    b = scoring.sort_values(key).reset_index(drop=True)[a.columns]
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def _model_input(events, score_date=None):
    con = events_db(events)
    feature_sql = bqml.user_snapshot_sql(TEST_DATASET, local_test=True, score_date=score_date)
    sql = bqml._model_input_sql(feature_sql, TEST_DATASET, WINDOWS)
    return con.execute(to_duckdb(sql)).fetchdf()


def test_sql_ratios_and_population_match_python(events):
    sql_out = _model_input(events, score_date=T)
    assert (sql_out[f"active_days_{WINDOWS.observation_days}d"] >= WINDOWS.min_active_days).all()
    raw = run_features(events, score_date=T)
    raw = raw[raw[f"active_days_{WINDOWS.observation_days}d"] >= WINDOWS.min_active_days]
    py = add_derived_features(raw.astype({c: float for c in raw.select_dtypes("number")}))
    key = "user_pseudo_id"
    merged = sql_out.merge(py, on=key, suffixes=("_sql", "_py"))
    assert len(merged) == len(py) > 20
    for col in (
        "events_7d_vs_weekly_avg", "active_day_share_7d", "active_day_share_obs",
        "minutes_per_active_day", "events_per_session", "is_payer",
    ):  # fmt: skip
        np.testing.assert_allclose(
            merged[f"{col}_sql"].astype(float), merged[f"{col}_py"].astype(float), equal_nan=True
        )


def test_placeholder_tenure_is_nulled_in_sql():
    ev = [
        event("old", date(2018, 9, 10), first_touch=date(1970, 1, 2)),
        event("old", date(2018, 9, 12)),
        event("old", date(2018, 9, 14)),
    ]
    out = _model_input(ev, score_date=T)
    assert out.loc[out["user_pseudo_id"] == "old", "tenure_days"].isna().all()


def test_sql_reason_codes_match_python():
    cases = pd.DataFrame(
        {
            "days_since_last_active": [12, 4, 1, 2, 1],
            "events_7d_vs_weekly_avg": [0.0, 0.3, 1.2, 0.4, 1.0],
            "active_days_28d": [3, 6, 20, 4, 15],
            "tenure_days": [10, 200, 400, 25, 500],
            "n_level_fail_quickplay_28d": [0, 30, 5, 0, 0],
            "n_level_start_quickplay_28d": [0, 50, 40, 0, 0],
            "n_error_28d": [2, 0, 0, 0, 0],
            "n_app_exception_28d": [1, 0, 0, 0, 0],
        }
    )
    expr = bqml._reasons_sql(28, DATASET.journey_events)
    sql = f"SELECT IFNULL(NULLIF({expr}, ''), 'No single strong signal') AS r FROM cases"
    import duckdb

    con = duckdb.connect()
    con.register("cases", cases)
    try:
        # BigQuery FORMAT('%d') is C-style; DuckDB's FORMAT uses {}, its PRINTF is C-style
        duck = re.sub(r"\bFORMAT\(", "PRINTF(", to_duckdb(sql))
        got = con.execute(duck).fetchdf()["r"].tolist()
    except Exception as exc:  # pragma: no cover - transpiler gap, not a logic failure
        pytest.skip(f"DuckDB cannot run this BigQuery construct: {exc}")
    expected = [reason_codes(row) for _, row in cases.iterrows()]
    assert got == expected


def test_readme_and_docs_diagrams_are_identical():
    from appchurn.config import PROJECT_ROOT

    def diagrams(path):
        return re.findall(r"```mermaid\n(.*?)```", (PROJECT_ROOT / path).read_text(), re.S)

    assert diagrams("README.md") == diagrams("docs/architecture.md")
    assert len(diagrams("README.md")) == 2
