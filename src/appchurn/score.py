"""Score users for churn risk as of a snapshot date: the production step.

    python -m appchurn.score                  # latest snapshot in the extract
    python -m appchurn.score --date 2018-09-11

The model used here is fitted only on snapshots whose 14-day outcome was already
known on the score date, exactly as a scheduled job would be. When the extract also
contains the outcome for the score date (true for every historical snapshot), it is
kept aside for the report's backtest and never used for scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from appchurn.config import FEATURES_PATH, REPORTS_DIR, WINDOWS, WindowConfig
from appchurn.features import CATEGORICAL, TARGET, clean, model_population, numeric_features
from appchurn.model import candidate_models

DEFAULT_MODEL = "Gradient boosting"

# Rank-based bands: probabilities drift between periods (see the calibration section
# of the report), rankings are far more stable.
BANDS = (("High", 0.20), ("Medium", 0.30), ("Low", 0.50))


@dataclass
class ScoreResult:
    score_date: date
    scored: pd.DataFrame  # one row per user, ranked, with band and reasons
    outcomes: pd.Series | None  # actual churn for the score date, if already known
    train_rows: int
    train_snapshots: list[date]
    model_name: str


def assign_bands(probabilities: np.ndarray) -> np.ndarray:
    """Top 20% High, next 30% Medium, remaining 50% Low, by rank."""
    n = len(probabilities)
    order = np.argsort(-probabilities, kind="stable")
    bands = np.empty(n, dtype=object)
    start = 0
    for i, (name, share) in enumerate(BANDS):
        end = n if i == len(BANDS) - 1 else start + int(round(n * share))
        bands[order[start:end]] = name
        start = end
    return bands


def reason_codes(row: pd.Series, obs: int = WINDOWS.observation_days, limit: int = 3) -> str:
    """Plain-language reasons a user ranks as risky, most important first.

    Descriptive only: they explain the ranking, they do not prove that acting on a
    reason will change the outcome.
    """
    reasons: list[str] = []
    silent = row.get("days_since_last_active")
    if pd.notna(silent) and silent >= 7:
        reasons.append(f"Silent {int(silent)} days")
    elif pd.notna(silent) and silent >= 3:
        reasons.append(f"Not seen for {int(silent)} days")

    trend = row.get("events_7d_vs_weekly_avg")
    if pd.notna(trend) and trend < 0.5 and (pd.isna(silent) or silent < 7):
        reasons.append(f"Activity down {int(round((1 - trend) * 100))}% vs usual week")

    active = row.get(f"active_days_{obs}d")
    if pd.notna(active) and active <= 4:
        reasons.append(f"Only {int(active)} active days in {obs}")

    tenure = row.get("tenure_days")
    if pd.notna(tenure) and tenure < 30:
        reasons.append(f"New user ({int(tenure)} days)")

    fails = row.get(f"n_level_fail_quickplay_{obs}d")
    starts = row.get(f"n_level_start_quickplay_{obs}d")
    if pd.notna(fails) and pd.notna(starts) and starts >= 10 and fails / starts >= 0.4:
        reasons.append(f"Fails often ({int(round(fails / starts * 100))}% of attempts)")

    errors = sum(row.get(c, 0) or 0 for c in (f"n_error_{obs}d", f"n_app_exception_{obs}d"))
    if errors > 0:
        reasons.append(f"Hit app errors ({int(errors)})")

    return "; ".join(reasons[:limit]) if reasons else "No single strong signal"


def latest_snapshot(df: pd.DataFrame) -> date:
    return max(df["snapshot_date"])


def score(
    df: pd.DataFrame,
    score_date: date | None = None,
    windows: WindowConfig = WINDOWS,
    model_name: str = DEFAULT_MODEL,
) -> ScoreResult:
    data = model_population(clean(df))
    score_date = score_date or latest_snapshot(data)
    gap = timedelta(days=windows.churn_days)

    train = data[data["snapshot_date"] + gap <= score_date]
    today = data[data["snapshot_date"] == score_date].copy()
    if train.empty or today.empty:
        raise ValueError(f"No training history or no users to score on {score_date}")

    numeric = numeric_features(data)
    pipe = candidate_models(numeric)[model_name]
    pipe.fit(train[numeric + CATEGORICAL], train[TARGET])

    outcomes = today[TARGET].copy() if TARGET in today else None
    today = today.drop(columns=[TARGET], errors="ignore")
    today["churn_score"] = pipe.predict_proba(today[numeric + CATEGORICAL])[:, 1]
    today["risk_band"] = assign_bands(today["churn_score"].to_numpy())
    today["reasons"] = today.apply(reason_codes, axis=1)
    today = today.sort_values("churn_score", ascending=False)
    today["rank"] = np.arange(1, len(today) + 1)
    if outcomes is not None:
        outcomes = outcomes.loc[today.index]
    today = today.reset_index(drop=True)
    if outcomes is not None:
        outcomes = outcomes.reset_index(drop=True)

    return ScoreResult(
        score_date=score_date,
        scored=today,
        outcomes=outcomes,
        train_rows=len(train),
        train_snapshots=sorted(train["snapshot_date"].unique()),
        model_name=model_name,
    )


EXPORT_COLUMNS = [
    "rank",
    "user_pseudo_id",
    "risk_band",
    "churn_score",
    "reasons",
    "days_since_last_active",
    f"active_days_{WINDOWS.observation_days}d",
    "active_days_7d",
    "events_7d_vs_weekly_avg",
    "tenure_days",
    "platform",
    "country",
    "device_category",
]


def export(result: ScoreResult, path=REPORTS_DIR / "at_risk_customers.csv") -> None:
    """The file a campaign platform would pick up. Outcomes are never included."""
    out = result.scored[EXPORT_COLUMNS].copy()
    out.insert(0, "score_date", result.score_date)
    out["churn_score"] = out["churn_score"].round(4)
    out["events_7d_vs_weekly_avg"] = out["events_7d_vs_weekly_avg"].round(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score users for churn risk")
    parser.add_argument("--date", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    import db_dtypes  # noqa: F401  older extracts store BigQuery date types

    result = score(pd.read_parquet(FEATURES_PATH), args.date)
    export(result)
    counts = result.scored["risk_band"].value_counts()
    print(
        f"Scored {len(result.scored):,} users as of {result.score_date} "
        f"(model fitted on {result.train_rows:,} rows): "
        + ", ".join(f"{b} {counts.get(b, 0):,}" for b, _ in BANDS)
    )


if __name__ == "__main__":
    main()
