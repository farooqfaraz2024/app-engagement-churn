"""Train and evaluate on the snapshot table produced by appchurn.extract.

    python -m appchurn.train

1. Out-of-time split: earlier snapshots train the model, later snapshots test it.
2. Model selection by forward-chaining validation inside the training period:
   each late training week is predicted by a model fitted on earlier weeks only.
3. Compare against two benchmarks: the churn rate, and a "recency rule" (rank by
   days since last activity), which is what a CRM team could do without a model.
4. Refit the best model and evaluate once on the test period: ROC-AUC, PR-AUC,
   calibration, gains/lift for campaign sizing, stability per snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from appchurn import plots
from appchurn.config import FEATURES_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, WINDOWS
from appchurn.features import (
    CATEGORICAL,
    TARGET,
    clean,
    forward_chaining_splits,
    model_population,
    numeric_features,
    out_of_time_split,
)
from appchurn.model import candidate_models, gains_curve, gains_table, recency_rule_score

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


def compare_models(
    train: pd.DataFrame, numeric: list[str], splits: list[tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """Score every candidate, and the no-model recency rule, on the same splits."""
    X, y = train[numeric + CATEGORICAL], train[TARGET]
    rule = recency_rule_score(train)
    rows = []

    rule_auc = [roc_auc_score(y.iloc[v], rule[v]) for _, v in splits]
    rule_ap = [average_precision_score(y.iloc[v], rule[v]) for _, v in splits]
    rows.append(
        {
            "model": "Recency rule (no model)",
            "roc_auc_mean": np.mean(rule_auc),
            "roc_auc_std": np.std(rule_auc),
            "pr_auc_mean": np.mean(rule_ap),
        }
    )
    for name, pipe in candidate_models(numeric).items():
        s = cross_validate(pipe, X, y, cv=splits, scoring=SCORING, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "roc_auc_mean": s["test_roc_auc"].mean(),
                "roc_auc_std": s["test_roc_auc"].std(),
                "pr_auc_mean": s["test_pr_auc"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def grouped_splits(train: pd.DataFrame, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(train, train[TARGET], train["user_pseudo_id"]))


def run(
    df: pd.DataFrame,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
    fast: bool = False,
) -> dict:
    figures = reports_dir / "figures"
    for d in (reports_dir, figures, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    everyone = clean(df)
    df = model_population(everyone)
    print(
        f"Population: {len(df):,} of {len(everyone):,} user-snapshots are regular users "
        f"(>= {WINDOWS.min_active_days} active days); churn {df[TARGET].mean():.1%} "
        f"vs {everyone[TARGET].mean():.1%} overall"
    )
    numeric = numeric_features(df)
    train, test = out_of_time_split(df)
    print(
        f"Train: {len(train):,} user-snapshots, {train['snapshot_date'].nunique()} snapshots, "
        f"churn {train[TARGET].mean():.1%}"
    )
    print(
        f"Test:  {len(test):,} user-snapshots, {test['snapshot_date'].nunique()} snapshots, "
        f"churn {test[TARGET].mean():.1%}"
    )

    # Model selection uses forward-chaining (time-ordered) validation. User-grouped
    # folds are also reported, to show how much a time-blind method overstates.
    cv_table = compare_models(train, numeric, forward_chaining_splits(train))
    cv_table.to_csv(reports_dir / "cv_results.csv", index=False)
    grouped_table = compare_models(train, numeric, grouped_splits(train, 3 if fast else 5))
    grouped_table.to_csv(reports_dir / "cv_results_grouped.csv", index=False)
    print("\nForward-chaining validation (training period, used for model selection):")
    print(cv_table.round(3).to_string(index=False))
    print("\nUser-grouped CV (time-blind, for comparison only):")
    print(grouped_table.round(3).to_string(index=False))

    fitted = cv_table[~cv_table["model"].str.startswith(("Recency", "Baseline"))]
    best_name = fitted.iloc[0]["model"]
    pipe = candidate_models(numeric)[best_name]
    X_train, X_test = train[numeric + CATEGORICAL], test[numeric + CATEGORICAL]
    y_test = test[TARGET].to_numpy()
    pipe.fit(X_train, train[TARGET])
    proba = pipe.predict_proba(X_test)[:, 1]
    rule = recency_rule_score(test)

    gains_model = gains_table(y_test, proba)
    gains_rule = gains_table(y_test, rule)
    gains = gains_model.merge(
        gains_rule[["contacted_share", "share_of_churners_reached", "lift"]],
        on="contacted_share",
        suffixes=("", "_recency_rule"),
    )
    gains.to_csv(reports_dir / "gains_test.csv", index=False)

    per_snapshot = (
        test.assign(p=proba, r=rule)
        .groupby("snapshot_date")
        .apply(
            lambda g: pd.Series(
                {
                    "users": len(g),
                    "churn_rate": g[TARGET].mean(),
                    "roc_auc_model": roc_auc_score(g[TARGET], g["p"]),
                    "roc_auc_recency_rule": roc_auc_score(g[TARGET], g["r"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    per_snapshot.to_csv(reports_dir / "test_by_snapshot.csv", index=False)

    top20 = gains.loc[np.isclose(gains["contacted_share"], 0.2)].iloc[0]
    metrics = {
        "selected_model": best_name,
        "population": f">= {WINDOWS.min_active_days} active days in {WINDOWS.observation_days}",
        "all_user_snapshots": len(everyone),
        "all_users_churn_rate": float(everyone[TARGET].mean()),
        "population_user_snapshots": len(df),
        "population_churn_rate": float(df[TARGET].mean()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_users": int(train["user_pseudo_id"].nunique()),
        "test_users": int(test["user_pseudo_id"].nunique()),
        "train_snapshots": sorted(str(d) for d in train["snapshot_date"].unique()),
        "test_snapshots": sorted(str(d) for d in test["snapshot_date"].unique()),
        "test_churn_rate": float(y_test.mean()),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "mean_predicted_churn": float(proba.mean()),
        "forward_chaining_val_roc_auc": float(
            cv_table.loc[cv_table["model"] == best_name, "roc_auc_mean"].iloc[0]
        ),
        "grouped_cv_roc_auc": float(
            grouped_table.loc[grouped_table["model"] == best_name, "roc_auc_mean"].iloc[0]
        ),
        "recency_rule_roc_auc": roc_auc_score(y_test, rule),
        "recency_rule_pr_auc": average_precision_score(y_test, rule),
        "top20_share_of_churners_model": float(top20["share_of_churners_reached"]),
        "top20_share_of_churners_rule": float(top20["share_of_churners_reached_recency_rule"]),
        "top20_lift_model": float(top20["lift"]),
        "top20_lift_rule": float(top20["lift_recency_rule"]),
        "top20_churn_rate_model": float(top20["churn_rate_in_group"]),
    }

    sample = test.sample(min(len(test), 20_000), random_state=RANDOM_STATE)
    imp = permutation_importance(
        pipe,
        sample[numeric + CATEGORICAL],
        sample[TARGET],
        scoring="roc_auc",
        n_repeats=3 if fast else 10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": numeric + CATEGORICAL,
                "importance": imp.importances_mean,
                "std": imp.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)

    joblib.dump(
        {"pipeline": pipe, "numeric": numeric, "categorical": CATEGORICAL, "model_name": best_name},
        models_dir / "app_churn_model.joblib",
    )
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    plots.plot_gains(
        {
            best_name: gains_curve(y_test, proba),
            "Recency rule (no model)": gains_curve(y_test, rule),
        },
        figures / "gains_curve.png",
    )
    plots.plot_importance(importance, figures / "feature_importance.png")
    plots.plot_churn_by_recency(df, figures / "churn_by_recency.png")
    plots.plot_churn_by_engagement(
        everyone,
        figures / "churn_by_engagement.png",
        f"active_days_{WINDOWS.observation_days}d",
        WINDOWS.min_active_days,
    )

    print(f"\nSelected: {best_name}")
    print(
        f"Test ROC-AUC {metrics['roc_auc']:.3f} "
        f"(recency rule {metrics['recency_rule_roc_auc']:.3f})"
        f" | PR-AUC {metrics['pr_auc']:.3f} (rule {metrics['recency_rule_pr_auc']:.3f})"
    )
    print(
        f"Top 20% riskiest reach {metrics['top20_share_of_churners_model']:.0%} of churners "
        f"(rule {metrics['top20_share_of_churners_rule']:.0%})"
    )
    print("\nTest by snapshot:")
    print(per_snapshot.round(3).to_string(index=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the app churn model")
    parser.add_argument("--input", type=Path, default=FEATURES_PATH)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"{args.input} not found. Run `python -m appchurn.extract` first.")
    import db_dtypes  # noqa: F401  registers BigQuery date types for older extracts

    run(pd.read_parquet(args.input), fast=args.fast)


if __name__ == "__main__":
    main()
