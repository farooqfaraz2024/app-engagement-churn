"""Preprocessing, candidate models and the evaluation used for campaign targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from appchurn.config import RANDOM_STATE, WINDOWS
from appchurn.features import CATEGORICAL


def build_preprocessor(numeric: list[str]) -> ColumnTransformer:
    # Counts are heavily skewed (a few users have thousands of events), so they are
    # log-transformed before scaling. All engineered features are non-negative.
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )


def candidate_models(numeric: list[str]) -> dict[str, Pipeline]:
    estimators = {
        "Baseline (churn rate)": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=3000, C=0.5),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline([("preprocess", build_preprocessor(numeric)), ("model", est)])
        for name, est in estimators.items()
    }


def recency_rule_score(df: pd.DataFrame) -> np.ndarray:
    """What a CRM team does without a model: rank users by days since last activity.

    Ties are broken by how few days the user was active. This is the benchmark the
    model has to beat to be worth deploying.
    """
    active_days = df[f"active_days_{WINDOWS.observation_days}d"].to_numpy()
    return df["days_since_last_active"].to_numpy() - 0.01 * active_days


def gains_table(y_true: np.ndarray, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Share of all churners reached when contacting the top k% riskiest users."""
    order = np.argsort(-np.asarray(score), kind="stable")
    y_sorted = np.asarray(y_true)[order]
    n, total = len(y_sorted), y_sorted.sum()
    rows = []
    for i in range(1, bins + 1):
        k = int(round(n * i / bins))
        caught = y_sorted[:k].sum()
        rows.append(
            {
                "contacted_share": i / bins,
                "users_contacted": k,
                "churners_reached": int(caught),
                "share_of_churners_reached": caught / total if total else np.nan,
                "churn_rate_in_group": caught / k if k else np.nan,
                "lift": (caught / k) / (total / n) if k and total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def gains_curve(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(score), kind="stable")
    y_sorted = np.asarray(y_true)[order]
    x = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    y = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    return np.concatenate([[0], x]), np.concatenate([[0], y])
