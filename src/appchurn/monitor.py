"""Monitor the churn model the way it would be watched in production.

    python -m appchurn.monitor                  # writes reports/monitoring.json + .csv
    python -m appchurn.monitor --fail-on-alert  # exit code 1 if the latest week alerts

The monitor replays production week by week. Each week is scored by a model fitted
only on weeks whose 14-day outcome was already known on that date (the same rule as
``appchurn.score``). It then runs four groups of checks:

* Performance, once outcomes arrive: ranking quality (ROC-AUC) against the no-model
  recency rule, and against the first monitored week.
* Calibration: average predicted churn against actual churn.
* Drift: how far this week's inputs have moved from the data the model was trained
  on, and its scores from last week's, measured with the population stability index.
* Data health: population size, and the volume of every tracked event per user. A
  sudden drop to zero usually means the app's tracking changed, not the customers.

In real use the most recent 14 days of scores are still waiting for their outcome;
those weeks show performance and calibration as "pending" while drift and data
health are checked immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from appchurn.config import DATASET, FEATURES_PATH, REPORTS_DIR, WINDOWS, WindowConfig
from appchurn.features import clean, model_population
from appchurn.model import recency_rule_score
from appchurn.score import score

OK, WARN, ALERT, PENDING = "ok", "warning", "alert", "pending"
_RANK = {OK: 0, PENDING: 0, WARN: 1, ALERT: 2}


@dataclass(frozen=True)
class MonitorConfig:
    """Alert thresholds. Values follow common industry practice for churn models."""

    min_train_weeks: int = 3  # weeks of known outcomes before the first scored week
    auc_drop_warn: float = 0.05  # ROC-AUC fall versus the first monitored week
    calibration_warn: float = 0.05  # |average predicted - actual churn rate|
    calibration_alert: float = 0.10
    psi_warn: float = 0.10  # population stability index
    psi_alert: float = 0.25
    population_change_warn: float = 0.30  # week-on-week change in users scored
    event_ratio_low: float = 0.5  # event volume per user versus training data
    event_ratio_high: float = 2.0
    event_min_baseline: float = 0.05  # ignore events too rare to judge


MONITOR = MonitorConfig()
OBS = WINDOWS.observation_days
DRIFT_FEATURES = [
    "days_since_last_active",
    f"active_days_{OBS}d",
    "engagement_minutes_7d",
    f"sessions_{OBS}d",
    "events_per_session",
    "tenure_days",
]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index: 0 = identical, >0.1 some shift, >0.25 large shift."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, edges)[0] / len(ref)
    cur_share = np.histogram(cur, edges)[0] / len(cur)
    ref_share = np.clip(ref_share, 1e-4, None)
    cur_share = np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _level(value: float, warn: float, alert: float | None = None) -> str:
    if value is None or np.isnan(value):
        return PENDING
    if alert is not None and value > alert:
        return ALERT
    return WARN if value > warn else OK


def _worst(levels: list[str]) -> str:
    return max(levels, key=lambda lv: _RANK[lv]) if levels else OK


def monitored_weeks(
    data: pd.DataFrame, windows: WindowConfig = WINDOWS, cfg: MonitorConfig = MONITOR
) -> list[date]:
    gap = timedelta(days=windows.churn_days)
    snaps = sorted(data["snapshot_date"].unique())
    return [s for s in snaps if sum(t + gap <= s for t in snaps) >= cfg.min_train_weeks]


def run(
    snapshots: pd.DataFrame,
    windows: WindowConfig = WINDOWS,
    cfg: MonitorConfig = MONITOR,
    journey_events: tuple[str, ...] = DATASET.journey_events,
) -> dict:
    data = model_population(clean(snapshots), windows)
    weeks = monitored_weeks(data, windows, cfg)
    if not weeks:
        raise ValueError("Not enough weeks of history to monitor")

    gap = timedelta(days=windows.churn_days)
    event_cols = [
        f"n_{e.lower()}_{OBS}d" for e in journey_events if f"n_{e.lower()}_{OBS}d" in data
    ]

    rows, drift_rows, event_rows = [], [], []
    prev_scores = None
    first_auc = None
    prev_users = None

    for week in weeks:
        result = score(snapshots, week, windows)
        scored = result.scored
        p = scored["churn_score"].to_numpy()
        # Reference = exactly the rows this week's model learned from
        reference = data[data["snapshot_date"] + gap <= week]
        y = result.outcomes.to_numpy() if result.outcomes is not None else None
        known = y is not None and not np.isnan(y.astype(float)).any()

        row = {
            "week": week.isoformat(),
            "users": len(scored),
            "mean_predicted": float(p.mean()),
            "train_rows": result.train_rows,
        }
        checks = []

        # Performance and calibration (need outcomes)
        if known:
            auc = roc_auc_score(y, p)
            rule_auc = roc_auc_score(y, recency_rule_score(scored))
            first_auc = auc if first_auc is None else first_auc
            high = (scored["risk_band"] == "High").to_numpy()
            gap_cal = float(p.mean() - y.mean())
            row.update(
                roc_auc=float(auc),
                rule_roc_auc=float(rule_auc),
                actual_churn=float(y.mean()),
                high_band_churn=float(y[high].mean()),
                calibration_gap=gap_cal,
            )
            beats_rule = ALERT if auc <= rule_auc else OK
            checks.append(
                (
                    "Beats the recency rule",
                    beats_rule,
                    f"ROC-AUC {auc:.3f} vs rule {rule_auc:.3f}",
                )
            )
            drop = first_auc - auc
            checks.append(
                (
                    "Ranking quality stable",
                    WARN if drop > cfg.auc_drop_warn else OK,
                    f"ROC-AUC {auc:.3f}, first monitored week {first_auc:.3f}",
                )
            )
            checks.append(
                (
                    "Probabilities match reality",
                    _level(abs(gap_cal), cfg.calibration_warn, cfg.calibration_alert),
                    f"predicted {p.mean():.1%} vs actual {y.mean():.1%}",
                )
            )
        else:
            for name in ("Beats the recency rule", "Ranking quality stable",
                         "Probabilities match reality"):  # fmt: skip
                checks.append((name, PENDING, f"outcome known after {week + gap:%d %b}"))

        # Drift of inputs and scores
        current = data[data["snapshot_date"] == week]
        worst_feature, worst_psi = None, 0.0
        for feat in DRIFT_FEATURES:
            value = psi(reference[feat], current[feat])
            drift_rows.append({"week": week.isoformat(), "feature": feat, "psi": value})
            if not np.isnan(value) and value > worst_psi:
                worst_feature, worst_psi = feat, value
        score_psi = psi(prev_scores, p) if prev_scores is not None else 0.0
        prev_scores = p
        drift_rows.append({"week": week.isoformat(), "feature": "churn_score", "psi": score_psi})
        row.update(max_feature_psi=worst_psi, max_psi_feature=worst_feature, score_psi=score_psi)
        checks.append(
            (
                "Inputs match training data",
                _level(worst_psi, cfg.psi_warn, cfg.psi_alert),
                f"largest shift: {worst_feature} (PSI {worst_psi:.2f})",
            )
        )
        checks.append(
            (
                "Scores stable",
                # Weekly retraining reshapes scores by itself, so a score shift is a
                # prompt to review the new model, never an alert on its own.
                _level(score_psi, cfg.psi_warn),
                f"PSI {score_psi:.2f} vs previous week (model retrained weekly)",
            )
        )

        # Data health: population and event volumes
        change = 0.0 if prev_users is None else (len(scored) - prev_users) / prev_users
        prev_users = len(scored)
        row["population_change"] = change
        checks.append(
            (
                "Population size normal",
                WARN if abs(change) > cfg.population_change_warn else OK,
                f"{len(scored):,} users ({change:+.0%} vs previous week)",
            )
        )
        flagged = []
        for col in event_cols:
            base = float(reference[col].mean())
            now = float(current[col].mean())
            ratio = now / base if base > 0 else float("nan")
            status = OK
            if base >= cfg.event_min_baseline:
                if now == 0:
                    status = ALERT
                elif ratio < cfg.event_ratio_low or ratio > cfg.event_ratio_high:
                    status = WARN
            event_rows.append(
                {"week": week.isoformat(), "event": col[2 : -len(f"_{OBS}d")],
                 "per_user": now, "baseline_per_user": base, "ratio": ratio, "status": status}
            )  # fmt: skip
            if status != OK:
                flagged.append((col[2 : -len(f"_{OBS}d")], status, ratio))
        events_level = _worst([s for _, s, _ in flagged])
        detail = (
            "; ".join(f"{e} {'stopped' if s == ALERT else f'×{r:.1f}'}" for e, s, r in flagged[:3])
            if flagged
            else f"{len(event_cols)} events within normal range"
        )
        checks.append(("Event tracking healthy", events_level, detail))

        row["status"] = _worst([lv for _, lv, _ in checks])
        row["checks"] = [{"check": c, "status": s, "detail": d} for c, s, d in checks]
        rows.append(row)

    latest = rows[-1]
    return {
        "config": asdict(cfg),
        "weeks": rows,
        "latest": {"week": latest["week"], "status": latest["status"], "checks": latest["checks"]},
        "drift": drift_rows,
        "events": event_rows,
    }


def write(result: dict, reports_dir=REPORTS_DIR) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "monitoring.json").write_text(json.dumps(result, indent=2, default=float))
    weekly = pd.DataFrame([{k: v for k, v in r.items() if k != "checks"} for r in result["weeks"]])
    weekly.to_csv(reports_dir / "monitoring_weekly.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor the churn model")
    parser.add_argument("--fail-on-alert", action="store_true",
                        help="exit with code 1 if the latest week has an alert")  # fmt: skip
    args = parser.parse_args()
    import db_dtypes  # noqa: F401  older extracts store BigQuery date types

    result = run(pd.read_parquet(FEATURES_PATH))
    write(result)
    for r in result["weeks"]:
        auc = f"{r['roc_auc']:.3f} (rule {r['rule_roc_auc']:.3f})" if "roc_auc" in r else "pending"
        print(f"{r['week']}  {r['status'].upper():8}  users {r['users']:>5,}  ROC-AUC {auc}")
    print(f"\nLatest week {result['latest']['week']}: {result['latest']['status'].upper()}")
    for c in result["latest"]["checks"]:
        print(f"  [{c['status']:>7}] {c['check']}: {c['detail']}")
    if args.fail_on_alert and result["latest"]["status"] == ALERT:
        sys.exit(1)


if __name__ == "__main__":
    main()
