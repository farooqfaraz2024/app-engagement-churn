"""Build the interactive churn report as one self-contained HTML page.

    python -m appchurn.train      # evaluation reports and the evaluation model
    python -m appchurn.report     # scores the latest snapshot and writes docs/index.html

The page is written to ``docs/index.html`` so GitHub Pages can serve it directly.
Python prepares every number; the page only draws them (plotly.js from a CDN).

Dates on the page can be shifted forward by a whole number of weeks
(``DISPLAY_SHIFT_WEEKS``) so a demo built on historical data reads as current.
Weekdays are preserved, a banner says so, and every figure is computed from the
real data.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from appchurn.config import (
    DATASET,
    EVENT_PROFILE_PATH,
    FEATURES_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    WINDOWS,
)
from appchurn.features import (
    TARGET,
    clean,
    model_population,
    out_of_time_split,
)
from appchurn.model import recency_rule_score
from appchurn.monitor import run as run_monitor
from appchurn.monitor import write as write_monitor
from appchurn.score import BANDS, ScoreResult, export, score

DISPLAY_SHIFT_WEEKS = int(os.environ.get("DISPLAY_SHIFT_WEEKS", "365"))
DOCS_DIR = PROJECT_ROOT / "docs"
OBS = WINDOWS.observation_days

FEATURE_LABELS = {
    "days_since_last_active": "Days since last active",
    f"active_days_{OBS}d": f"Active days (last {OBS})",
    "active_days_7d": "Active days (last 7)",
    "engagement_minutes_7d": "Minutes in app (last 7 days)",
    f"engagement_minutes_{OBS}d": f"Minutes in app (last {OBS} days)",
    "tenure_days": "Tenure (days since install)",
    "events_7d_vs_weekly_avg": "This week vs usual week",
    f"events_{OBS}d": f"Events (last {OBS} days)",
    "events_7d": "Events (last 7 days)",
    f"sessions_{OBS}d": f"Sessions (last {OBS} days)",
    "minutes_per_active_day": "Minutes per active day",
    "events_per_session": "Events per session",
    "active_day_share_7d": "Share of last 7 days active",
    "active_day_share_obs": f"Share of last {OBS} days active",
    "device_category": "Device type",
    "platform": "Platform",
    "country": "Country",
    "is_payer": "Has paid",
}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("n_") and name.endswith(f"_{OBS}d"):
        event = name[2 : -len(f"_{OBS}d")].replace("_", " ")
        return f"Count: {event}"
    return name.replace("_", " ").capitalize()


def shift(d: date, weeks: int = DISPLAY_SHIFT_WEEKS) -> date:
    return d + timedelta(weeks=weeks)


def fmt(d: date, weeks: int = DISPLAY_SHIFT_WEEKS) -> str:
    """Display date: shifted, with weekday, e.g. 'Tue 16 Sep 2025'."""
    return shift(d, weeks).strftime("%a %d %b %Y").replace(" 0", " ")


def _tenure_band(days: float) -> str:
    if pd.isna(days):
        return "Unknown"
    if days < 30:
        return "Under 30 days"
    if days < 90:
        return "30–89 days"
    if days < 365:
        return "90–364 days"
    return "1 year +"


def _segments(scored: pd.DataFrame, outcomes: pd.Series | None) -> dict:
    df = scored.copy()
    df["tenure_band"] = df["tenure_days"].map(_tenure_band)
    if outcomes is not None:
        df["_actual"] = outcomes.to_numpy()
    out = {}
    dims = {
        "Platform": "platform",
        "Device type": "device_category",
        "Country": "country",
        "Tenure": "tenure_band",
    }
    tenure_order = ["Under 30 days", "30–89 days", "90–364 days", "1 year +", "Unknown"]
    for label, col in dims.items():
        g = df.groupby(col)
        rows = pd.DataFrame(
            {
                "users": g.size(),
                "high_share": g["risk_band"].apply(lambda s: (s == "High").mean()),
                "mean_score": g["churn_score"].mean(),
            }
        )
        if outcomes is not None:
            rows["actual_churn"] = g["_actual"].mean()
        rows = rows[rows["users"] >= 15]
        if col == "tenure_band":
            rows = rows.reindex([t for t in tenure_order if t in rows.index])
        else:
            rows = rows.sort_values("users", ascending=False).head(8)
        out[label] = [
            {
                "segment": str(idx),
                "users": int(r.users),
                "high_share": round(float(r.high_share), 4),
                "mean_score": round(float(r.mean_score), 4),
                "actual_churn": None if outcomes is None else round(float(r.actual_churn), 4),
            }
            for idx, r in rows.iterrows()
        ]
    return out


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append(
            {
                "predicted": round(float(p[chunk].mean()), 4),
                "actual": round(float(y[chunk].mean()), 4),
                "users": int(len(chunk)),
            }
        )
    return rows


def build_payload(
    snapshots: pd.DataFrame,
    result: ScoreResult,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
    weeks: int = DISPLAY_SHIFT_WEEKS,
    monitoring: dict | None = None,
) -> dict:
    metrics = json.loads((reports_dir / "metrics.json").read_text())
    per_week = pd.read_csv(reports_dir / "test_by_snapshot.csv")
    importance = pd.read_csv(reports_dir / "feature_importance.csv").head(12)

    everyone = clean(snapshots)
    population = model_population(everyone)
    scored = result.scored
    outcomes = result.outcomes

    # Section 1: summary
    band_counts = scored["risk_band"].value_counts()
    summary = {
        "users": len(scored),
        "bands": {b: int(band_counts.get(b, 0)) for b, _ in BANDS},
        "expected_churners": round(float(scored["churn_score"].sum())),
        "score_date": fmt(result.score_date, weeks),
        "horizon_end": fmt(result.score_date + timedelta(days=WINDOWS.churn_days - 1), weeks),
        "model": result.model_name,
        "train_from": fmt(result.train_snapshots[0], weeks),
        "train_to": fmt(result.train_snapshots[-1], weeks),
        "train_rows": result.train_rows,
    }

    # Section 2 and 3: customer list and reasons
    rows = []
    for r in scored.itertuples(index=False):
        rows.append(
            [
                int(r.rank),
                r.user_pseudo_id,
                r.risk_band,
                round(float(r.churn_score), 4),
                None if pd.isna(r.days_since_last_active) else int(r.days_since_last_active),
                int(getattr(r, f"active_days_{OBS}d")),
                None
                if pd.isna(r.events_7d_vs_weekly_avg)
                else round(float(r.events_7d_vs_weekly_avg), 2),
                None if pd.isna(r.tenure_days) else int(r.tenure_days),
                r.platform,
                r.country,
                r.reasons,
            ]
        )
    reason_counts = {}
    for band in ("High", "Medium"):
        exploded = (
            scored.loc[scored["risk_band"] == band, "reasons"]
            .str.split("; ")
            .explode()
            .str.replace(r"\d+", "N", regex=True)
        )
        reason_counts[band] = exploded.value_counts().to_dict()

    # Section 4: campaign planner (backtest on the score date when outcomes are known)
    planner = None
    if outcomes is not None:
        y = outcomes.to_numpy()
        rule = recency_rule_score(scored)
        rule_order = np.argsort(-rule, kind="stable")
        planner = {
            "model_cum": np.cumsum(y).astype(int).tolist(),  # scored is already ranked
            "rule_cum": np.cumsum(y[rule_order]).astype(int).tolist(),
            "total_churners": int(y.sum()),
            "base_rate": round(float(y.mean()), 4),
        }

    # Section 5: performance
    edge = metrics["model_vs_recency_rule_auc"]
    performance = {
        "roc_auc": metrics["roc_auc"],
        "rule_roc_auc": metrics["recency_rule_roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "rule_pr_auc": metrics["recency_rule_pr_auc"],
        "edge": edge,
        "test_rows": metrics["test_rows"],
        "test_churn_rate": metrics["test_churn_rate"],
        "weeks": [
            {
                "week": fmt(date.fromisoformat(str(w)[:10]), weeks),
                "model": round(float(m), 4),
                "rule": round(float(rr), 4),
                "users": int(u),
                "churn_rate": round(float(c), 4),
            }
            for w, m, rr, u, c in per_week[
                ["snapshot_date", "roc_auc_model", "roc_auc_recency_rule", "users", "churn_rate"]
            ].itertuples(index=False)
        ],
        "validation": [
            {"method": "User-grouped CV (time-blind)", "roc_auc": metrics["grouped_cv_roc_auc"]},
            {
                "method": "Forward-chaining validation",
                "roc_auc": metrics["forward_chaining_val_roc_auc"],
            },
            {"method": "Out-of-time test (actual)", "roc_auc": metrics["roc_auc"]},
        ],
    }

    # Section 7: drivers
    silence_bins = [0, 1, 3, 7, 14, 21, 28]
    silence_labels = ["1", "2–3", "4–7", "8–14", "15–21", "22–28"]
    bucket = pd.cut(population["days_since_last_active"], silence_bins, labels=silence_labels)
    silence = population.groupby(bucket, observed=False)[TARGET].agg(["mean", "size"])
    drivers = {
        "importance": [
            {"feature": feature_label(f), "importance": round(float(v), 4)}
            for f, v in importance[["feature", "importance"]].itertuples(index=False)
        ],
        "silence": [
            {"days": str(k), "churn_rate": round(float(v["mean"]), 4), "users": int(v["size"])}
            for k, v in silence.iterrows()
        ],
    }

    # Section 8: calibration (evaluation model on the test period) and band backtest
    calibration = {"deciles": [], "bands": []}
    model_path = models_dir / "app_churn_model.joblib"
    if model_path.exists():
        bundle = joblib.load(model_path)
        _, test = out_of_time_split(population)
        cols = bundle["numeric"] + bundle["categorical"]
        p = bundle["pipeline"].predict_proba(test[cols])[:, 1]
        calibration["deciles"] = _calibration(test[TARGET].to_numpy(), p)
        calibration["test_mean_predicted"] = round(float(p.mean()), 4)
        calibration["test_actual"] = round(float(test[TARGET].mean()), 4)
    if outcomes is not None:
        for band, _ in BANDS:
            mask = (scored["risk_band"] == band).to_numpy()
            calibration["bands"].append(
                {
                    "band": band,
                    "users": int(mask.sum()),
                    "expected": round(float(scored.loc[mask, "churn_score"].mean()), 4),
                    "actual": round(float(outcomes.to_numpy()[mask].mean()), 4),
                }
            )
        calibration["expected_total"] = summary["expected_churners"]
        calibration["actual_total"] = int(outcomes.sum())

    # Section 9: data quality
    profile = pd.read_csv(EVENT_PROFILE_PATH) if EVENT_PROFILE_PATH.exists() else None
    extract_log = reports_dir / "extract_log.json"
    log = json.loads(extract_log.read_text()) if extract_log.exists() else {}
    raw_tenure = pd.to_numeric(snapshots["tenure_days"], errors="coerce")
    placeholder = int((raw_tenure.notna() & everyone["tenure_days"].isna()).sum())
    snaps = sorted(everyone["snapshot_date"].unique())
    quality = [
        ["Source", "Public GA4 / Firebase demo export (Flood-It! puzzle game)"],
        [
            "Event dates covered",
            f"{fmt(DATASET.data_start, weeks)} to {fmt(DATASET.data_end, weeks)}",
        ],
        [
            "Original dates",
            f"{DATASET.data_start:%d %b %Y} to {DATASET.data_end:%d %b %Y} "
            f"(shifted +{weeks} weeks for display)",
        ],
    ]
    if profile is not None:
        quality.append(["Raw events", f"{int(profile['events'].sum()):,}"])
        quality.append(["Distinct event types", f"{len(profile):,}"])
    quality += [
        ["Weekly snapshots", f"{len(snaps)} ({fmt(snaps[0], weeks)} to {fmt(snaps[-1], weeks)})"],
        [
            "User-snapshots (anyone active)",
            f"{len(everyone):,} from {everyone['user_pseudo_id'].nunique():,} users",
        ],
        [
            f"Regular users (≥{WINDOWS.min_active_days} active days)",
            f"{len(population):,} user-snapshots from "
            f"{population['user_pseudo_id'].nunique():,} users",
        ],
        ["Placeholder install dates set to unknown", f"{placeholder}"],
        [
            "Duplicate user-snapshot keys",
            f"{int(everyone.duplicated(['snapshot_date', 'user_pseudo_id']).sum())}",
        ],
        [
            "Missing values in extract",
            f"{int(snapshots.drop(columns=['tenure_days']).isna().sum().sum())}",
        ],
    ]
    if log.get("gb_billed") is not None:
        quality.append(["BigQuery data scanned (last extract)", f"{log['gb_billed']:.2f} GB"])

    return {
        "meta": {
            "title": "App Churn Risk Report",
            "shift_weeks": weeks,
            "original_score_date": result.score_date.isoformat(),
            "generated": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
            "obs_days": OBS,
            "churn_days": WINDOWS.churn_days,
            "min_active_days": WINDOWS.min_active_days,
        },
        "summary": summary,
        "columns": [
            "rank",
            "user",
            "band",
            "score",
            "silent",
            "active",
            "trend",
            "tenure",
            "platform",
            "country",
            "reasons",
        ],
        "rows": rows,
        "reasons": reason_counts,
        "planner": planner,
        "performance": performance,
        "segments": _segments(scored, outcomes),
        "drivers": drivers,
        "calibration": calibration,
        "quality": quality,
        "monitoring": _monitoring_payload(monitoring, weeks) if monitoring else None,
    }


def _monitoring_payload(monitoring: dict, weeks: int) -> dict:
    """Shift dates for display and keep only what the page draws."""

    def week_label(iso: str) -> str:
        return fmt(date.fromisoformat(iso), weeks)

    drift_features = [
        "days_since_last_active",
        f"active_days_{OBS}d",
        "engagement_minutes_7d",
        f"sessions_{OBS}d",
        "events_per_session",
        "tenure_days",
    ]
    drift = pd.DataFrame(monitoring["drift"])
    drift = drift[drift["feature"].isin(drift_features)]
    events = pd.DataFrame(monitoring["events"])
    latest_week = monitoring["latest"]["week"]
    latest_events = events[events["week"] == latest_week].copy()
    latest_events = latest_events[latest_events["baseline_per_user"] >= 0.05]
    latest_events = latest_events.sort_values("ratio", key=lambda r: (r - 1).abs(), ascending=False)
    return {
        "weeks": [
            {
                "week": week_label(w["week"]),
                "status": w["status"],
                "users": w["users"],
                "roc_auc": w.get("roc_auc"),
                "rule_roc_auc": w.get("rule_roc_auc"),
                "mean_predicted": w["mean_predicted"],
                "actual_churn": w.get("actual_churn"),
                "high_band_churn": w.get("high_band_churn"),
                "max_psi": w["max_feature_psi"],
                "max_psi_feature": feature_label(w["max_psi_feature"] or ""),
                "issues": [c for c in w["checks"] if c["status"] not in ("ok",)],
            }
            for w in monitoring["weeks"]
        ],
        "latest": {
            "week": week_label(latest_week),
            "status": monitoring["latest"]["status"],
            "checks": monitoring["latest"]["checks"],
        },
        "drift": {
            "features": [feature_label(f) for f in drift_features],
            "weeks": [week_label(w) for w in sorted(drift["week"].unique())],
            "psi": [
                [
                    round(float(v), 3)
                    for v in drift[drift["feature"] == f].sort_values("week")["psi"]
                ]
                for f in drift_features
            ],
        },
        "events": [
            {
                "event": r.event.replace("_", " "),
                "per_user": round(float(r.per_user), 2),
                "baseline": round(float(r.baseline_per_user), 2),
                "ratio": round(float(r.ratio), 2),
                "status": r.status,
            }
            for r in latest_events.itertuples(index=False)
        ],
        "thresholds": monitoring["config"],
    }


def render(payload: dict) -> str:
    template = files("appchurn").joinpath("report_template.html").read_text()
    data = json.dumps(payload, separators=(",", ":"), default=str).replace("</", "<\\/")
    return template.replace("__REPORT_DATA__", data)


def build(
    snapshots: pd.DataFrame,
    out_path: Path = DOCS_DIR / "index.html",
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
    score_date: date | None = None,
    weeks: int = DISPLAY_SHIFT_WEEKS,
) -> Path:
    result = score(snapshots, score_date)
    export(result, reports_dir / "at_risk_customers.csv")
    monitoring = run_monitor(snapshots)
    write_monitor(monitoring, reports_dir)
    payload = build_payload(snapshots, result, reports_dir, models_dir, weeks, monitoring)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(payload), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the interactive churn report")
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="score date")
    parser.add_argument("--out", type=Path, default=DOCS_DIR / "index.html")
    parser.add_argument("--shift-weeks", type=int, default=DISPLAY_SHIFT_WEEKS)
    args = parser.parse_args()
    import db_dtypes  # noqa: F401  older extracts store BigQuery date types

    path = build(
        pd.read_parquet(FEATURES_PATH), args.out, score_date=args.date, weeks=args.shift_weeks
    )
    print(f"Report written to {path}  (open it with: open {path})")


if __name__ == "__main__":
    main()
