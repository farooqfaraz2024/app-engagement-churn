"""Pull data from BigQuery with google-cloud-bigquery.

    python -m appchurn.extract --dry-run      # show how much data each query scans
    python -m appchurn.extract --profile      # which events exist (reports/event_profile.csv)
    python -m appchurn.extract                # build the user-snapshot table (data/*.parquet)

Authentication uses Application Default Credentials
(``gcloud auth application-default login``), so no key file is ever stored in the
repository. The billing project is read from the ``GCP_PROJECT`` environment
variable or ``--project``.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from appchurn.config import BIGQUERY, DATA_DIR, EVENT_PROFILE_PATH, FEATURES_PATH, REPORTS_DIR
from appchurn.sql import event_profile_sql, user_snapshot_sql

GB = 1024**3


def get_client(project: str | None):
    from google.cloud import bigquery  # imported lazily so tests don't need it

    if not project:
        sys.exit(
            "No Google Cloud project set. Run with --project YOUR_PROJECT_ID or "
            "set the GCP_PROJECT environment variable."
        )
    return bigquery.Client(project=project, location=BIGQUERY.location)


def bytes_scanned(client, sql: str) -> int:
    """Dry run: BigQuery validates the query and reports bytes it would read, free."""
    from google.cloud import bigquery

    job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
    return int(job.total_bytes_processed)


def run_query(client, sql: str, max_gb: float = BIGQUERY.max_gb_billed) -> pd.DataFrame:
    """Run with a hard cap: BigQuery rejects the job instead of scanning more than max_gb."""
    from google.cloud import bigquery

    estimate = bytes_scanned(client, sql)
    print(f"  will scan {estimate / GB:.2f} GB (cap {max_gb:.0f} GB)")
    config = bigquery.QueryJobConfig(maximum_bytes_billed=int(max_gb * GB))
    job = client.query(sql, job_config=config)
    df = job.result().to_dataframe(create_bqstorage_client=False)
    print(f"  done: {len(df):,} rows, {job.total_bytes_billed / GB:.2f} GB billed")
    return to_plain_dtypes(df)


def to_plain_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert BigQuery's ``dbdate``/``dbtime`` columns to standard pandas types.

    Otherwise the saved parquet file can only be read where the db-dtypes package
    is imported first.
    """
    df = df.copy()
    for col in df.columns:
        if str(df[col].dtype) in {"dbdate", "dbtime"}:
            df[col] = pd.to_datetime(df[col].astype(str))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract data from BigQuery")
    parser.add_argument("--project", default=BIGQUERY.project, help="billing project ID")
    parser.add_argument("--profile", action="store_true", help="only run the event profile")
    parser.add_argument("--dry-run", action="store_true", help="estimate bytes, run nothing")
    parser.add_argument("--print-sql", action="store_true", help="print the feature SQL and exit")
    args = parser.parse_args()

    if args.print_sql:
        print(user_snapshot_sql())
        return

    client = get_client(args.project)
    queries = {"event profile": event_profile_sql()}
    if not args.profile:
        queries["user snapshots"] = user_snapshot_sql()

    if args.dry_run:
        for name, sql in queries.items():
            print(f"{name}: {bytes_scanned(client, sql) / GB:.2f} GB")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Event profile")
    profile = run_query(client, queries["event profile"])
    profile.to_csv(EVENT_PROFILE_PATH, index=False)
    print(profile.head(25).to_string(index=False))

    if not args.profile:
        print("\nUser snapshots")
        snapshots = run_query(client, queries["user snapshots"])
        snapshots.to_parquet(FEATURES_PATH, index=False)
        print(f"  saved {FEATURES_PATH.relative_to(FEATURES_PATH.parents[1])}")


if __name__ == "__main__":
    main()
