"""The BigQuery calls themselves, checked with a stand-in client (no network)."""

import pandas as pd
import pytest

from appchurn import extract


class _Result:
    def to_dataframe(self, create_bqstorage_client=True):
        assert create_bqstorage_client is False
        return pd.DataFrame({"x": [1, 2]})


class _Job:
    def __init__(self, config):
        self.config = config
        self.total_bytes_processed = 3 * extract.GB
        self.total_bytes_billed = 3 * extract.GB

    def result(self):
        return _Result()


class FakeClient:
    def __init__(self):
        self.configs = []

    def query(self, sql, job_config=None):
        self.configs.append(job_config)
        return _Job(job_config)


def test_dry_run_is_used_for_the_estimate():
    client = FakeClient()
    assert extract.bytes_scanned(client, "SELECT 1") == 3 * extract.GB
    assert client.configs[0].dry_run is True


def test_query_runs_with_a_hard_byte_cap():
    pytest.importorskip("google.cloud.bigquery")
    client = FakeClient()
    df = extract.run_query(client, "SELECT 1", max_gb=5)
    assert len(df) == 2
    real_run = client.configs[-1]
    assert real_run.maximum_bytes_billed == 5 * extract.GB


def test_missing_project_gives_a_clear_message():
    with pytest.raises(SystemExit, match="GCP_PROJECT"):
        extract.get_client(None)


def test_bigquery_date_types_are_converted_for_parquet(tmp_path):
    import db_dtypes  # noqa: F401

    df = pd.DataFrame({"snapshot_date": pd.Series(["2018-07-10", "2018-07-17"], dtype="dbdate")})
    plain = extract.to_plain_dtypes(df)
    assert str(plain["snapshot_date"].dtype).startswith("datetime64")
    path = tmp_path / "x.parquet"
    plain.to_parquet(path)
    # Readable without db-dtypes registered: the schema must hold no extension type
    import pyarrow.parquet as pq

    assert "dbdate" not in str(pq.read_schema(path).metadata)
