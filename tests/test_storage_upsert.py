# tests/test_storage_upsert.py
"""Tests for storage upsert functionality."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.car_valuation.storage.upsert import (
    UpsertResult,
    _chunk,
    _get_table_cfg,
    upsert_from_config,
    upsert_records,
)


class TestChunk:
    """Tests for _chunk helper."""

    def test_exact_batches(self):
        records = [{"id": i} for i in range(10)]
        batches = list(_chunk(records, batch_size=5))
        assert len(batches) == 2
        assert all(len(b) == 5 for b in batches)

    def test_partial_last_batch(self):
        records = [{"id": i} for i in range(7)]
        batches = list(_chunk(records, batch_size=3))
        assert len(batches) == 3
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 1

    def test_empty_records(self):
        batches = list(_chunk([], batch_size=10))
        assert batches == []

    def test_single_record(self):
        batches = list(_chunk([{"id": 1}], batch_size=10))
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_batch_size_larger_than_records(self):
        records = [{"id": i} for i in range(3)]
        batches = list(_chunk(records, batch_size=100))
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            list(_chunk([{"id": 1}], batch_size=0))

        with pytest.raises(ValueError, match="batch_size must be > 0"):
            list(_chunk([{"id": 1}], batch_size=-1))


class TestGetTableCfg:
    """Tests for _get_table_cfg helper."""

    def test_returns_table_config(self):
        db_cfg = {
            "tables": {
                "scraped_items": {
                    "table": "Cars",
                    "upsert": {"conflict_columns": ["listing_id"]},
                }
            }
        }
        result = _get_table_cfg(db_cfg, "scraped_items")
        assert result["table"] == "Cars"

    def test_missing_table_key_raises(self):
        db_cfg = {"tables": {"other": {}}}
        with pytest.raises(KeyError, match="Missing tables.scraped_items"):
            _get_table_cfg(db_cfg, "scraped_items")

    def test_missing_tables_section(self):
        db_cfg = {}
        with pytest.raises(KeyError):
            _get_table_cfg(db_cfg, "scraped_items")


class TestUpsertRecords:
    """Tests for upsert_records function."""

    def test_empty_records_returns_zero_counts(self):
        mock_client = MagicMock()
        result = upsert_records(
            mock_client,
            "Cars",
            [],
            conflict_columns=["listing_id"],
        )
        assert result.attempted == 0
        assert result.succeeded == 0
        assert result.failed_batches == 0

    def test_missing_conflict_columns_raises(self):
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="conflict_columns must be provided"):
            upsert_records(mock_client, "Cars", [{"id": 1}], conflict_columns=[])

    def test_successful_upsert(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [{"listing_id": 1}, {"listing_id": 2}]
        mock_client.table.return_value.upsert.return_value.execute.return_value = mock_response

        records = [{"listing_id": 1, "price": 100}, {"listing_id": 2, "price": 200}]
        result = upsert_records(
            mock_client,
            "Cars",
            records,
            conflict_columns=["listing_id"],
        )

        assert result.table == "Cars"
        assert result.attempted == 2
        assert result.succeeded == 2
        assert result.failed_batches == 0

    def test_batching_with_batch_size(self):
        # Track which batches were processed
        batches_processed = []

        def track_upsert(batch, on_conflict):
            batches_processed.append(batch)
            mock_query = MagicMock()
            mock_resp = MagicMock()
            mock_resp.data = batch
            mock_query.execute.return_value = mock_resp
            return mock_query

        # upsert_records does getattr(sb, "client", sb) which returns sb.client
        # So we need to set up the chain on mock_wrapper.client
        mock_wrapper = MagicMock()
        mock_wrapper.client.table.return_value.upsert.side_effect = track_upsert

        records = [{"listing_id": i} for i in range(5)]
        result = upsert_records(
            mock_wrapper,
            "Cars",
            records,
            conflict_columns=["listing_id"],
            batch_size=2,
        )

        # 5 records with batch_size=2 means 3 batches (2, 2, 1)
        assert len(batches_processed) == 3
        assert len(batches_processed[0]) == 2
        assert len(batches_processed[1]) == 2
        assert len(batches_processed[2]) == 1
        assert result.succeeded == 5

    def test_continue_on_error(self):
        call_count = [0]

        def track_upsert(batch, on_conflict):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Network error")
            mock_query = MagicMock()
            mock_resp = MagicMock()
            mock_resp.data = batch
            mock_query.execute.return_value = mock_resp
            return mock_query

        mock_wrapper = MagicMock()
        mock_wrapper.client.table.return_value.upsert.side_effect = track_upsert

        records = [{"listing_id": i} for i in range(4)]
        result = upsert_records(
            mock_wrapper,
            "Cars",
            records,
            conflict_columns=["listing_id"],
            batch_size=2,
            continue_on_error=True,
        )

        assert result.failed_batches == 1
        assert result.succeeded == 2  # Second batch of 2 succeeds
        assert len(result.errors) == 1

    def test_stop_on_error_when_continue_false(self):
        call_count = [0]

        def track_upsert(batch, on_conflict):
            call_count[0] += 1
            raise Exception("Error")

        mock_wrapper = MagicMock()
        mock_wrapper.client.table.return_value.upsert.side_effect = track_upsert

        records = [{"listing_id": i} for i in range(4)]
        result = upsert_records(
            mock_wrapper,
            "Cars",
            records,
            conflict_columns=["listing_id"],
            batch_size=2,
            continue_on_error=False,
        )

        assert result.failed_batches == 1
        # Only 1 batch attempted before stopping
        assert call_count[0] == 1

    def test_accepts_supabase_client_wrapper(self):
        mock_inner_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [{"id": 1}]
        mock_inner_client.table.return_value.upsert.return_value.execute.return_value = mock_response

        wrapper = MagicMock()
        wrapper.client = mock_inner_client

        records = [{"listing_id": 1}]
        result = upsert_records(
            wrapper,
            "Cars",
            records,
            conflict_columns=["listing_id"],
        )

        assert result.succeeded == 1
        mock_inner_client.table.assert_called_with("Cars")


class TestUpsertFromConfig:
    """Tests for upsert_from_config function."""

    def test_uses_config_values(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [{"id": 1}]
        mock_client.table.return_value.upsert.return_value.execute.return_value = mock_response

        db_cfg = {
            "tables": {
                "scraped_items": {
                    "table": "Cars",
                    "upsert": {
                        "enabled": True,
                        "conflict_columns": ["listing_id"],
                        "batch_size": 100,
                    },
                }
            }
        }

        records = [{"listing_id": 1}]
        result = upsert_from_config(mock_client, db_cfg, records)

        assert result.table == "Cars"
        assert result.attempted == 1

    def test_disabled_upsert_returns_zero_succeeded(self):
        mock_client = MagicMock()
        db_cfg = {
            "tables": {
                "scraped_items": {
                    "table": "Cars",
                    "upsert": {"enabled": False},
                }
            }
        }

        records = [{"listing_id": 1}]
        result = upsert_from_config(mock_client, db_cfg, records)

        assert result.succeeded == 0
        assert result.attempted == 1
        mock_client.table.assert_not_called()

    def test_missing_table_name_raises(self):
        mock_client = MagicMock()
        db_cfg = {
            "tables": {
                "scraped_items": {
                    "upsert": {"conflict_columns": ["listing_id"]},
                }
            }
        }

        with pytest.raises(KeyError, match="Missing tables.scraped_items.table"):
            upsert_from_config(mock_client, db_cfg, [{"id": 1}])

    def test_custom_table_key(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = []
        mock_client.table.return_value.upsert.return_value.execute.return_value = mock_response

        db_cfg = {
            "tables": {
                "embeddings": {
                    "table": "CarEmbeddings",
                    "upsert": {
                        "conflict_columns": ["listing_id"],
                    },
                }
            }
        }

        records = [{"listing_id": 1, "embedding": [0.1, 0.2]}]
        result = upsert_from_config(mock_client, db_cfg, records, table_key="embeddings")

        assert result.table == "CarEmbeddings"
