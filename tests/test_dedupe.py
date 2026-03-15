# tests/test_dedupe.py
"""Tests for deduplication logic."""
from __future__ import annotations

import pytest

from src.car_valuation.collection.dedupe import (
    DedupeOptions,
    dedupe_batch,
    iter_deduped_records,
    iter_unique_ids,
    seen_set_dedupe,
)


class TestDedupeOptions:
    """Tests for DedupeOptions defaults."""

    def test_defaults(self):
        opts = DedupeOptions()
        assert opts.id_field == "listing_id"
        assert opts.prefer_last is True
        assert opts.drop_missing_ids is True


class TestDedupeBatch:
    """Tests for dedupe_batch function."""

    def test_no_duplicates(self):
        records = [
            {"listing_id": 1, "title": "Car A"},
            {"listing_id": 2, "title": "Car B"},
        ]
        result = dedupe_batch(records)
        assert len(result) == 2
        assert {r["listing_id"] for r in result} == {1, 2}

    def test_duplicates_prefer_last(self):
        records = [
            {"listing_id": 1, "title": "Old"},
            {"listing_id": 1, "title": "New"},
        ]
        result = dedupe_batch(records, options=DedupeOptions(prefer_last=True))
        assert len(result) == 1
        assert result[0]["title"] == "New"

    def test_duplicates_prefer_first(self):
        records = [
            {"listing_id": 1, "title": "First"},
            {"listing_id": 1, "title": "Second"},
        ]
        result = dedupe_batch(records, options=DedupeOptions(prefer_last=False))
        assert len(result) == 1
        assert result[0]["title"] == "First"

    def test_string_ids_parsed(self):
        records = [
            {"listing_id": "123", "title": "A"},
            {"listing_id": 123, "title": "B"},
        ]
        result = dedupe_batch(records)
        assert len(result) == 1

    def test_missing_id_dropped_by_default(self):
        records = [
            {"listing_id": 1, "title": "Good"},
            {"title": "No ID"},
            {"listing_id": None, "title": "Null ID"},
        ]
        result = dedupe_batch(records)
        assert len(result) == 1
        assert result[0]["listing_id"] == 1

    def test_missing_id_raises_if_configured(self):
        records = [
            {"listing_id": 1},
            {"title": "Missing"},
        ]
        opts = DedupeOptions(drop_missing_ids=False)
        with pytest.raises(ValueError, match="missing/invalid"):
            dedupe_batch(records, options=opts)

    def test_empty_batch(self):
        result = dedupe_batch([])
        assert result == []

    def test_custom_id_field(self):
        records = [
            {"id": 1, "name": "A"},
            {"id": 1, "name": "B"},
            {"id": 2, "name": "C"},
        ]
        opts = DedupeOptions(id_field="id", prefer_last=True)
        result = dedupe_batch(records, options=opts)
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert "B" in names
        assert "C" in names


class TestIterDedupedRecords:
    """Tests for iter_deduped_records generator."""

    def test_yields_unique_only(self):
        records = [
            {"listing_id": 1, "v": "a"},
            {"listing_id": 2, "v": "b"},
            {"listing_id": 1, "v": "c"},
        ]
        result = list(iter_deduped_records(records))
        assert len(result) == 2
        assert result[0]["listing_id"] == 1
        assert result[0]["v"] == "a"  # first seen wins in streaming

    def test_skips_missing_ids(self):
        records = [
            {"listing_id": 1},
            {"other": "no id"},
        ]
        result = list(iter_deduped_records(records))
        assert len(result) == 1

    def test_raises_on_missing_id_if_configured(self):
        records = [{"listing_id": 1}, {"no_id": True}]
        opts = DedupeOptions(drop_missing_ids=False)
        with pytest.raises(ValueError):
            list(iter_deduped_records(records, options=opts))


class TestIterUniqueIds:
    """Tests for iter_unique_ids generator."""

    def test_yields_unique_ids(self):
        records = [
            {"listing_id": 10},
            {"listing_id": 20},
            {"listing_id": 10},
            {"listing_id": 30},
        ]
        result = list(iter_unique_ids(records))
        assert result == [10, 20, 30]

    def test_skips_invalid_ids(self):
        records = [
            {"listing_id": 1},
            {"listing_id": "not_a_number"},
            {"listing_id": 2},
        ]
        result = list(iter_unique_ids(records))
        assert result == [1, 2]

    def test_empty_input(self):
        result = list(iter_unique_ids([]))
        assert result == []


class TestSeenSetDedupe:
    """Tests for seen_set_dedupe with external seen set."""

    def test_uses_external_seen_set(self):
        seen = {1, 2}
        records = [
            {"listing_id": 1},
            {"listing_id": 3},
            {"listing_id": 2},
            {"listing_id": 4},
        ]
        result = list(seen_set_dedupe(records, seen=seen))
        assert len(result) == 2
        assert {r["listing_id"] for r in result} == {3, 4}

    def test_updates_seen_set(self):
        seen = set()
        records = [{"listing_id": 1}, {"listing_id": 2}]
        list(seen_set_dedupe(records, seen=seen))
        assert seen == {1, 2}

    def test_persists_across_batches(self):
        seen = set()
        batch1 = [{"listing_id": 1}, {"listing_id": 2}]
        batch2 = [{"listing_id": 2}, {"listing_id": 3}]

        result1 = list(seen_set_dedupe(batch1, seen=seen))
        result2 = list(seen_set_dedupe(batch2, seen=seen))

        assert len(result1) == 2
        assert len(result2) == 1
        assert result2[0]["listing_id"] == 3
        assert seen == {1, 2, 3}
