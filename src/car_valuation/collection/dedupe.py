# src/my_project/collection/dedupe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Set
from src.car_valuation.collection.normalise import parse_int_from_text


@dataclass(frozen=True)
class DedupeOptions:
    """
    Controls deduplication behavior for canonical records.

    Attributes:
        id_field: Field to use as the unique identifier (default "listing_id").
        prefer_last: If duplicates appear, keep the last occurrence (True) or first (False).
        drop_missing_ids: If True, skip records where id_field is missing/None/not int-able.
                          If False, raise ValueError on such records.
    """
    id_field: str = "listing_id"
    prefer_last: bool = True
    drop_missing_ids: bool = True


def dedupe_batch(
    records: Iterable[Mapping[str, Any]],
    *,
    options: DedupeOptions = DedupeOptions(),
) -> List[Dict[str, Any]]:
    """
    Dedupe a batch of records by listing_id (or id_field).

    Use this when you already have batches (e.g., right before upsert).

    Rules:
      - If prefer_last=True, later duplicates overwrite earlier ones.
      - If prefer_last=False, first occurrence wins.
      - If drop_missing_ids=True, records without a usable id are skipped.

    Returns:
        List of dict records with unique ids.
    """

    by_id: Dict[int, Dict[str, Any]] = {}

    for r in records:
        raw_id = r.get(options.id_field)
        lid = parse_int_from_text(raw_id)

        if lid is None:
            if options.drop_missing_ids:
                continue
            raise ValueError(f"Record missing/invalid {options.id_field}: {raw_id!r}")

        if options.prefer_last:
            by_id[lid] = dict(r)
        else:
            by_id.setdefault(lid, dict(r))  # keep first

    return list(by_id.values())


def iter_deduped_records(
    records: Iterable[Mapping[str, Any]],
    *,
    options: DedupeOptions = DedupeOptions(),
) -> Iterator[Dict[str, Any]]:
    """
    Stream dedupe records by listing_id across an entire run.

    Behavior:
      - Tracks seen ids in-memory using a set.
      - If a duplicate id is encountered:
          continue
      - drop_missing_ids controls whether missing ids are skipped or raise.

    Yields:
        Deduped dict records.
    """
    seen_ids: set[int] = set()

    for r in records:
        lid = parse_int_from_text(r.get(options.id_field))

        if lid is None:
            if options.drop_missing_ids:
                continue
            raise ValueError(f"Record missing/invalid {options.id_field}: {lid!r}")

        if lid in seen_ids:
            continue
        
        seen_ids.add(lid)
        yield dict(r)


def iter_unique_ids(
    records: Iterable[Mapping[str, Any]],
    *,
    options: DedupeOptions = DedupeOptions(),
) -> Iterator[int]:
    """
    Yield unique listing ids from a stream of records.

    Useful if you want to:
      - build a DB query for existing ids
      - or log coverage stats

    Yields:
        Unique int ids (each at most once).
    """

    seen_ids: set[int] = set()

    for r in records:
        lid = parse_int_from_text(r.get(options.id_field))

        if lid is None:
            if options.drop_missing_ids:
                continue
            raise ValueError(f"Record missing/invalid {options.id_field}: {lid!r}")

        if lid in seen_ids:
            continue
        
        seen_ids.add(lid)
        yield lid


def seen_set_dedupe(
    records: Iterable[Mapping[str, Any]],
    *,
    seen: Set[int],
    options: DedupeOptions = DedupeOptions(),
) -> Iterator[Dict[str, Any]]:
    """
    Dedupe records using an externally-managed `seen` set.

    Use this if want the pipeline to control memory/state, e.g.:
      - keep `seen` across multiple pages or multiple runs in one process

    Yields:
        Records not yet seen, while adding ids into `seen`.
    """

    for r in records:
        lid = parse_int_from_text(r.get(options.id_field))

        if lid is None:
            if options.drop_missing_ids:
                continue
            raise ValueError(f"Record missing/invalid {options.id_field}: {lid!r}")

        if lid in seen:
            continue
        
        seen.add(lid)
        yield dict(r)