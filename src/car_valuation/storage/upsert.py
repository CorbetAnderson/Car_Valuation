# src/car_valuation/storage/upsert.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpsertResult:
    """
    Summary of an upsert run.

    Attributes:
        table: Table name upserted into.
        attempted: Total records attempted.
        succeeded: Total records accepted by Supabase (best-effort count).
        failed_batches: Number of batches that raised exceptions.
        errors: List of string error messages (one per failed batch).
    """
    table: str
    attempted: int
    succeeded: int
    failed_batches: int
    errors: Tuple[str, ...] = ()


def _get_table_cfg(db_cfg: Mapping[str, Any], table_key: str) -> Mapping[str, Any]:
    tables = db_cfg.get("tables", {}) or {}
    if table_key not in tables:
        raise KeyError(f"Missing tables.{table_key} in db config")
    return tables[table_key] or {}


def _chunk(records: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(records), batch_size):
        yield list(records[i : i + batch_size])


def upsert_records(
    sb: Any,
    table: str,
    records: Sequence[Dict[str, Any]],
    *,
    conflict_columns: Sequence[str],
    batch_size: int = 500,
    ignore_duplicates: bool = False,
    continue_on_error: bool = True,
) -> UpsertResult:
    """
    Upsert records into a Supabase table.

    Args:
        sb: SupabaseClient wrapper OR a raw supabase client.
        table: Table name (e.g. "Cars").
        records: List of dict payloads (already normalized).
        conflict_columns: Column(s) used for upsert conflict resolution (e.g. ["listing_id"]).
        batch_size: Batch size for each upsert call.
        ignore_duplicates: If True, sets upsert(ignore_duplicates=True) where supported.
        continue_on_error: If True, errors are logged and remaining batches continue.

    Returns:
        UpsertResult summary.

    Notes:
        - Supabase response doesn't always clearly report inserted vs updated counts,
          so `succeeded` is best-effort (len(returned rows) if available, else batch size).
    """
    client = getattr(sb, "client", sb)

    if not records:
        return UpsertResult(table=table, attempted=0, succeeded=0, failed_batches=0, errors=())

    if not conflict_columns:
        raise ValueError("conflict_columns must be provided for upsert")

    on_conflict = ",".join(conflict_columns)

    attempted = len(records)
    succeeded = 0
    failed_batches = 0
    errors: List[str] = []

    for batch in _chunk(records, batch_size):
        try:
            q = client.table(table).upsert(batch, on_conflict=on_conflict)

            if ignore_duplicates:
                try:
                    q = client.table(table).upsert(
                        batch, on_conflict=on_conflict, ignore_duplicates=True
                    )
                except TypeError:
                    q = client.table(table).upsert(batch, on_conflict=on_conflict)

            resp = q.execute()
            data = getattr(resp, "data", None)

            if isinstance(data, list):
                succeeded += len(data)
            else:
                succeeded += len(batch)

        except Exception as e:
            failed_batches += 1
            msg = f"Upsert failed for batch (size={len(batch)}): {e}"
            errors.append(msg)
            log.exception(msg)

            if not continue_on_error:
                break

    return UpsertResult(
        table=table,
        attempted=attempted,
        succeeded=succeeded,
        failed_batches=failed_batches,
        errors=tuple(errors),
    )


def upsert_from_config(
    sb: Any,
    db_cfg: Mapping[str, Any],
    records: Sequence[Dict[str, Any]],
    *,
    table_key: str = "scraped_items",
    continue_on_error: bool = True,
) -> UpsertResult:
    """
    Upsert using db.yaml config (tables.<table_key>.upsert + tables.<table_key>.table).

    Args:
        sb: Supabase client wrapper or raw client.
        db_cfg: Loaded db.yaml dict.
        records: Normalized canonical records.
        table_key: Which logical table config to use.
        continue_on_error: Continue after batch failures.

    Returns:
        UpsertResult
    """
    tcfg = _get_table_cfg(db_cfg, table_key)
    table = tcfg.get("table")
    if not table:
        raise KeyError(f"Missing tables.{table_key}.table in db config")

    up = (tcfg.get("upsert", {}) or {})
    enabled = bool(up.get("enabled", True))
    if not enabled:
        return UpsertResult(table=str(table), attempted=len(records), succeeded=0, failed_batches=0, errors=())

    conflict_columns = up.get("conflict_columns") or []
    batch_size = int(up.get("batch_size", 500))
    ignore_duplicates = bool(up.get("ignore_duplicates", False))

    return upsert_records(
        sb,
        str(table),
        records,
        conflict_columns=list(conflict_columns),
        batch_size=batch_size,
        ignore_duplicates=ignore_duplicates,
        continue_on_error=continue_on_error,
    )
