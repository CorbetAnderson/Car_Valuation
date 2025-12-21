# src/car_valuation/storage/queries.py
from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class QueryOptions:
    """
    Default options for read queries.

    Attributes:
        schema: Postgres schema (Supabase default is "public").
        default_limit: Default max rows to return for list queries.
    """
    schema: str = "public"
    default_limit: int = 100


def get_table_name(db_cfg: Mapping[str, Any], table_key: str = "scraped_items") -> str:
    """
    Resolve the physical table name from db.yaml.

    Example db.yaml:
      tables:
        scraped_items:
          table: "Cars"

    Returns:
        Actual table name (e.g. "Cars").
    """
    tables = db_cfg.get("tables", {}) or {}
    t = tables.get(table_key, {}) or {}
    table_name = t.get("table")
    if not table_name:
        raise KeyError(f"Missing tables.{table_key}.table in db config")
    return str(table_name)


def health_check(sb: Any, table: str) -> bool:
    """
    Minimal DB connectivity check by selecting 1 row from a known table.
    """
    try:
        (
            sb.client.table(table)
            .select("listing_id")
            .limit(1)
            .execute()
        )
        return True
    except Exception:
        return False


def count_rows(sb: Any, table: str) -> int:
    """
    Count rows in a table.
    """
    resp = (
        sb.client.table(table)
        .select("listing_id", count="exact")
        .limit(1)
        .execute()
    )
    # resp.count is usually available; fall back to len(data) if not
    c = getattr(resp, "count", None)
    if isinstance(c, int):
        return c
    data = getattr(resp, "data", None) or []
    return len(data)


def get_last_scraped(sb: Any, table: str) -> Optional[datetime]:
    """
    Return the most recent created_at from `table` (timestamptz).
    """
    resp = (
        sb.client.table(table)
        .select("created_at")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    data = getattr(resp, "data", None) or []
    if not data:
        return None

    created_at = data[0].get("created_at")
    if not created_at:
        return None
    
    return datetime.fromisoformat(created_at.replace("Z", "+00:00"))


def fetch_by_listing_id(
    sb: Any,
    table: str,
    listing_id: int,
    *,
    columns: str = "*",
) -> Optional[Dict[str, Any]]:
    """
    Fetch a single record by listing_id.
    Returns None if not found.
    """
    resp = (
        sb.client.table(table)
        .select(columns)
        .eq("listing_id", int(listing_id))
        .limit(1)
        .execute()
    )
    data = getattr(resp, "data", None) or []
    return data[0] if data else None


def fetch_recent(
    sb: Any,
    table: str,
    *,
    limit: int = 50,
    order_by: str = "listing_id",
    desc: bool = True,
    columns: str = "*",
) -> List[Dict[str, Any]]:
    """
    Fetch recent rows for quick inspection.
    """
    resp = (
        sb.client.table(table)
        .select(columns)
        .order(order_by, desc=desc)
        .limit(int(limit))
        .execute()
    )
    return list(getattr(resp, "data", None) or [])


def fetch_listing_ids(sb: Any, table: str) -> List[int]:
    """
    Fetch listing_id values (best-effort coercion).
    """
    resp = (
        sb.client.table(table)
        .select("listing_id")
        .execute()
    )
    data = getattr(resp, "data", None) or []
    out: List[int] = []
    for row in data:
        try:
            out.append(int(row.get("listing_id")))
        except Exception:
            continue
    return out


def fetch_existing_ids(sb: Any, table: str, ids: Sequence[int]) -> List[int]:
    """
    Given a list of ids, return the subset that already exist in the table.
    """
    ids = [int(x) for x in ids if x is not None]
    if not ids:
        return []

    resp = (
        sb.client.table(table)
        .select("listing_id")
        .in_("listing_id", ids)
        .execute()
    )
    data = getattr(resp, "data", None) or []
    out: List[int] = []
    for row in data:
        try:
            out.append(int(row.get("listing_id")))
        except Exception:
            continue
    return out


def fetch_range(
    sb: Any,
    table: str,
    *,
    offset: int = 0,
    limit: int = 1000,
    columns: str = "*",
) -> List[Dict[str, Any]]:
    """
    Fetch a slice using offset/limit.
    """
    offset_i = int(offset)
    limit_i = int(limit)
    end = offset_i + limit_i - 1
    resp = (
        sb.client.table(table)
        .select(columns)
        .range(offset_i, end)
        .execute()
    )
    return list(getattr(resp, "data", None) or [])


def fetch_for_training(
    sb: Any,
    table: str,
    *,
    feature_columns: Sequence[str],
    label_column: Optional[str] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch the columns needed for ML training.

    Returns:
        List[dict] where each dict has only the requested columns.
    """
    cols = list(feature_columns)
    if label_column:
        cols.append(label_column)

    columns_str = ",".join(cols) if cols else "*"

    q = sb.client.table(table).select(columns_str)

    if order_by is not None:
        q = q.order(order_by, desc=True)

    if limit is not None:
        q = q.limit(int(limit))

    resp = q.execute()
    return list(getattr(resp, "data", None) or [])
