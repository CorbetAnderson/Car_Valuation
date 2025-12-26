# src/car_valuation/storage/embeddings.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def fetch_cars_for_embedding(
    sb: Any,
    *,
    table: str,
    text_field: str,
    image_field: str,
    offset: int = 0,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Fetch a slice of Cars rows needed for embedding.
    """
    offset_i = int(offset)
    limit_i = int(limit)
    end = offset_i + limit_i - 1

    columns = f"listing_id,{text_field},{image_field}"
    resp = (
        sb.client.table(table)
        .select(columns)
        .range(offset_i, end)
        .execute()
    )
    return list(getattr(resp, "data", None) or [])


def fetch_existing_embedding_ids_for_model(
    sb: Any,
    *,
    table: str,
    model_tag: str,
    listing_ids: Sequence[int],
) -> List[int]:
    """
    Return listing_ids that already exist in CarEmbeddings for a given model tag.
    """

    resp = (
        sb.client.table(table)
        .select("listing_id")
        .eq("model", model_tag)
        .in_("listing_id", listing_ids)
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


def fetch_embedding_row(
    sb: Any,
    *,
    table: str,
    listing_id: int,
    model_tag: str,
    columns: str = "*",
) -> Optional[Dict[str, Any]]:
    """
    Fetch one embedding row for debugging/inspection.
    """
    resp = (
        sb.client.table(table)
        .select(columns)
        .eq("listing_id", int(listing_id))
        .eq("model", model_tag)
        .limit(1)
        .execute()
    )
    data = getattr(resp, "data", None) or []
    return data[0] if data else None
