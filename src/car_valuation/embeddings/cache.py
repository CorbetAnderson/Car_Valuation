# src/car_valuations/embeddings/cache
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Sequence, Set
from src.car_valuation.storage.queries import get_table_name
from src.car_valuation.storage.embeddings import fetch_existing_embedding_ids_for_model



@dataclass(frozen=True)
class CacheConfig:
    """
    Controls how embedding caching / skipping behaves.

    Attributes:
        enabled: If False, never skip; always compute and upsert.
        skip_if_exists: If True, skip rows that already have an embedding row for (listing_id, model).
        batch_size: How many ids to check at once when querying CarEmbeddings.
    """
    enabled: bool = True
    skip_if_exists: bool = True
    batch_size: int = 500


def _chunk_ids(ids: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(ids), batch_size):
        yield list(ids[i : i + batch_size])


def filter_missing_embedding_ids(
    sb: Any,
    *,
    db_cfg: Mapping[str, Any],
    table_key: str,
    model_tag: str,
    listing_ids: Sequence[int],
    cfg: CacheConfig,
) -> List[int]:
    """
    Return only listing_ids that DO NOT already have an embedding row for (listing_id, model_tag).
    """
    ids = [int(x) for x in listing_ids if x is not None]
    if not ids:
        return []

    # If caching disabled (or skipping disabled), dont skip anything
    if not cfg.enabled or not cfg.skip_if_exists:
        return ids

    embeddings_table = get_table_name(db_cfg, table_key=table_key)

    existing: set[int] = set()
    for batch in _chunk_ids(ids, cfg.batch_size):
        found = fetch_existing_embedding_ids_for_model(
            sb,
            table=embeddings_table,
            model_tag=model_tag,
            listing_ids=batch,
        )
        existing.update(int(x) for x in found)

    return [i for i in ids if i not in existing]
    
