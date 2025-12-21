# src/car_valuation/collection/fetch.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List

from src.car_valuation.clients.api_client import APIClient

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchConfig:
    """
    Trade Me Motors Used search response puts records under "List"
    and each record has a "ListingId".
    """
    endpoint_name: str = "search"
    record_path: str = "List"
    id_field: str = "ListingId"


@dataclass(frozen=True)
class DetailConfig:
    """
    How to call the detail endpoint.

    scrape.yaml path should look like: /v1/Listings/{listing_id}.json
    """
    endpoint_name: str = "detail"
    path_param_name: str = "listing_id"


def iter_search_records(
    client: APIClient,
    *,
    cfg: SearchConfig = SearchConfig(),
) -> Iterator[Dict[str, Any]]:
    """
    Stream records from the search endpoint.
    """

    for rec in client.iter_endpoint_records(
        cfg.endpoint_name,
        record_path=cfg.record_path,
    ):
        yield rec


def iter_listing_id_pairs(
    client: APIClient,
    *,
    search_cfg: SearchConfig = SearchConfig(),
) -> Iterator[tuple[int, Dict[str, Any]]]:
    """
    Yield (ListingId, search_record) pairs from search results.
    """
    for rec in iter_search_records(client, cfg=search_cfg):
        lid = rec.get(search_cfg.id_field)
        if lid is None:
            continue
        try:
            yield int(lid), rec
        except (TypeError, ValueError):
            continue


def fetch_detail(
    client: APIClient,
    listing_id: int,
    *,
    detail_cfg: DetailConfig = DetailConfig(),
) -> Dict[str, Any]:
    """
    Fetch a single listing detail payload.
    """
    return client.fetch_endpoint_json(
        detail_cfg.endpoint_name,
        path_params={detail_cfg.path_param_name: listing_id},
    )

def enrich_detail_with_search_fields(
    detail: Dict[str, Any],
    search_rec: Dict[str, Any],
    *,
    fields: tuple[str, ...] = ("Make", "Model", "Is4WD", "IsDealer"),
) -> Dict[str, Any]:
    """
    Attach selected fields from the search record onto the detail payload.
    """
    detail = dict(detail)
    detail.update({k: search_rec.get(k) for k in fields})
    return detail

def iter_detail_payloads(
    client: APIClient,
    listing_ids: Iterable[int],
    *,
    detail_cfg: DetailConfig = DetailConfig(),
    continue_on_error: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Stream detail JSON payloads for a list/iterator of listing IDs.

    If continue_on_error=True, errors are logged and the generator continues.
    """
    for lid in listing_ids:
        try:
            yield fetch_detail(client, int(lid), detail_cfg=detail_cfg)
        except Exception as e:
            if not continue_on_error:
                raise
            log.warning("Detail fetch failed for listing_id=%s: %s", lid, e)


def iter_search_then_detail(
    client: APIClient,
    *,
    search_cfg: SearchConfig = SearchConfig(),
    detail_cfg: DetailConfig = DetailConfig(),
    continue_on_error: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    search -> (id, search_rec) -> detail -> enriched detail
    """
    for listing_id, search_rec in iter_listing_id_pairs(client, search_cfg=search_cfg):
        try:
            detail = fetch_detail(client, listing_id, detail_cfg=detail_cfg)
            yield enrich_detail_with_search_fields(detail, search_rec)
        except Exception as e:
            if not continue_on_error:
                raise
            log.warning("Detail fetch failed for listing_id=%s: %s", listing_id, e)


def batch_iter(
    it: Iterable[Dict[str, Any]],
    batch_size: int,
) -> Iterator[List[Dict[str, Any]]]:
    """
    Utility: turn a stream of dicts into batches for upsert.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    batch: List[Dict[str, Any]] = []
    for item in it:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

