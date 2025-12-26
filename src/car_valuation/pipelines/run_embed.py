# src/car_valuation/pipelines/run_embed.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Mapping, Optional

from src.car_valuation.embeddings.model import load_openclip_from_config, LoadedOpenCLIP
from src.car_valuation.embeddings.embed import (
    EmbedConfig,
    EmbedResult,
    embed_row,
    results_to_records,
    resolve_input_fields,
    validate_loaded_model,
    embed_cfg_table_key,
)
from src.car_valuation.embeddings.preprocess import PreprocessConfig
from src.car_valuation.embeddings.image_fetch import ImageFetchConfig
from src.car_valuation.embeddings.cache import CacheConfig, filter_missing_embedding_ids

from src.car_valuation.storage.supabase_client import SupabaseClient
from src.car_valuation.storage.upsert import upsert_from_config, UpsertResult
from src.car_valuation.storage.queries import get_table_name
from src.car_valuation.storage.embeddings import fetch_cars_for_embedding

from src.car_valuation.utils.config import find_project_root, load_env, load_yaml
from src.car_valuation.utils.logging import setup_logging

log = logging.getLogger(__name__)

def _override_upsert_batch_size(db_cfg: Mapping[str, Any], table_key: str, batch_size_override: Optional[int]) -> None:
    """
    Best-effort: override db.yml tables.<table_key>.upsert.batch_size in-memory.
    """
    if batch_size_override is None:
        return
    try:
        tables = db_cfg.get("tables") or {}
        t = tables.get(table_key) or {}
        up = t.get("upsert") or {}
        up["batch_size"] = int(batch_size_override)
        t["upsert"] = up
        tables[table_key] = t
        # if db_cfg is a plain dict, this mutation sticks
        db_cfg["tables"] = tables  # type: ignore[index]
    except Exception:
        # not fatal; you'll just use config batch_size
        log.warning("Could not override upsert batch_size for table_key=%s", table_key)


def run_embed(
    embed_cfg: Mapping[str, Any],
    db_cfg: Mapping[str, Any],
    *,
    cars_table_key: str = "scraped_items",
    embeddings_table_key: Optional[str] = None,
    page_size: int = 500,
    batch_size_override: Optional[int] = None,
    max_pages: Optional[int] = None,
    continue_on_error: bool = True,
) -> UpsertResult:
    """
    End-to-end embedding pipeline:
      page Cars -> (optional) skip existing embeddings -> embed -> upsert

    Notes:
      - This is designed to scale: it checks "already embedded?" per page, not by loading 50k ids.
      - Upserts into tables.<embeddings_table_key> from db.yml, default taken from embed.yml output.table_key.
    """
    sb = SupabaseClient(db_cfg)

    # Resolve table keys / names
    embeddings_table_key = str(embeddings_table_key or embed_cfg_table_key(embed_cfg) or "embeddings")
    cars_table = get_table_name(db_cfg, table_key=cars_table_key)
    emb_table = get_table_name(db_cfg, table_key=embeddings_table_key)

    # Apply upsert batch size override to embeddings table (in-memory)
    _override_upsert_batch_size(db_cfg, embeddings_table_key, batch_size_override)

    # Resolve input fields
    text_field, image_field = resolve_input_fields(embed_cfg)

    # Load model bundle
    loaded = load_openclip_from_config(embed_cfg)
    validate_loaded_model(loaded)

    model_tag = str(getattr(loaded, "model_tag", "") or "").strip()
    embed_dim = int(getattr(loaded, "embed_dim", 0) or 0)

    # Component configs (defaults; later you can read from embed.yml sections if you add them)
    preprocess_cfg = PreprocessConfig()
    image_fetch_cfg = ImageFetchConfig()
    cache_cfg = CacheConfig(enabled=True, skip_if_exists=True, batch_size=500)
    embed_runtime_cfg = EmbedConfig(normalize=True, require_one_of=True)

    log.info(
        "Run embed starting (cars_table=%s embeddings_table=%s embeddings_table_key=%s model_tag=%s embed_dim=%s page_size=%s)",
        cars_table,
        emb_table,
        embeddings_table_key,
        model_tag,
        embed_dim,
        int(page_size),
    )

    attempted_total = 0
    succeeded_total = 0
    failed_batches_total = 0
    errors: List[str] = []

    offset = 0
    page_num = 0

    while True:
        if max_pages is not None and page_num >= int(max_pages):
            log.info("Reached max_pages=%s, stopping.", max_pages)
            break

        # Fetch one page of Cars (only the needed columns)
        rows = fetch_cars_for_embedding(
            sb,
            table=cars_table,
            text_field=text_field,
            image_field=image_field,
            offset=int(offset),
            limit=int(page_size),
        )

        if not rows:
            log.info("No more Cars rows at offset=%s. Done.", offset)
            break

        page_num += 1

        # Optional: skip ids that already exist for (listing_id, model_tag)
        listing_ids: List[int] = []
        for r in rows:
            try:
                listing_ids.append(int(r.get("listing_id")))
            except Exception:
                continue

        ids_to_embed = listing_ids
        if cache_cfg.enabled and cache_cfg.skip_if_exists and listing_ids:
            ids_to_embed = filter_missing_embedding_ids(
                sb,
                db_cfg=db_cfg,
                table_key=embeddings_table_key,
                model_tag=model_tag,
                listing_ids=listing_ids,
                cfg=cache_cfg,
            )

        ids_to_embed_set = set(int(x) for x in ids_to_embed)
        rows_to_embed = []
        for r in rows:
            try:
                if int(r.get("listing_id")) in ids_to_embed_set:
                    rows_to_embed.append(r)
            except Exception:
                continue

        if not rows_to_embed:
            log.info("Page %s: nothing to embed after cache-skip (offset=%s, fetched=%s).", page_num, offset, len(rows))
            offset += int(page_size)
            continue

        # Embed this page
        results: List[EmbedResult] = []
        try:
            for r in rows_to_embed:
                results.append(
                    embed_row(
                        r,
                        loaded=loaded,
                        embed_cfg=embed_cfg,
                        preprocess_cfg=preprocess_cfg,
                        image_fetch_cfg=image_fetch_cfg,
                        cfg=embed_runtime_cfg,
                    )
                )
        except Exception:
            log.exception("Embedding failed for page %s (offset=%s).", page_num, offset)
            if not continue_on_error:
                raise

        # Convert to DB payloads
        try:
            records = results_to_records(results, embed_dim=embed_dim)
        except Exception:
            log.exception("results_to_records failed for page %s (offset=%s).", page_num, offset)
            if not continue_on_error:
                raise
            records = []

        attempted_total += len(records)

        # Collect any embed_row-level errors (best effort)
        for r in results:
            if getattr(r, "error", None):
                errors.append(f"listing_id={getattr(r, 'listing_id', None)} error={getattr(r, 'error')}")

        # Upsert into embeddings table via db.yml table_key
        res = upsert_from_config(
            sb,
            db_cfg,
            records,
            table_key=embeddings_table_key,
            continue_on_error=continue_on_error,
        )

        succeeded_total += res.succeeded
        failed_batches_total += res.failed_batches
        errors.extend(list(res.errors))

        if res.failed_batches == 0:
            log.info(
                "Page %s upsert OK (offset=%s fetched=%s to_embed=%s records=%s succeeded_total=%s)",
                page_num,
                offset,
                len(rows),
                len(rows_to_embed),
                len(records),
                succeeded_total,
            )
        else:
            log.warning(
                "Page %s upsert had errors (offset=%s records=%s failed_batches_in_call=%s succeeded_total=%s)",
                page_num,
                offset,
                len(records),
                res.failed_batches,
                succeeded_total,
            )

        if (not continue_on_error) and res.failed_batches:
            break

        offset += int(page_size)

    summary = UpsertResult(
        table=str(emb_table),
        attempted=int(attempted_total),
        succeeded=int(succeeded_total),
        failed_batches=int(failed_batches_total),
        errors=tuple(errors),
    )

    log.info(
        "Run embed finished (table=%s attempted=%s succeeded=%s failed_batches=%s)",
        summary.table,
        summary.attempted,
        summary.succeeded,
        summary.failed_batches,
    )

    if summary.errors:
        for msg in summary.errors[:5]:
            log.error(msg)
        if len(summary.errors) > 5:
            log.error("... plus %s more errors", len(summary.errors) - 5)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Cars -> upsert into CarEmbeddings (all records).")
    parser.add_argument("--embed", default="configs/embed.yml", help="Path to embed.yml")
    parser.add_argument("--db", default="configs/db.yml", help="Path to db.yml")
    parser.add_argument("--cars-table-key", default="scraped_items", help="tables.<key> to read Cars from db.yml")
    parser.add_argument(
        "--embeddings-table-key",
        default=None,
        help="Override output table_key (default: embed.yml output.table_key)",
    )
    parser.add_argument("--page-size", type=int, default=500, help="How many Cars rows to fetch per page")
    parser.add_argument("--batch-size", type=int, default=None, help="Override db.yml embeddings upsert.batch_size")
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages (debug/safety)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error (embed/upsert)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    setup_logging(level=args.log_level, name=None)

    root = find_project_root()
    load_env(root)  # loads .env so Supabase env vars exist

    embed_path = (Path(root) / args.embed).resolve() if not Path(args.embed).is_absolute() else Path(args.embed)
    db_path = (Path(root) / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)

    embed_cfg = load_yaml(embed_path)
    db_cfg = load_yaml(db_path)

    run_embed(
        embed_cfg,
        db_cfg,
        cars_table_key=str(args.cars_table_key),
        embeddings_table_key=(str(args.embeddings_table_key) if args.embeddings_table_key else None),
        page_size=int(args.page_size),
        batch_size_override=args.batch_size,
        max_pages=(int(args.max_pages) if args.max_pages is not None else None),
        continue_on_error=not bool(args.fail_fast),
    )


if __name__ == "__main__":
    main()
