# src/car_valuation/pipelines/run_scrape.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from src.car_valuation.clients.api_client import APIClient
from src.car_valuation.collection.fetch import batch_iter, iter_search_then_detail
from src.car_valuation.collection.normalise import normalize_car_detail
from src.car_valuation.collection.dedupe import dedupe_batch, DedupeOptions
from src.car_valuation.storage.supabase_client import SupabaseClient
from src.car_valuation.storage.upsert import upsert_from_config, UpsertResult
from src.car_valuation.utils.config import find_project_root, load_env, load_yaml
from src.car_valuation.utils.logging import setup_logging

log = logging.getLogger(__name__)


def run_scrape(
    scrape_cfg: Mapping[str, Any],
    db_cfg: Mapping[str, Any],
    *,
    table_key: str = "scraped_items",
    batch_size_override: Optional[int] = None,
    continue_on_error: bool = True,
) -> UpsertResult:
    """
    End-to-end scrape pipeline:
      search -> detail -> normalize -> dedupe (per-batch) -> upsert

    Runs ALL available records (no artificial limits).
    """
    client = APIClient(scrape_cfg)
    sb = SupabaseClient(db_cfg)

    # Use config batch size unless overridden
    cfg_batch_size = int(
        (
            (((db_cfg.get("tables") or {}).get(table_key) or {}).get("upsert") or {}).get("batch_size")
        )
        or 500
    )
    batch_size = int(batch_size_override or cfg_batch_size)

    log.info("Run scrape starting (table_key=%s, batch_size=%s)", table_key, batch_size)

    # Stream detail payloads
    details = iter_search_then_detail(client, continue_on_error=continue_on_error)

    # Normalize stream
    def normalized_stream() -> Iterator[Dict[str, Any]]:
        for d in details:
            try:
                out = normalize_car_detail(d)
                # drop records with missing ids at the pipeline boundary
                if out and out.get("listing_id") is not None:
                    yield out
            except Exception:
                if not continue_on_error:
                    raise
                # keep it quiet, but still record the exception in logs
                log.exception("Normalize failed")

    attempted_total = 0
    succeeded_total = 0
    failed_batches_total = 0
    errors: List[str] = []

    batch_num = 0

    for batch in batch_iter(normalized_stream(), batch_size=batch_size):
        # Dedupe within the batch
        deduped = dedupe_batch(
            batch,
            options=DedupeOptions(id_field="listing_id", prefer_last=True, drop_missing_ids=True),
        )
        if not deduped:
            continue

        attempted_total += len(deduped)

        res = upsert_from_config(
            sb,
            db_cfg,
            deduped,
            table_key=table_key,
            continue_on_error=continue_on_error,
        )

        batch_num += 1

        succeeded_total += res.succeeded
        failed_batches_total += res.failed_batches
        errors.extend(list(res.errors))

        if res.failed_batches == 0:
            log.info(
                "Batch %s upserted OK (rows=%s, succeeded_total=%s)",
                batch_num,
                len(deduped),
                succeeded_total,
            )
        else:
            log.warning(
                "Batch %s upsert had errors (rows=%s, failed_batches_in_call=%s, succeeded_total=%s)",
                batch_num,
                len(deduped),
                res.failed_batches,
                succeeded_total,
            )

        if (not continue_on_error) and res.failed_batches:
            break

    # Determine the real table name for reporting
    table_name = (((db_cfg.get("tables") or {}).get(table_key) or {}).get("table")) or table_key

    summary = UpsertResult(
        table=str(table_name),
        attempted=int(attempted_total),
        succeeded=int(succeeded_total),
        failed_batches=int(failed_batches_total),
        errors=tuple(errors),
    )

    log.info(
        "Run scrape finished (table=%s attempted=%s succeeded=%s failed_batches=%s)",
        summary.table,
        summary.attempted,
        summary.succeeded,
        summary.failed_batches,
    )

    if summary.errors:
        # Only show first few to avoid spam
        for msg in summary.errors[:5]:
            log.error(msg)
        if len(summary.errors) > 5:
            log.error("... plus %s more errors", len(summary.errors) - 5)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape -> normalize -> upsert pipeline (all records).")
    parser.add_argument("--scrape", default="configs/scrape.yml", help="Path to scrape.yml")
    parser.add_argument("--db", default="configs/db.yml", help="Path to db.yml")
    parser.add_argument("--table-key", default="scraped_items", help="tables.<key> to use from db.yml")
    parser.add_argument("--batch-size", type=int, default=None, help="Override db.yml upsert.batch_size")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error (fetch/normalize/upsert)")
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    setup_logging(level=args.log_level, name=None)

    root = find_project_root()
    load_env(root)  # loads .env so OAuth + Supabase env vars exist

    scrape_path = (Path(root) / args.scrape).resolve() if not Path(args.scrape).is_absolute() else Path(args.scrape)
    db_path = (Path(root) / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)

    scrape_cfg = load_yaml(scrape_path)
    db_cfg = load_yaml(db_path)

    run_scrape(
        scrape_cfg,
        db_cfg,
        table_key=str(args.table_key),
        batch_size_override=args.batch_size,
        continue_on_error=not bool(args.fail_fast),
    )


if __name__ == "__main__":
    main()
