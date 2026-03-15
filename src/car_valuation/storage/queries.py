# src/car_valuation/storage/queries.py
from __future__ import annotations

from typing import Any, Mapping


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
