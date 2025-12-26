# src/car_valuation/datasets/stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass
class DatasetStats:
    """
    Summary statistics for a dataset artifact.

    Intended to be written as:
      - stats.json (machine-readable)
      - stats.md (human-readable)
    """
    n_rows: int
    n_train: int
    n_val: int
    n_test: int
    missingness: Dict[str, float]          # col -> fraction missing
    target_summary: Dict[str, float]       # e.g., mean/median/min/max
    embedding_presence: Dict[str, float]   # e.g., has_text_emb %, has_image_emb %
    top_categories: Dict[str, List[str]]   # col -> top-k values


def compute_dataset_stats(df: Any, splits: Mapping[str, Sequence[int]], cfg: Mapping[str, Any]) -> DatasetStats:
    """
    Compute QA stats for the built dataset.

    Args:
        df: full dataframe (post-cleaning, pre or post preprocessing—your choice, but be consistent).
        splits: mapping of split name -> listing_id list.
        cfg: optional knobs (top_k categories, which cols to compute, etc.)

    Returns:
        DatasetStats
    """
    raise NotImplementedError


def write_stats_json(stats: DatasetStats, path: str) -> None:
    """
    Write dataset stats to a JSON file.
    """
    raise NotImplementedError


def write_stats_md(stats: DatasetStats, path: str) -> None:
    """
    Write dataset stats to a small, readable Markdown report.

    This is great for GitHub portfolio value.
    """
    raise NotImplementedError
