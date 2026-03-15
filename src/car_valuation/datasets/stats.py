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
    import pandas as pd

    n_rows = len(df)
    n_train = len(splits.get("train", []))
    n_val = len(splits.get("val", []))
    n_test = len(splits.get("test", []))

    # Missingness per column
    missingness: Dict[str, float] = {}
    for col in df.columns:
        if df[col].dtype != object or df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            continue  # Skip embedding columns
        missing_frac = df[col].isna().sum() / len(df) if len(df) > 0 else 0.0
        missingness[col] = round(missing_frac, 4)

    # Target summary (assuming 'y' column)
    target_summary: Dict[str, float] = {}
    if "y" in df.columns:
        target_summary = {
            "mean": round(df["y"].mean(), 4),
            "median": round(df["y"].median(), 4),
            "min": round(df["y"].min(), 4),
            "max": round(df["y"].max(), 4),
            "std": round(df["y"].std(), 4),
        }

    # Embedding presence
    embedding_presence: Dict[str, float] = {}
    for col in ["text_embedding", "image_embedding"]:
        if col in df.columns:
            present = df[col].notna().sum() / len(df) if len(df) > 0 else 0.0
            embedding_presence[col] = round(present, 4)

    # Top categories
    top_k = cfg.get("stats_top_k_categories", 10)
    categorical_cols = cfg.get("categorical", [])
    top_categories: Dict[str, List[str]] = {}
    for col in categorical_cols:
        if col in df.columns:
            top_vals = df[col].value_counts().head(top_k).index.tolist()
            top_categories[col] = [str(v) for v in top_vals]

    return DatasetStats(
        n_rows=n_rows,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        missingness=missingness,
        target_summary=target_summary,
        embedding_presence=embedding_presence,
        top_categories=top_categories,
    )


def write_stats_json(stats: DatasetStats, path: str) -> None:
    """
    Write dataset stats to a JSON file.
    """
    import json
    from dataclasses import asdict

    with open(path, "w") as f:
        json.dump(asdict(stats), f, indent=2)


def write_stats_md(stats: DatasetStats, path: str) -> None:
    """
    Write dataset stats to a small, readable Markdown report.

    This is great for GitHub portfolio value.
    """
    lines = [
        "# Dataset Statistics",
        "",
        "## Overview",
        f"- **Total rows**: {stats.n_rows:,}",
        f"- **Train**: {stats.n_train:,} ({stats.n_train/stats.n_rows*100:.1f}%)" if stats.n_rows > 0 else "- **Train**: 0",
        f"- **Val**: {stats.n_val:,} ({stats.n_val/stats.n_rows*100:.1f}%)" if stats.n_rows > 0 else "- **Val**: 0",
        f"- **Test**: {stats.n_test:,} ({stats.n_test/stats.n_rows*100:.1f}%)" if stats.n_rows > 0 else "- **Test**: 0",
        "",
    ]

    if stats.target_summary:
        lines.extend([
            "## Target (y)",
            f"- Mean: {stats.target_summary.get('mean', 'N/A')}",
            f"- Median: {stats.target_summary.get('median', 'N/A')}",
            f"- Min: {stats.target_summary.get('min', 'N/A')}",
            f"- Max: {stats.target_summary.get('max', 'N/A')}",
            f"- Std: {stats.target_summary.get('std', 'N/A')}",
            "",
        ])

    if stats.embedding_presence:
        lines.extend([
            "## Embeddings",
        ])
        for col, pct in stats.embedding_presence.items():
            lines.append(f"- {col}: {pct*100:.1f}% present")
        lines.append("")

    if stats.top_categories:
        lines.extend([
            "## Top Categories",
        ])
        for col, vals in stats.top_categories.items():
            lines.append(f"- **{col}**: {', '.join(vals[:5])}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
