# src/car_valuation/datasets/splits.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration for splitting dataset rows into train/val/test.

    split_type:
      - "time": time-based split using time_col
      - "random": random split using seed
      - "group": group-aware split to reduce leakage (e.g., group by make+model)
    """
    split_type: str  # "time" | "random" | "group"
    time_col: str = "scraped_at"
    group_col: Optional[str] = None
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42


def time_split(ids: Sequence[int], times: Sequence, cfg: SplitConfig) -> Dict[str, List[int]]:
    """
    Create a time-based split.

    Args:
        ids: listing identifiers aligned with `times`.
        times: timestamps aligned with `ids` (must be sortable).
        cfg: SplitConfig with split_type="time" and fractions.

    Returns:
        Dict with keys: "train", "val", "test" mapping to listing_id lists.

    Notes:
        - Sort by time ascending (oldest -> newest).
        - Allocate rows by fraction.
        - This reduces future leakage and mirrors real deployment.
    """
    raise NotImplementedError


def random_split(ids: Sequence[int], cfg: SplitConfig) -> Dict[str, List[int]]:
    """
    Create a random split (fallback for when time split is not suitable).

    Args:
        ids: listing identifiers.
        cfg: SplitConfig with split_type="random".

    Returns:
        Dict with keys: "train", "val", "test".
    """
    raise NotImplementedError


def group_split(ids: Sequence[int], groups: Sequence[str], cfg: SplitConfig) -> Dict[str, List[int]]:
    """
    Create a group-aware split (e.g., keep all 'Toyota Corolla' together).

    Args:
        ids: listing identifiers aligned with `groups`.
        groups: group labels aligned with `ids`.
        cfg: SplitConfig with split_type="group" and group_col set.

    Returns:
        Dict with keys: "train", "val", "test".

    Notes:
        - Prevents leakage where the same make/model appears in train and test.
        - Best effort: exact balancing isn't always possible.
    """
    raise NotImplementedError


def validate_split_fractions(cfg: SplitConfig) -> None:
    """
    Validate that train/val/test fractions are positive and sum to 1.0 (within tolerance).

    Raises:
        ValueError: on invalid fractions.
    """
    raise NotImplementedError
