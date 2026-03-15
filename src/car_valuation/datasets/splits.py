# src/car_valuation/datasets/splits.py
from __future__ import annotations

import random
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
      - "holdout": hold out a specific model for val/test, train on other models of same make
    """
    split_type: str  # "random" | "holdout"
    time_col: str = "scraped_at"
    group_col: Optional[str] = None
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42
    holdout_make: Optional[str] = None
    holdout_model: Optional[str] = None


def random_split(ids: Sequence[int], cfg: SplitConfig) -> Dict[str, List[int]]:
    """
    Create a random split (fallback for when time split is not suitable).

    Args:
        ids: listing identifiers.
        cfg: SplitConfig with split_type="random".

    Returns:
        Dict with keys: "train", "val", "test".
    """

    validate_split_fractions(cfg)
    
    if cfg.split_type != "random":
        raise ValueError("random_split called with split_type != 'random'")
    
    # Set seed and shuffle the ids
    ids_list = list(ids)
    rng = random.Random(cfg.seed)
    rng.shuffle(ids_list)

    num_ids = len(ids_list)

    # Splitting indices
    test_idx = int(num_ids * cfg.test_frac)
    val_idx = test_idx + int(num_ids * cfg.val_frac)

    return {
        "train": ids_list[val_idx:num_ids],
        "val": ids_list[test_idx:val_idx],
        "test": ids_list[0:test_idx]
    }

def holdout_split(
    ids: Sequence[int],
    makes: Sequence[str],
    models: Sequence[str],
    cfg: SplitConfig
) -> Dict[str, List[int]]:
    """
    Create a model holdout split: train on other models of the same make,
    val/test on the held-out model (50/50 split).

    Args:
        ids: listing identifiers aligned with `makes` and `models`.
        makes: make labels aligned with `ids` (e.g., "BMW", "Toyota").
        models: model labels aligned with `ids` (e.g., "e46", "Corolla").
        cfg: SplitConfig with split_type="holdout", holdout_make and holdout_model set.

    Returns:
        Dict with keys: "train", "val", "test".

    Example:
        For holdout_make="BMW", holdout_model="e46":
        - train: all BMW listings except e46
        - val: 50% of BMW e46 listings
        - test: 50% of BMW e46 listings
    """
    if cfg.split_type != "holdout":
        raise ValueError("holdout_split called with split_type != 'holdout'")

    if cfg.holdout_make is None or cfg.holdout_model is None:
        raise ValueError("holdout_make and holdout_model must be set for holdout split")

    train_ids: List[int] = []
    holdout_ids: List[int] = []

    for id_, make, model in zip(ids, makes, models):
        if make != cfg.holdout_make:
            continue
        if model == cfg.holdout_model:
            holdout_ids.append(id_)
        else:
            train_ids.append(id_)

    if len(holdout_ids) == 0:
        raise ValueError(
            f"No listings found for holdout model: {cfg.holdout_make} {cfg.holdout_model}"
        )

    rng = random.Random(cfg.seed)
    rng.shuffle(holdout_ids)

    mid = len(holdout_ids) // 2
    val_ids = holdout_ids[:mid]
    test_ids = holdout_ids[mid:]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }


def validate_split_fractions(cfg: SplitConfig) -> None:
    """
    Validate that train/val/test fractions are positive and sum to 1.0 (within tolerance).

    For holdout splits, fractions are ignored (always 50/50 val/test).

    Raises:
        ValueError: on invalid fractions.
    """
    if cfg.split_type == "holdout":
        return

    # Tolerance
    tol = 1e-8

    # Fractions from config
    train_frac = cfg.train_frac
    val_frac = cfg.val_frac
    test_frac = cfg.test_frac

    # Sum fractions
    sum = train_frac + val_frac + test_frac

    if abs(1-sum) > tol:
        raise ValueError(f"Invalid test/validation/train split fraction. Sum: {sum}")

    if (train_frac<0) or (val_frac<0) or (test_frac<0):
        raise ValueError("Fractions must be positive")
