# src/car_valuation/datasets/transforms.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Preprocessor:
    """
    Bundle of fitted preprocessing artifacts.

    Fitted on training split only, then applied to all splits.
    """
    imputer: Any
    scaler: Any
    encoder: Any
    numeric_cols: List[str]
    categorical_cols: List[str]
