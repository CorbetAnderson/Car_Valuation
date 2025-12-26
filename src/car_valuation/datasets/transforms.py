# src/car_valuation/datasets/transforms.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass
class NumericImputer:
    """
    Stores fitted imputation values for numeric columns.

    Typical strategy: median per column computed on training split only.
    """
    strategy: str  # "median" | "mean" | "constant"
    fill_values: Dict[str, float]


@dataclass
class StandardScaler:
    """
    Stores fitted normalization parameters for numeric columns.

    Fit on training split only; apply to train/val/test.
    """
    means: Dict[str, float]
    stds: Dict[str, float]


@dataclass
class CategoricalEncoder:
    """
    Stores fitted categorical encoding maps.

    encoding:
      - "onehot": store category vocabulary; downstream produces sparse/dense vectors
      - "ordinal": store category -> int mapping
    """
    encoding: str  # "onehot" | "ordinal"
    vocab: Dict[str, List[str]]              # col -> list of categories (for onehot)
    mapping: Dict[str, Dict[str, int]]       # col -> category -> id (for ordinal)
    unknown_token: str = "UNKNOWN"


@dataclass
class Preprocessor:
    """
    Bundle of fitted preprocessing artifacts.

    This object should be serialized to artifacts/datasets/<version>/preprocessor.json
    (or pickle if you prefer, but JSON is more 'portfolio-friendly').
    """
    numeric_imputer: Optional[NumericImputer] = None
    scaler: Optional[StandardScaler] = None
    cat_encoder: Optional[CategoricalEncoder] = None


def cap_outliers(df: Any, caps: Mapping[str, Tuple[float, float]]) -> Any:
    """
    Cap numeric columns to [min, max] ranges.

    Args:
        df: dataframe-like (pandas recommended).
        caps: dict like {"odometer": (0, 400000), "engine_size": (500, 8000)}

    Returns:
        Updated df with capped values.
    """
    raise NotImplementedError


def add_missingness_flags(df: Any, cols: Sequence[str]) -> Any:
    """
    Add boolean indicator columns for missingness for each input col.

    Example:
        odometer -> odometer_is_missing (True/False)

    Returns:
        Updated df with added flag columns.
    """
    raise NotImplementedError


def fit_numeric_imputer(df_train: Any, cols: Sequence[str], strategy: str) -> NumericImputer:
    """
    Fit a numeric imputer on training data only.

    Returns:
        NumericImputer with per-column fill values.
    """
    raise NotImplementedError


def apply_numeric_imputer(df: Any, imputer: NumericImputer, cols: Sequence[str]) -> Any:
    """
    Apply fitted numeric imputer to any split (train/val/test).

    Returns:
        Updated df with missing values filled.
    """
    raise NotImplementedError


def fit_standard_scaler(df_train: Any, cols: Sequence[str]) -> StandardScaler:
    """
    Fit a standard scaler (mean/std) on training data only.
    """
    raise NotImplementedError


def apply_standard_scaler(df: Any, scaler: StandardScaler, cols: Sequence[str]) -> Any:
    """
    Apply fitted scaler to numeric columns.

    Notes:
        - Must handle std=0 safely (e.g., leave as 0 or skip).
    """
    raise NotImplementedError


def fill_categoricals(df: Any, cols: Sequence[str], value: str = "UNKNOWN") -> Any:
    """
    Fill null/empty categorical values with a sentinel token.
    """
    raise NotImplementedError


def fit_categorical_encoder(df_train: Any, cols: Sequence[str], encoding: str, unknown_token: str = "UNKNOWN") -> CategoricalEncoder:
    """
    Fit a categorical encoder (vocab/mapping) on training data only.

    Args:
        encoding: "onehot" or "ordinal"
    """
    raise NotImplementedError


def apply_categorical_encoder(df: Any, encoder: CategoricalEncoder, cols: Sequence[str]) -> Any:
    """
    Apply categorical encoding.

    For "ordinal":
        - Adds integer id columns (e.g., make_id)
    For "onehot":
        - Typically returns a matrix; you can also materialize onehot columns
          if you prefer transparency over performance.
    """
    raise NotImplementedError


def concat_embeddings(
    text_emb: Any,
    image_emb: Any,
    mode: str = "concat",
    fill_value: float = 0.0,
) -> Any:
    """
    Combine text and image embeddings into a single vector.

    Args:
        text_emb: vector or None
        image_emb: vector or None
        mode:
          - "concat": [text ; image]
          - "text_only": text
          - "image_only": image
        fill_value: used when one side is missing and you choose to keep rows.

    Returns:
        Combined embedding vector.
    """
    raise NotImplementedError


def fit_preprocessor(df_train: Any, numeric_cols: Sequence[str], categorical_cols: Sequence[str], cfg: Mapping[str, Any]) -> Preprocessor:
    """
    Fit all preprocessing artifacts on the training split only.

    Args:
        df_train: training dataframe
        numeric_cols: numeric feature columns
        categorical_cols: categorical feature columns
        cfg: preprocessing config blob (impute, scale, encode settings)

    Returns:
        Preprocessor bundle.
    """
    raise NotImplementedError


def apply_preprocessor(df: Any, pre: Preprocessor, numeric_cols: Sequence[str], categorical_cols: Sequence[str], cfg: Mapping[str, Any]) -> Any:
    """
    Apply fitted preprocessing to any split (train/val/test).

    Returns:
        Updated df (and/or encoded matrices depending on your implementation).
    """
    raise NotImplementedError


def save_preprocessor(pre: Preprocessor, path: str) -> None:
    """
    Serialize a Preprocessor to disk.

    Prefer JSON for portfolio readability unless your objects require pickling.
    """
    raise NotImplementedError


def load_preprocessor(path: str) -> Preprocessor:
    """
    Load a previously saved Preprocessor from disk.
    """
    raise NotImplementedError
