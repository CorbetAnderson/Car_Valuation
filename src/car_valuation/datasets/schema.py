# src/car_valuation/datasets/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class FeatureSpec:
    """
    Defines the feature columns to use for model training.

    Keep this as the single source of truth for:
    - numeric columns (float/int)
    - categorical columns (strings / enums)
    - embedding columns (vectors)
    - metadata columns (kept for traceability but not used as features)
    """
    target_col: str = "price"
    id_col: str = "listing_id"
    time_col: str = "scraped_at"

    numeric_cols: Sequence[str] = field(default_factory=lambda: (
        "year", "odometer", "engine_size", "doors", "seats",
        "cylinders", "owners", "wof_months", "reg_months",
    ))

    categorical_cols: Sequence[str] = field(default_factory=lambda: (
        "make", "model", "body_style", "transmission", "fuel", "region",
    ))

    embedding_text_col: str = "text_embedding"
    embedding_image_col: str = "image_embedding"

    # Columns that should be retained in the dataset artifact for debugging/traceability
    metadata_cols: Sequence[str] = field(default_factory=lambda: (
        "listing_id", "scraped_at", "region", "suburb", "image_url_used", "embedded_at",
    ))


@dataclass
class DatasetSpec:
    """
    Describes a built dataset artifact.

    This is what you should write into artifacts/datasets/<version>/metadata.json
    so training/evaluation is reproducible and auditable.
    """
    name: str
    version: str
    source_tables: Dict[str, str]
    embedding_model: str
    feature_spec: FeatureSpec
    label_transform: str = "none"  # "none" | "log1p"
    require_embeddings: bool = True


def required_input_columns(spec: FeatureSpec) -> List[str]:
    """
    Return the full set of required columns expected in the *joined* dataframe
    before preprocessing/encoding.

    This is used to validate DB query output.
    """
    raise NotImplementedError


def validate_columns_present(df_columns: Sequence[str], required: Sequence[str]) -> None:
    """
    Validate that all required columns are present.

    Raises:
        ValueError: if any required columns are missing.
    """
    raise NotImplementedError
