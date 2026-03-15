# src/car_valuation/datasets/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence



@dataclass(frozen=True)
class FeatureSpec:
    """
    Defines the feature columns to use for model training.

    Keep this as the single source of truth for:
    - numeric columns (float/int/bool)
    - categorical columns (strings / enums)
    - embedding columns (vectors)
    - metadata columns (kept for traceability but not used as features)
    - raw_required_cols: columns required from the DB join to *derive* features
    """
    target_col: str = "price"
    id_col: str = "listing_id"
    time_col: str = "scraped_at"

    # Structured numeric/bool features (include bools here; you'll cast to 0/1 later)
    numeric_cols: Sequence[str] = field(default_factory=lambda: (
        "year",
        "odometer",
        "engine_size",
        "doors",
        "seats",
        "cylinders",
        "owners",
        "wof_months",
        "reg_months",
        "is_new",
        "is_dealer",
        "is_4wd",
    ))

    # Structured categorical features
    categorical_cols: Sequence[str] = field(default_factory=lambda: (
        "make",
        "model",
        "body_style",
        "transmission",
        "fuel",
        "region",
        "suburb",
        "exterior_colour",
    ))

    embedding_text_col: str = "text_embedding"
    embedding_image_col: str = "image_embedding"

    # Columns required from DB to derive features, but not used directly as features
    raw_required_cols: Sequence[str] = field(default_factory=lambda: (
        "was_price",
    ))

    engineered_required_cols: Sequence[str] = field(default_factory=lambda: (
       "is_discounted",
    ))

    # Columns retained in dataset artifact for debugging/traceability
    metadata_cols: Sequence[str] = field(default_factory=lambda: (
        "listing_id",
        "scraped_at",
        "region",
        "suburb",
        "image_url_used",
        "embedded_at",
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
    label_transform: str = "none"
    require_embeddings: bool = True


def required_input_columns(spec: FeatureSpec) -> List[str]:
    """
    Return the full set of required columns expected in the *joined* dataframe
    before preprocessing/encoding.

    This is used to validate DB query output.
    """
    cols = [
        spec.target_col,
        *spec.numeric_cols,
        *spec.categorical_cols,
        spec.embedding_text_col,
        spec.embedding_image_col,
        *spec.raw_required_cols,
        *spec.metadata_cols,
    ]
    return list(dict.fromkeys(cols)) 


def validate_columns_present(df_columns: Sequence[str], required: Sequence[str]) -> None:
    """
    Validate that all required columns are present.

    Raises:
        ValueError: if any required columns are missing.
    """        
    required_set = set(required)
    have_set = set(df_columns)
    missing = sorted(required_set - have_set)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    
def validate_engineered_columns_present(df_columns: Sequence[str], spec: FeatureSpec) -> None:
    """
    Validate that engineered feature columns exist AFTER feature_engineering() runs.
    """
    missing = [c for c in spec.engineered_required_cols if c not in df_columns]
    if missing:
        raise ValueError(f"Missing engineered columns (post-feature-engineering): {missing}")
