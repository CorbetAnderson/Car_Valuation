# src/car_valuation/datasets/build_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .schema import DatasetSpec, FeatureSpec
from .splits import SplitConfig
from .transforms import Preprocessor


@dataclass(frozen=True)
class BuildDatasetConfig:
    """
    Configuration for building a dataset artifact from DB tables.

    This should be loaded from configs/dataset.yml and then written back into
    artifacts/datasets/<version>/metadata.json for reproducibility.
    """
    dataset_name: str
    dataset_version: str

    embedding_model: str
    require_embeddings: bool = True

    label_col: str = "price"
    label_transform: str = "log1p"  # "none" | "log1p"

    feature_spec: FeatureSpec = FeatureSpec()
    split: SplitConfig = SplitConfig(split_type="time")

    preprocessing: Dict[str, Any] = None  # impute/scale/encode/caps
    output_dir: str = "artifacts/datasets"


def load_build_config(path: str) -> BuildDatasetConfig:
    """
    Load BuildDatasetConfig from YAML.

    Notes:
        - You already have utils/config.py; call into that if it supports YAML.
        - Consider env overrides for output_dir if you want.
    """
    raise NotImplementedError


def fetch_joined_rows(db_cfg: Mapping[str, Any], embedding_model: str) -> Any:
    """
    Fetch the joined dataset rows from Supabase/Postgres.

    Must:
        - Select from Cars
        - Join latest CarEmbeddings per (listing_id, model) using created_at DESC
        - Filter embeddings model == embedding_model

    Returns:
        A dataframe-like object (pandas recommended).
    """
    raise NotImplementedError


def clean_rows(df: Any, cfg: BuildDatasetConfig) -> Any:
    """
    Apply row-level cleaning rules:
      - drop invalid price/year/odometer ranges
      - standardize types
      - optionally cap outliers
      - enforce require_embeddings if cfg.require_embeddings is True

    Returns:
        Cleaned df.
    """
    raise NotImplementedError


def feature_engineering(df: Any, cfg: BuildDatasetConfig) -> Any:
    """
    Add derived columns used for training:
      - age = current_year - year
      - has_was_price, discount, discount_pct
      - has_text_emb, has_image_emb
      - any other cheap, useful signals

    Returns:
        Updated df.
    """
    raise NotImplementedError


def transform_label(df: Any, cfg: BuildDatasetConfig) -> Any:
    """
    Create the label column used for training.

    Examples:
      - if label_transform == "log1p": y = log1p(price)
      - else: y = price

    Returns:
        Updated df with a training label column (e.g., "y").
    """
    raise NotImplementedError


def make_splits(df: Any, cfg: BuildDatasetConfig) -> Dict[str, Sequence[int]]:
    """
    Create train/val/test splits according to cfg.split.

    Returns:
        dict: {"train": [...], "val": [...], "test": [...]}
    """
    raise NotImplementedError


def fit_preprocessing(df_train: Any, cfg: BuildDatasetConfig) -> Preprocessor:
    """
    Fit preprocessing artifacts on training split only.

    Returns:
        Preprocessor bundle (imputer/scaler/encoder).
    """
    raise NotImplementedError


def apply_preprocessing(df: Any, pre: Preprocessor, cfg: BuildDatasetConfig) -> Any:
    """
    Apply fitted preprocessing to the provided df (any split).

    Returns:
        Transformed df ready for export.
    """
    raise NotImplementedError


def build_dataset_spec(cfg: BuildDatasetConfig) -> DatasetSpec:
    """
    Construct the DatasetSpec metadata object to persist alongside the dataset artifact.
    """
    raise NotImplementedError


def save_dataset_artifact(
    df: Any,
    splits: Mapping[str, Sequence[int]],
    pre: Preprocessor,
    spec: DatasetSpec,
    cfg: BuildDatasetConfig,
) -> str:
    """
    Save the dataset artifact to disk.

    Should write:
      - data.parquet
      - splits.json
      - metadata.json (DatasetSpec + BuildDatasetConfig snapshot)
      - preprocessor.json
      - schema.json
      - stats.json / stats.md (optional but recommended)

    Returns:
        Path to the artifact directory.
    """
    raise NotImplementedError


def build_dataset(cfg: BuildDatasetConfig, db_cfg: Mapping[str, Any]) -> str:
    """
    End-to-end dataset build:
      1) fetch join
      2) clean + feature engineer
      3) create splits
      4) fit preprocessing on train
      5) apply preprocessing
      6) save artifacts

    Returns:
        Artifact directory path.
    """
    raise NotImplementedError


def main(dataset_cfg_path: str = "configs/dataset.yml", db_cfg_path: str = "configs/db.yml") -> None:
    """
    CLI-style entrypoint for building the dataset artifact.

    This should be called by pipelines/run_dataset.py and can also be invoked directly:
        python -m car_valuation.datasets.build_dataset --config configs/dataset.yml
    """
    raise NotImplementedError
