# src/car_valuation/datasets/build_dataset.py
from __future__ import annotations
import pandas as pd
import math

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from ..utils.config import load_yaml
from ..storage.queries import get_table_name
from .schema import DatasetSpec, FeatureSpec
from .transforms import Preprocessor


@dataclass(frozen=True)
class BuildDatasetConfig:
    """
    Configuration for building a dataset artifact from DB tables.

    Loaded from configs/dataset.yml. Stored as-is for flexibility.
    """
    # Basic metadata
    name: str
    version: str
    output_dir: str

    # Embedding config
    embedding_model: str
    require_embeddings: bool

    # Resolved table names (from db.yml)
    cars_table: str
    embeddings_table: str

    # Nested configs (stored as dicts for flexibility)
    label: Dict[str, Any]          # source_col, transform, output_col
    join: Dict[str, Any]           # cars_table_key, embeddings_table_key, embeddings_pick
    filters: Dict[str, Any]        # min_price, max_price, min_year, cap_outliers, etc.
    features: Dict[str, Any]       # id_col, time_col, numeric, categorical, embeddings, metadata
    preprocessing: Dict[str, Any]  # numeric_impute, numeric_scale, categorical_fill, etc.
    split: Dict[str, Any]          # type, time_col, train_frac, val_frac, test_frac, seed
    export: Dict[str, Any]         # format, include_raw_text, write_stats


def load_build_config(dataset_path: str, db_path: str) -> BuildDatasetConfig:
    """
    Load BuildDatasetConfig from dataset.yml and db.yml.

    Resolves table names from db.yml so they're directly accessible.
    """
    raw = load_yaml(dataset_path)
    cfg = raw["dataset"]

    db_cfg = load_yaml(db_path)

    # Resolve table names from db.yml
    cars_table = get_table_name(db_cfg, cfg["join"]["cars_table_key"])
    embeddings_table = get_table_name(db_cfg, cfg["join"]["embeddings_table_key"])

    return BuildDatasetConfig(
        name=cfg["name"],
        version=cfg["version"],
        output_dir=cfg.get("output_dir", "artifacts/datasets"),
        embedding_model=cfg["embedding_model"],
        require_embeddings=cfg.get("require_embeddings", True),
        cars_table=cars_table,
        embeddings_table=embeddings_table,
        label=cfg["label"],
        join=cfg["join"],
        filters=cfg["filters"],
        features=cfg["features"],
        preprocessing=cfg["preprocessing"],
        split=cfg["split"],
        export=cfg["export"],
    )


def flatten_embeddings(df: pd.DataFrame, embeddings_table: str) -> pd.DataFrame:
    """
    Expand the nested embeddings dict column into separate columns.

    Args:
        df: DataFrame with a nested embeddings column (e.g., 'CarEmbeddings')
        embeddings_table: Name of the embeddings table column to flatten

    Returns:
        DataFrame with embeddings fields as separate columns, original dict column removed.
    """
    if embeddings_table not in df.columns:
        return df

    # Extract the dict and convert to DataFrame
    embeddings_df = pd.json_normalize(df[embeddings_table])

    # Prefix columns to avoid conflicts (e.g., 'model' exists in both)
    embeddings_df.columns = [f"{col}" for col in embeddings_df.columns]

    # Drop the original nested column
    df = df.drop(columns=[embeddings_table])

    # Concatenate with original df
    df = pd.concat([df.reset_index(drop=True), embeddings_df.reset_index(drop=True)], axis=1)

    return df


def fetch_joined_rows(sb: Any, cfg: BuildDatasetConfig, limit: int) -> Any:
    """
    Fetch the joined dataset rows from Supabase/Postgres.

    Args:
        sb: SupabaseClient instance.
        cfg: BuildDatasetConfig
        limit: Max rows to fetch.

    Returns:
        A pandas dataframe with cars joined to embeddings.
    """
    # Build column list from config
    cols = (
        [cfg.features["id_col"]] +
        cfg.features["numeric"] +
        cfg.features["categorical"] +
        [cfg.label["source_col"]] +
        cfg.features.get("extra_cols", [])
    )

    embed_cols = [
        cfg.features["embeddings"]["text_col"],
        cfg.features["embeddings"]["image_col"]
    ]

    # Use !inner for INNER JOIN - only return cars WITH embeddings
    select_str = ", ".join(cols) + f", {cfg.embeddings_table}!inner(" + ", ".join(embed_cols) + ")"

    _PAGE = 500
    all_data: list = []
    offset = 0

    while True:
        end = offset + _PAGE - 1
        resp = (
            sb.client.table(cfg.cars_table)
            .select(select_str)
            .eq(f"{cfg.embeddings_table}.model", cfg.embedding_model)
            .range(offset, end)
            .execute()
        )
        page = getattr(resp, "data", None) or []
        all_data.extend(page)
        if len(page) < _PAGE or len(all_data) >= limit:
            break
        offset += _PAGE

    df = pd.DataFrame(all_data[:limit])

    # Flatten the nested embeddings dict
    df = flatten_embeddings(df, cfg.embeddings_table)

    return df


def fetch_joined_rows_make(sb: Any, cfg: BuildDatasetConfig, limit: int, make: str) -> Any:
    """
    Fetch the joined dataset rows from Supabase/Postgres by car make.

    Args:
        sb: SupabaseClient instance.
        cfg: BuildDatasetConfig
        limit: Max rows to fetch.
        make: Car make to filter by.

    Returns:
        A pandas dataframe with cars (filtered by make) joined to embeddings.
    """
    # Build column list from config
    cols = (
        [cfg.features["id_col"]] +
        cfg.features["numeric"] +
        cfg.features["categorical"] +
        [cfg.label["source_col"]] +
        cfg.features.get("extra_cols", [])
    )

    embed_cols = [
        cfg.features["embeddings"]["text_col"],
        cfg.features["embeddings"]["image_col"]
    ]

    # Use !inner for INNER JOIN - only return cars WITH embeddings
    select_str = ", ".join(cols) + f", {cfg.embeddings_table}!inner(" + ", ".join(embed_cols) + ")"

    _PAGE = 500
    all_data: list = []
    offset = 0

    while True:
        end = offset + _PAGE - 1
        resp = (
            sb.client.table(cfg.cars_table)
            .select(select_str)
            .eq("make", make)
            .eq(f"{cfg.embeddings_table}.model", cfg.embedding_model)
            .range(offset, end)
            .execute()
        )
        page = getattr(resp, "data", None) or []
        all_data.extend(page)
        if len(page) < _PAGE or len(all_data) >= limit:
            break
        offset += _PAGE

    df = pd.DataFrame(all_data[:limit])

    # Flatten the nested embeddings dict
    df = flatten_embeddings(df, cfg.embeddings_table)

    return df


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

    # Remove rows without a price or make
    df = df.dropna(subset=['price', 'make'])

    # Remove any rows with any cars that are too cheap or too expensive.
    df = df[(df['price'] > cfg.filters['min_price']) & (df['price'] < cfg.filters['max_price'])]

    # Remove any rows with engine size outside range (allow NaN through)
    engine_range = cfg.filters['cap_outliers']['engine_size']
    df = df[df['engine_size'].isna() | ((df['engine_size'] >= engine_range[0]) & (df['engine_size'] <= engine_range[1]))]

    # Remove any rows with too many doors (allow NaN through)
    df = df[df['doors'].isna() | (df['doors'] <= cfg.filters['max_doors'])]

    # Remove any rows with too many seats (allow NaN through)
    df = df[df['seats'].isna() | (df['seats'] <= cfg.filters['max_seats'])]

    # Remove odometer outliers (allow NaN through)
    odometer_range = cfg.filters['cap_outliers']['odometer']
    df = df[df['odometer'].isna() | ((df['odometer'] >= odometer_range[0]) & (df['odometer'] <= odometer_range[1]))]

    # Remove cars that were made in the future
    df = df[df['year'] <= cfg.filters['current_year']]

    # Remove cars with invalid wof or rego (allow NaN through)
    df = df[(df['wof_months'].isna() | (df['wof_months'] <= cfg.filters['longest_wof'])) &
            (df['reg_months'].isna() | (df['reg_months'] <= cfg.filters['longest_rego']))]

    # Remove cars with too many owners (allow NaN through)
    df = df[df['owners'].isna() | (df['owners'] <= cfg.filters['max_owners'])]

    if cfg.require_embeddings:
        embed_cols = ['text_embedding', 'image_embedding']
        missing_cols = [c for c in embed_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"require_embeddings=True but missing columns: {missing_cols}. "
                           "Check that embeddings exist in DB for the embedding_model.")
        df = df.dropna(subset=embed_cols)

    return df


def transform_label(df: Any, cfg: BuildDatasetConfig) -> Any:
    """
    Create the label column used for training.

    Examples:
      - if label_transform == "log1p": y = log1p(price)
      - else: y = price

    Returns:
        Updated df with a training label column (e.g., "y").
    """
    # Transform the label
    source_col = cfg.label["source_col"]
    df["y"] = df[source_col] if cfg.label["transform"] == "none" else df[source_col].apply(math.log1p)

    # Drop the source col
    df = df.drop(columns=[source_col])

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine columns and drop raw components after splitting.

    Called AFTER make_splits since splits need make/model for holdout.

    - make + model → make_model
    - suburb + region → location
    - Drop: make, model, suburb, region

    Returns:
        DataFrame with engineered features.
    """
    # Combine make + model
    df["make_model"] = df["make"].fillna("") + "_" + df["model"].fillna("")

    # Combine suburb + region
    df["location"] = df["suburb"].fillna("") + "_" + df["region"].fillna("")

    # Drop the raw columns
    df = df.drop(columns=["make", "model", "suburb", "region"])

    return df


def make_splits(df: Any, cfg: BuildDatasetConfig) -> Dict[str, Sequence[int]]:
    """
    Create train/val/test splits according to cfg.split.

    Returns:
        dict: {"train": [...], "val": [...], "test": [...]}
    """
    from .splits import random_split, holdout_split, SplitConfig

    # Create SplitConfig from cfg.split dict
    split_cfg = SplitConfig(**cfg.split)

    # Get listing IDs
    ids = df[cfg.features["id_col"]].tolist()

    if split_cfg.split_type == "random":
        return random_split(ids, split_cfg)
    elif split_cfg.split_type == "holdout":
        makes = df["make"].tolist()
        models = df["model"].tolist()
        return holdout_split(ids, makes, models, split_cfg)
    else:
        raise ValueError(f"Unknown split_type: {split_cfg.split_type}")


def fit_preprocessing(df_train: pd.DataFrame, cfg: BuildDatasetConfig) -> Preprocessor:
    """
    Fit preprocessing artifacts on training split only.

    Returns:
        Preprocessor bundle (imputer/scaler/encoder).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

    numeric_cols = cfg.features["numeric"]
    # After feature_engineering: make_model, location replace make, model, suburb, region
    categorical_cols = ["make_model", "location", "body_style", "transmission", "fuel"]

    # Handle all-NaN columns by filling with 0 before fitting
    numeric_data = df_train[numeric_cols].copy()
    all_nan_cols = numeric_data.columns[numeric_data.isna().all()].tolist()
    if all_nan_cols:
        numeric_data[all_nan_cols] = 0

    # Fit imputer
    imputer = SimpleImputer(strategy=cfg.preprocessing["numeric_impute"])
    imputer.fit(numeric_data)

    # Fit scaler only if configured
    scaler = None
    if cfg.preprocessing["numeric_scale"] == "standard":
        scaler = StandardScaler()
        scaler.fit(numeric_data)

    # Fit encoder based on config
    cat_data = df_train[categorical_cols].fillna(cfg.preprocessing["categorical_fill"])
    if cfg.preprocessing["categorical_encoding"] == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(cat_data)

    return Preprocessor(
        imputer=imputer,
        scaler=scaler,
        encoder=encoder,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def apply_preprocessing(df: pd.DataFrame, pre: Preprocessor, cfg: BuildDatasetConfig) -> pd.DataFrame:
    """
    Apply fitted preprocessing to the provided df (any split).

    Returns:
        Transformed df ready for export.
    """

    # Add missingness flags before imputation
    for col in cfg.preprocessing.get("add_missingness_flags", []):
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Handle all-NaN columns by filling with 0 before imputing
    numeric_data = df[pre.numeric_cols].copy()
    all_nan_cols = numeric_data.columns[numeric_data.isna().all()].tolist()
    if all_nan_cols:
        numeric_data[all_nan_cols] = 0

    # Apply imputer to numeric cols
    df[pre.numeric_cols] = pre.imputer.transform(numeric_data)

    # Apply scaler to numeric cols (if fitted)
    if pre.scaler is not None:
        df[pre.numeric_cols] = pre.scaler.transform(df[pre.numeric_cols])

    # Fill categorical NaN and apply encoder
    cat_data = df[pre.categorical_cols].fillna(cfg.preprocessing["categorical_fill"])
    encoded = pre.encoder.transform(cat_data)

    if cfg.preprocessing["categorical_encoding"] == "onehot":
        # OneHotEncoder outputs array with expanded columns
        encoded_cols = pre.encoder.get_feature_names_out(pre.categorical_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        df = df.drop(columns=pre.categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)
    else:
        # OrdinalEncoder keeps same columns
        df[pre.categorical_cols] = encoded

    # Convert binary columns to 0/1
    if "is_dealer" in df.columns:
        df["is_dealer"] = df["is_dealer"].astype(int)
    if "is_new" in df.columns:
        df["is_new"] = df["is_new"].astype(int)

    return df


def build_dataset_spec(cfg: BuildDatasetConfig) -> DatasetSpec:
    """
    Construct the DatasetSpec metadata object to persist alongside the dataset artifact.
    """
    feature_spec = FeatureSpec(
        target_col=cfg.label["source_col"],
        id_col=cfg.features["id_col"],
        time_col=cfg.features["time_col"],
        numeric_cols=tuple(cfg.features["numeric"]),
        categorical_cols=tuple(cfg.features["categorical"]),
        embedding_text_col=cfg.features["embeddings"]["text_col"],
        embedding_image_col=cfg.features["embeddings"]["image_col"],
        metadata_cols=tuple(cfg.features.get("metadata", [])),
    )

    return DatasetSpec(
        name=cfg.name,
        version=cfg.version,
        source_tables={"cars": cfg.cars_table, "embeddings": cfg.embeddings_table},
        embedding_model=cfg.embedding_model,
        feature_spec=feature_spec,
        label_transform=cfg.label["transform"],
        require_embeddings=cfg.require_embeddings,
    )


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
      - preprocessor.pkl

    Returns:
        Path to the artifact directory.
    """
    import json
    import pickle
    from pathlib import Path
    from dataclasses import asdict

    out_dir = Path(cfg.output_dir) / cfg.version
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save data as parquet
    df.to_parquet(out_dir / "data.parquet", index=False)

    # Save splits
    with open(out_dir / "splits.json", "w") as f:
        json.dump({k: list(v) for k, v in splits.items()}, f)

    # Save metadata (spec + config snapshot)
    metadata = {
        "spec": asdict(spec),
        "config": asdict(cfg),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save preprocessor (pickle for sklearn objects)
    with open(out_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)

    return str(out_dir)


def build_dataset(cfg: BuildDatasetConfig, sb: Any, limit: Optional[int] = None) -> str:
    """
    End-to-end dataset build:
      1) fetch join
      2) clean + transform label
      3) create splits (needs make/model for holdout)
      4) feature engineer
      5) fit preprocessing on train
      6) apply preprocessing to all
      7) save artifacts

    Args:
        cfg: Build configuration.
        sb: Supabase client.
        limit: Optional row limit for fetching (default: 100000).

    Returns:
        Artifact directory path.
    """
    fetch_limit = limit if limit is not None else 100000

    # 1. Fetch data
    print("Fetching data...")
    df = fetch_joined_rows(sb, cfg, limit=fetch_limit)
    print(f"  Fetched {len(df)} rows")

    # 2. Clean
    print("Cleaning rows...")
    df = clean_rows(df, cfg)
    print(f"  {len(df)} rows after cleaning")

    # 3. Transform label
    df = transform_label(df, cfg)

    # 4. Make splits (needs make/model columns for holdout)
    print("Creating splits...")
    splits = make_splits(df, cfg)
    print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # 5. Feature engineering (after splits since holdout needs make/model)
    df = feature_engineering(df)

    # 6. Get train subset for fitting
    train_ids = set(splits["train"])
    df_train = df[df["listing_id"].isin(train_ids)]

    # 7. Fit preprocessing on train
    print("Fitting preprocessing...")
    pre = fit_preprocessing(df_train, cfg)

    # 8. Apply preprocessing to all rows
    print("Applying preprocessing...")
    df = apply_preprocessing(df, pre, cfg)

    # 9. Build spec
    spec = build_dataset_spec(cfg)

    # 10. Save artifacts
    print("Saving artifacts...")
    artifact_path = save_dataset_artifact(df, splits, pre, spec, cfg)
    print(f"  Saved to: {artifact_path}")

    return artifact_path


def main(
    dataset_cfg_path: str = "configs/dataset.yml",
    db_cfg_path: str = "configs/db.yml",
    limit: Optional[int] = None,
) -> None:
    """
    CLI-style entrypoint for building the dataset artifact.

    This should be called by pipelines/run_dataset.py and can also be invoked directly:
        python -m car_valuation.datasets.build_dataset --config configs/dataset.yml

    Args:
        dataset_cfg_path: Path to dataset config YAML.
        db_cfg_path: Path to database config YAML.
        limit: Optional row limit for fetching data.
    """
    from ..storage.supabase_client import SupabaseClient
    from ..utils.config import find_project_root, load_env

    # Setup
    root = find_project_root()
    load_env(root)

    # Load configs
    cfg = load_build_config(str(root / dataset_cfg_path), str(root / db_cfg_path))
    print(f"Building dataset: {cfg.name} {cfg.version}")
    if limit:
        print(f"  (limited to {limit} rows)")

    # Create Supabase client
    db_cfg = load_yaml(str(root / db_cfg_path))
    sb = SupabaseClient(db_cfg)

    # Build dataset
    artifact_path = build_dataset(cfg, sb, limit=limit)
    print(f"\nDataset artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
