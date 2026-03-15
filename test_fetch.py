#!/usr/bin/env python3
"""Test building the dataset with a small sample."""

from src.car_valuation.datasets.build_dataset import (
    load_build_config,
    fetch_joined_rows,
    clean_rows,
    transform_label,
    make_splits,
    feature_engineering,
    fit_preprocessing,
    apply_preprocessing,
    build_dataset_spec,
    save_dataset_artifact,
)
from src.car_valuation.storage.supabase_client import SupabaseClient
from src.car_valuation.utils.config import load_env, find_project_root, load_yaml

# Setup
root = find_project_root()
load_env(root)

# Load configs
dataset_path = root / "configs/dataset.yml"
db_path = root / "configs/db.yml"

cfg = load_build_config(str(dataset_path), str(db_path))
print(f"Config loaded: {cfg.name} v{cfg.version}")

# Create client
db_cfg = load_yaml(str(db_path))
sb = SupabaseClient(db_cfg)

# 1. Fetch
print("\n1. Fetching 100 rows...")
df = fetch_joined_rows(sb, cfg, limit=100)
print(f"   Fetched {len(df)} rows")
print(f"   Columns: {df.columns.tolist()}")

# 2. Clean
print("\n2. Cleaning rows...")
df = clean_rows(df, cfg)
print(f"   {len(df)} rows after cleaning")

# 3. Transform label
print("\n3. Transforming label...")
df = transform_label(df, cfg)
print(f"   'y' column added, range: {df['y'].min():.2f} - {df['y'].max():.2f}")

# 4. Make splits
print("\n4. Creating splits...")
splits = make_splits(df, cfg)
print(f"   Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

# 5. Feature engineering
print("\n5. Feature engineering...")
df = feature_engineering(df)
print(f"   New columns: make_model, location")
print(f"   Columns now: {df.columns.tolist()}")

# 6. Fit preprocessing on train
print("\n6. Fitting preprocessing on train split...")
train_ids = set(splits["train"])
df_train = df[df["listing_id"].isin(train_ids)]
pre = fit_preprocessing(df_train, cfg)
print(f"   Imputer fitted on {len(pre.numeric_cols)} numeric cols")
print(f"   Encoder fitted on {len(pre.categorical_cols)} categorical cols")

# 7. Apply preprocessing
print("\n7. Applying preprocessing...")
df = apply_preprocessing(df, pre, cfg)
print(f"   Final shape: {df.shape}")
print(f"   Final columns: {df.columns.tolist()}")

# 8. Build spec
print("\n8. Building dataset spec...")
spec = build_dataset_spec(cfg)
print(f"   Spec: {spec.name} v{spec.version}")

# 9. Save artifact
print("\n9. Saving artifact...")
artifact_path = save_dataset_artifact(df, splits, pre, spec, cfg)
print(f"   Saved to: {artifact_path}")

print("\nDone!")
