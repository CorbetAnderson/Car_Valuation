# src/car_valuation/datasets/torch_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class DatasetArtifactPaths:
    """
    Paths to a dataset artifact on disk.

    Expecting the structure created by datasets/build_dataset.py:
      - data.parquet
      - splits.json
      - metadata.json
      - preprocessor.pkl
    """
    artifact_dir: str
    data_path: str
    splits_path: str
    metadata_path: str
    preprocessor_path: str


def resolve_artifact_paths(artifact_dir: str) -> DatasetArtifactPaths:
    """
    Resolve and validate expected files in an artifact directory.

    Raises:
        FileNotFoundError: if any required file is missing.
    """
    base = Path(artifact_dir)

    data_path = base / "data.parquet"
    splits_path = base / "splits.json"
    metadata_path = base / "metadata.json"
    preprocessor_path = base / "preprocessor.pkl"

    for p in [data_path, splits_path, metadata_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    return DatasetArtifactPaths(
        artifact_dir=str(base),
        data_path=str(data_path),
        splits_path=str(splits_path),
        metadata_path=str(metadata_path),
        preprocessor_path=str(preprocessor_path),
    )


def load_parquet_dataset(path: str) -> pd.DataFrame:
    """
    Load the parquet dataset file into memory.

    Returns:
        pandas DataFrame.
    """
    return pd.read_parquet(path)


def load_splits(path: str) -> Dict[str, List[int]]:
    """
    Load train/val/test splits mapping from splits.json.
    """
    with open(path, "r") as f:
        return json.load(f)


class CarPriceDataset(Dataset):
    """
    Torch Dataset for car valuation.

    Expects a built dataset artifact where preprocessing has already been applied
    (preferred), so this class mostly:
      - selects a split
      - converts feature columns to tensors
      - returns consistent shapes
    """

    def __init__(
        self,
        df: pd.DataFrame,
        splits: Mapping[str, Sequence[int]],
        split: str,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
        embedding_col: Optional[str],
        target_col: str = "y",
    ) -> None:
        """
        Args:
            df: dataset dataframe (ideally already cleaned/encoded).
            splits: dict of split -> listing_id list.
            split: "train" | "val" | "test"
            numeric_cols: numeric feature column names.
            categorical_cols: categorical feature column names (if ordinal encoded, these are *_id).
            embedding_col: column containing the combined embedding vector, if used.
            target_col: label column to predict (e.g., "y").
        """
        if split not in splits:
            raise ValueError(f"Unknown split: {split}. Expected one of {list(splits.keys())}")

        # Filter to this split's IDs
        split_ids = set(splits[split])
        self.df = df[df["listing_id"].isin(split_ids)].reset_index(drop=True)

        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.embedding_col = embedding_col
        self.target_col = target_col

        # Filter to columns that actually exist
        self.numeric_cols = [c for c in self.numeric_cols if c in self.df.columns]
        self.categorical_cols = [c for c in self.categorical_cols if c in self.df.columns]

        # Pre-convert to numpy for faster access
        if self.numeric_cols:
            self.x_num = self.df[self.numeric_cols].values.astype(np.float32)
        else:
            self.x_num = np.zeros((len(self.df), 0), dtype=np.float32)

        if self.categorical_cols:
            self.x_cat = self.df[self.categorical_cols].values.astype(np.float32)
        else:
            self.x_cat = np.zeros((len(self.df), 0), dtype=np.float32)

        if self.embedding_col and self.embedding_col in self.df.columns:
            # Embeddings are stored as strings/lists - parse them
            emb_data = self.df[self.embedding_col].tolist()
            if emb_data and isinstance(emb_data[0], str):
                emb_data = [json.loads(e) if e else [] for e in emb_data]
            self.x_emb = np.array(emb_data, dtype=np.float32)
        else:
            self.x_emb = None

        self.y = self.df[self.target_col].values.astype(np.float32)

    def __len__(self) -> int:
        """Return number of rows in the selected split."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single training example.

        Recommended return keys:
          - "x_num": float tensor [n_numeric]
          - "x_cat": float tensor [n_cat]
          - "x_emb": float tensor [d_emb] (if present)
          - "y": float tensor [1]
        """
        result: Dict[str, torch.Tensor] = {
            "x_num": torch.from_numpy(self.x_num[idx]),
            "x_cat": torch.from_numpy(self.x_cat[idx]),
            "y": torch.tensor([self.y[idx]], dtype=torch.float32),
        }

        if self.x_emb is not None:
            result["x_emb"] = torch.from_numpy(self.x_emb[idx])

        return result


def default_collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of examples into a batch.

    Only needed if you have any special handling; otherwise DataLoader's default works.
    """
    result: Dict[str, torch.Tensor] = {}

    keys = batch[0].keys()
    for key in keys:
        result[key] = torch.stack([item[key] for item in batch])

    return result


def get_dataloaders(
    artifact_dir: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    embedding_col: Optional[str],
    target_col: str = "y",
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience helper to produce train/val/test dataloaders from an artifact dir.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    paths = resolve_artifact_paths(artifact_dir)
    df = load_parquet_dataset(paths.data_path)
    splits = load_splits(paths.splits_path)

    train_ds = CarPriceDataset(
        df, splits, "train", numeric_cols, categorical_cols, embedding_col, target_col
    )
    val_ds = CarPriceDataset(
        df, splits, "val", numeric_cols, categorical_cols, embedding_col, target_col
    )
    test_ds = CarPriceDataset(
        df, splits, "test", numeric_cols, categorical_cols, embedding_col, target_col
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
