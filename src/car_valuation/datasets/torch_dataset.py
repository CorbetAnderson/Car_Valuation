# src/car_valuation/datasets/torch_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
      - preprocessor.json
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
    raise NotImplementedError


def load_parquet_dataset(path: str) -> Any:
    """
    Load the parquet dataset file into memory.

    Returns:
        pandas DataFrame (recommended).
    """
    raise NotImplementedError


def load_splits(path: str) -> Dict[str, Sequence[int]]:
    """
    Load train/val/test splits mapping from splits.json.
    """
    raise NotImplementedError


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
        df: Any,
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
        raise NotImplementedError

    def __len__(self) -> int:
        """Return number of rows in the selected split."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single training example.

        Recommended return keys:
          - "x_num": float tensor [n_numeric]
          - "x_cat": long tensor [n_cat] (ordinal ids) OR float tensor (onehot)
          - "x_emb": float tensor [d_emb] (if present)
          - "y": float tensor [1]
        """
        raise NotImplementedError


def default_collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of examples into a batch.

    Only needed if you have any special handling; otherwise DataLoader's default works.
    """
    raise NotImplementedError


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
    raise NotImplementedError
