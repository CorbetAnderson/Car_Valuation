# tests/test_torch_dataset.py
"""Tests for PyTorch dataset and dataloader utilities."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.car_valuation.datasets.torch_dataset import (
    CarPriceDataset,
    DatasetArtifactPaths,
    default_collate,
    get_dataloaders,
    load_parquet_dataset,
    load_splits,
    resolve_artifact_paths,
)


class TestResolveArtifactPaths:
    """Tests for resolve_artifact_paths."""

    def test_valid_artifact_dir(self, tmp_path):
        (tmp_path / "data.parquet").touch()
        (tmp_path / "splits.json").touch()
        (tmp_path / "metadata.json").touch()

        paths = resolve_artifact_paths(str(tmp_path))

        assert paths.artifact_dir == str(tmp_path)
        assert paths.data_path == str(tmp_path / "data.parquet")
        assert paths.splits_path == str(tmp_path / "splits.json")
        assert paths.metadata_path == str(tmp_path / "metadata.json")
        assert paths.preprocessor_path == str(tmp_path / "preprocessor.pkl")

    def test_missing_data_parquet_raises(self, tmp_path):
        (tmp_path / "splits.json").touch()
        (tmp_path / "metadata.json").touch()

        with pytest.raises(FileNotFoundError, match="data.parquet"):
            resolve_artifact_paths(str(tmp_path))

    def test_missing_splits_json_raises(self, tmp_path):
        (tmp_path / "data.parquet").touch()
        (tmp_path / "metadata.json").touch()

        with pytest.raises(FileNotFoundError, match="splits.json"):
            resolve_artifact_paths(str(tmp_path))

    def test_missing_metadata_json_raises(self, tmp_path):
        (tmp_path / "data.parquet").touch()
        (tmp_path / "splits.json").touch()

        with pytest.raises(FileNotFoundError, match="metadata.json"):
            resolve_artifact_paths(str(tmp_path))


class TestLoadParquetDataset:
    """Tests for load_parquet_dataset."""

    def test_loads_parquet_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        loaded = load_parquet_dataset(str(path))
        assert len(loaded) == 3
        assert list(loaded.columns) == ["a", "b"]


class TestLoadSplits:
    """Tests for load_splits."""

    def test_loads_splits_json(self, tmp_path):
        splits = {"train": [1, 2, 3], "val": [4], "test": [5]}
        path = tmp_path / "splits.json"
        path.write_text(json.dumps(splits))

        loaded = load_splits(str(path))
        assert loaded["train"] == [1, 2, 3]
        assert loaded["val"] == [4]
        assert loaded["test"] == [5]


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a sample dataset artifact for testing."""
    df = pd.DataFrame({
        "listing_id": [1, 2, 3, 4, 5],
        "price_num": [10000.0, 20000.0, 15000.0, 25000.0, 30000.0],
        "odometer_num": [50000.0, 30000.0, 80000.0, 20000.0, 10000.0],
        "make_id": [1.0, 2.0, 1.0, 3.0, 2.0],
        "model_id": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [10.0, 11.0, 9.5, 12.0, 12.5],
    })

    splits = {
        "train": [1, 2, 3],
        "val": [4],
        "test": [5],
    }

    df.to_parquet(tmp_path / "data.parquet")
    (tmp_path / "splits.json").write_text(json.dumps(splits))
    (tmp_path / "metadata.json").write_text(json.dumps({"name": "test", "version": "v1"}))

    return tmp_path, df, splits


class TestCarPriceDataset:
    """Tests for CarPriceDataset."""

    def test_filters_to_split(self, sample_dataset):
        tmp_path, df, splits = sample_dataset

        train_ds = CarPriceDataset(
            df=df,
            splits=splits,
            split="train",
            numeric_cols=["price_num", "odometer_num"],
            categorical_cols=["make_id", "model_id"],
            embedding_col=None,
        )

        assert len(train_ds) == 3

    def test_invalid_split_raises(self, sample_dataset):
        tmp_path, df, splits = sample_dataset

        with pytest.raises(ValueError, match="Unknown split"):
            CarPriceDataset(
                df=df,
                splits=splits,
                split="invalid",
                numeric_cols=[],
                categorical_cols=[],
                embedding_col=None,
            )

    def test_getitem_returns_tensors(self, sample_dataset):
        tmp_path, df, splits = sample_dataset

        ds = CarPriceDataset(
            df=df,
            splits=splits,
            split="train",
            numeric_cols=["price_num", "odometer_num"],
            categorical_cols=["make_id", "model_id"],
            embedding_col=None,
        )

        item = ds[0]

        assert "x_num" in item
        assert "x_cat" in item
        assert "y" in item
        assert item["x_num"].shape == (2,)
        assert item["x_cat"].shape == (2,)
        assert item["y"].shape == (1,)

    def test_handles_missing_columns(self, sample_dataset):
        tmp_path, df, splits = sample_dataset

        ds = CarPriceDataset(
            df=df,
            splits=splits,
            split="train",
            numeric_cols=["price_num", "nonexistent_col"],
            categorical_cols=["make_id"],
            embedding_col=None,
        )

        item = ds[0]
        assert item["x_num"].shape == (1,)
        assert item["x_cat"].shape == (1,)

    def test_handles_no_numeric_cols(self, sample_dataset):
        tmp_path, df, splits = sample_dataset

        ds = CarPriceDataset(
            df=df,
            splits=splits,
            split="train",
            numeric_cols=[],
            categorical_cols=["make_id"],
            embedding_col=None,
        )

        item = ds[0]
        assert item["x_num"].shape == (0,)

    def test_handles_no_categorical_cols(self, sample_dataset):
        tmp_path, df, splits = sample_dataset

        ds = CarPriceDataset(
            df=df,
            splits=splits,
            split="train",
            numeric_cols=["price_num"],
            categorical_cols=[],
            embedding_col=None,
        )

        item = ds[0]
        assert item["x_cat"].shape == (0,)

    def test_embedding_column(self, tmp_path):
        df = pd.DataFrame({
            "listing_id": [1, 2],
            "y": [10.0, 11.0],
            "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        })
        splits = {"train": [1, 2], "val": [], "test": []}

        ds = CarPriceDataset(
            df=df,
            splits=splits,
            split="train",
            numeric_cols=[],
            categorical_cols=[],
            embedding_col="embedding",
        )

        item = ds[0]
        assert "x_emb" in item
        assert item["x_emb"].shape == (3,)


class TestDefaultCollate:
    """Tests for default_collate."""

    def test_stacks_batch_items(self):
        import torch

        batch = [
            {"x_num": torch.tensor([1.0, 2.0]), "y": torch.tensor([10.0])},
            {"x_num": torch.tensor([3.0, 4.0]), "y": torch.tensor([20.0])},
        ]

        result = default_collate(batch)

        assert result["x_num"].shape == (2, 2)
        assert result["y"].shape == (2, 1)


class TestGetDataloaders:
    """Tests for get_dataloaders convenience function."""

    def test_creates_three_loaders(self, sample_dataset):
        tmp_path, df, splits = sample_dataset

        train_loader, val_loader, test_loader = get_dataloaders(
            artifact_dir=str(tmp_path),
            numeric_cols=["price_num"],
            categorical_cols=["make_id"],
            embedding_col=None,
            batch_size=2,
        )

        assert len(train_loader.dataset) == 3
        assert len(val_loader.dataset) == 1
        assert len(test_loader.dataset) == 1

    def test_batch_iteration(self, sample_dataset):
        tmp_path, df, splits = sample_dataset

        train_loader, _, _ = get_dataloaders(
            artifact_dir=str(tmp_path),
            numeric_cols=["price_num"],
            categorical_cols=["make_id"],
            embedding_col=None,
            batch_size=2,
        )

        batches = list(train_loader)
        assert len(batches) == 2

        assert "x_num" in batches[0]
        assert "x_cat" in batches[0]
        assert "y" in batches[0]
