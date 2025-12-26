# src/car_valuation/pipelines/run_dataset.py
from __future__ import annotations


def main(dataset_cfg_path: str = "configs/dataset.yml", db_cfg_path: str = "configs/db.yml") -> None:
    """
    Pipeline entrypoint: build the dataset artifact from DB.

    Responsibilities:
      - load configs
      - call datasets.build_dataset.main / build_dataset
      - log resulting artifact path + row counts

    This file should stay thin; keep logic in datasets/build_dataset.py.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
