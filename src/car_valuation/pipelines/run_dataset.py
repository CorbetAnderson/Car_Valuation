# src/car_valuation/pipelines/run_dataset.py
from __future__ import annotations

import argparse


def main() -> None:
    """
    Pipeline entrypoint: build the dataset artifact from DB.

    Responsibilities:
      - parse CLI arguments
      - call datasets.build_dataset.main
      - log resulting artifact path + row counts

    This file should stay thin; keep logic in datasets/build_dataset.py.
    """
    parser = argparse.ArgumentParser(description="Build dataset artifact from Supabase")
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of rows to fetch (default: 100000)"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset.yml",
        help="Path to dataset config YAML"
    )
    parser.add_argument(
        "--db-config",
        type=str,
        default="configs/db.yml",
        help="Path to database config YAML"
    )
    args = parser.parse_args()

    from ..datasets.build_dataset import main as build_main

    build_main(
        dataset_cfg_path=args.dataset_config,
        db_cfg_path=args.db_config,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
