# src/car_valuation/pipelines/run_train.py
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path


def _detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train car valuation model")
    parser.add_argument("--train", default="configs/train.yml", help="Path to train.yml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    from ..utils.config import load_yaml, find_project_root, load_env
    from ..datasets.torch_dataset import get_dataloaders
    from ..models.architectures import build_model
    from ..models.train import Trainer

    root = find_project_root()
    load_env(root)

    cfg = load_yaml(str(root / args.train))
    ds_cfg = cfg["train"]["dataset"]
    dl_cfg = cfg["train"]["dataloader"]
    out_cfg = cfg["train"]["output"]

    device = _detect_device()
    logger.info(f"Using device: {device}")

    # Embeddings toggle
    use_embeddings = ds_cfg.get("use_embeddings", True)
    embedding_cols = ds_cfg.get("embedding_cols") if use_embeddings else None
    if not use_embeddings:
        logger.info("Embeddings disabled")

    # Build dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        artifact_dir=str(root / ds_cfg["artifact_dir"]),
        numeric_cols=ds_cfg["numeric_cols"],
        categorical_cols=ds_cfg["categorical_cols"],
        embedding_cols=embedding_cols,
        target_col=ds_cfg.get("target_col", "y"),
        batch_size=dl_cfg.get("batch_size", 256),
        num_workers=dl_cfg.get("num_workers", 0),
    )

    # Infer input dims from first batch
    sample = next(iter(train_loader))
    n_num = sample["x_num"].shape[-1]
    n_cat = sample["x_cat"].shape[-1]
    emb_dims = {
        k: v.shape[-1] for k, v in sample.items()
        if k.startswith("x_") and k not in ("x_num", "x_cat")
    }
    del sample
    dims_str = f"numeric: {n_num}, categorical: {n_cat}"
    for name, dim in emb_dims.items():
        dims_str += f", {name}: {dim}"
    logger.info(f"Input dims — {dims_str}")

    # Resolve run directory
    runs_dir = str(root / out_cfg.get("runs_dir", "artifacts/runs"))
    if args.resume:
        # Reuse the existing run dir: .../run_xxx/checkpoints/epoch=001.pt → .../run_xxx
        run_dir = str(Path(args.resume).parent.parent)
        logger.info(f"Resuming run: {run_dir}")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = str(Path(runs_dir) / f"run_{stamp}")
    checkpoint_dir = str(Path(run_dir) / "checkpoints")
    tensorboard_dir = str(Path(run_dir) / "tensorboard")

    model = build_model(
        cfg,
        n_num=n_num,
        n_cat=n_cat,
        use_embeddings=use_embeddings,
        emb_dims=emb_dims if use_embeddings else None,
    )
    logger.info(f"Model: {model.__class__.__name__}")

    trainer = Trainer(
        model, train_loader, val_loader, cfg, device,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
        resume=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
