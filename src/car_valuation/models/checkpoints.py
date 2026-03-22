# src/car_valuation/models/checkpoints.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_rmse: float,
    path: str,
) -> None:
    """
    Save model + optimizer state to disk.

    Args:
        model:     Model to checkpoint.
        optimizer: Optimizer state to save.
        epoch:     Current epoch number.
        val_rmse:  Validation RMSE (NZD) at this checkpoint.
        path:      Full file path to write (.pt).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_rmse": val_rmse,
        },
        path,
    )
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint into model (and optionally optimizer).

    Args:
        path:      Path to checkpoint file.
        model:     Model to load weights into.
        optimizer: If provided, restore optimizer state too.

    Returns:
        Full checkpoint dict (includes epoch, val_rmse).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Support old checkpoints that used "val_loss" instead of "val_rmse"
    if "val_rmse" not in ckpt and "val_loss" in ckpt:
        ckpt.pop("val_loss")
        ckpt["val_rmse"] = float("inf")  # Incompatible metric — reset best tracking
    logger.info(f"Checkpoint loaded: {path} (epoch={ckpt.get('epoch')}, val_rmse=NZD${ckpt.get('val_rmse'):.0f})")
    return ckpt


def get_checkpoint_path(checkpoint_dir: str, epoch: int, val_rmse: float) -> str:
    """
    Build a checkpoint filename.

    Example: artifacts/checkpoints/epoch=005_rmse=12450.pt
    """
    return str(Path(checkpoint_dir) / f"epoch={epoch:03d}_rmse={val_rmse:.0f}.pt")
