# src/car_valuation/models/predict.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def predict(model: nn.Module, batch: dict, device: str) -> np.ndarray:
    """
    Run inference on a single batch dict.

    Args:
        model:  Trained model in eval mode.
        batch:  Dict with keys x_num, x_cat, x_text_embedding, x_image_embedding.
        device: Device string.

    Returns:
        Predicted prices in NZD (reverse of log1p), shape [batch_size].
    """
    model.eval()
    with torch.no_grad():
        batch = {k: v.to(device) for k, v in batch.items() if k != "y"}
        log_preds = model(**batch)
        return torch.expm1(log_preds).squeeze(-1).cpu().numpy()


def predict_dataloader(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """
    Run inference over a full dataloader.

    Returns:
        All predicted prices in NZD, shape [n_samples].
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "y"}
            log_preds = model(**batch)
            all_preds.append(torch.expm1(log_preds).squeeze(-1).cpu())

    return torch.cat(all_preds).numpy()
