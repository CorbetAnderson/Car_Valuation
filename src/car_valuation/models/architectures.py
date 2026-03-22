# src/car_valuation/models/architectures.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class CarValuationMLP(nn.Module):
    """
    MLP for car price regression.

    Inputs (all [batch, dim]):
        x_num             — numeric features (scaled)
        x_cat             — categorical features (ordinal or onehot)
        x_text_embedding  — CLIP text embedding (optional)
        x_image_embedding — CLIP image embedding (optional)

    Output: [batch, 1] — predicted log1p(price)

    Architecture:
        1. Optionally project text + image embeddings separately
        2. Concat all inputs → MLP trunk (Linear → BN → Act → Dropout per layer)
        3. Linear head → scalar output
    """

    def __init__(
        self,
        n_num: int,
        n_cat: int,
        use_embeddings: bool = True,
        emb_dims: Optional[Dict[str, int]] = None,
        emb_proj_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)

        self.use_embeddings = use_embeddings
        self.emb_proj_layers = nn.ModuleDict()

        in_dim = n_num + n_cat

        if use_embeddings and emb_dims:
            for name, dim in emb_dims.items():
                self.emb_proj_layers[name] = nn.Linear(dim, emb_proj_dim)
            in_dim += len(emb_dims) * emb_proj_dim

        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                act_cls(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, **kwargs) -> torch.Tensor:
        parts = [x_num, x_cat]

        if self.use_embeddings:
            for name, proj in self.emb_proj_layers.items():
                parts.append(F.relu(proj(kwargs[name])))

        x = torch.cat(parts, dim=-1)
        return self.mlp(x)


def build_model(
    cfg: Dict[str, Any],
    n_num: int,
    n_cat: int,
    use_embeddings: bool = True,
    emb_dims: Optional[Dict[str, int]] = None,
) -> CarValuationMLP:
    """
    Build a CarValuationMLP from train.yml config.

    Args:
        cfg:             Full train.yml config dict (top-level key "train").
        n_num:           Number of numeric input features.
        n_cat:           Number of categorical input features.
        use_embeddings:  Whether to include embedding projections.
        emb_dims:        Dict of embedding name → dimension (e.g. {"x_text_embedding": 1024}).

    Returns:
        CarValuationMLP instance.
    """
    model_cfg = cfg["train"]["model"]
    return CarValuationMLP(
        n_num=n_num,
        n_cat=n_cat,
        use_embeddings=use_embeddings,
        emb_dims=emb_dims,
        hidden_dims=model_cfg.get("hidden_dims", [512, 256, 128]),
        dropout=model_cfg.get("dropout", 0.2),
        activation=model_cfg.get("activation", "relu"),
    )
