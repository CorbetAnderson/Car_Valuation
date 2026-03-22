# src/car_valuation/models/train.py
from __future__ import annotations

import logging
import math
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from .checkpoints import save_checkpoint, get_checkpoint_path

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop for car valuation regression.

    Inspired by dinozaur's Trainer pattern. Handles:
      - AdamW optimiser + CosineAnnealingLR
      - MSE loss on log1p(price)
      - Per-epoch logging of train/val loss + MAE/RMSE in NZD
      - Early stopping
      - Best-model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        device: str,
        checkpoint_dir: str,
        tensorboard_dir: str,
        resume: str | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        train_cfg = cfg["train"]["training"]
        out_cfg = cfg["train"]["output"]

        self.num_epochs: int = train_cfg["epochs"]
        self.checkpoint_dir: str = checkpoint_dir
        self.save_best_only: bool = out_cfg.get("save_best_only", True)

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        # Scheduler — cosine annealing over all epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6,
        )

        self.criterion = nn.MSELoss()

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.es_patience: int = es_cfg.get("patience", 10)
        self._es_counter: int = 0
        self._best_rmse: float = math.inf
        self._best_epoch: int = 0
        self._start_epoch: int = 1

        # TensorBoard
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Resume from checkpoint if provided
        if resume:
            from .checkpoints import load_checkpoint
            ckpt = load_checkpoint(resume, self.model, self.optimizer)
            self._start_epoch = ckpt["epoch"] + 1
            self._best_rmse = ckpt["val_rmse"]
            # Fast-forward scheduler to match resumed epoch
            for _ in range(ckpt["epoch"]):
                self.scheduler.step()
            logger.info(f"Resuming from epoch {ckpt['epoch']} (val_rmse=NZD${ckpt['val_rmse']:.0f})")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Full training loop."""
        logger.info(f"Training {self.model.__class__.__name__} for {self.num_epochs} epochs on {self.device}")
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {n_params:,}")

        for epoch in range(self._start_epoch, self.num_epochs + 1):
            train_loss = self.train_one_epoch()
            val_metrics = self.eval_one_epoch()

            self.scheduler.step()

            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("Metrics/val_mae_nzd", val_metrics["val_mae"], epoch)
            self.writer.add_scalar("Metrics/val_rmse_nzd", val_metrics["val_rmse"], epoch)
            self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            logger.info(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_mae=NZD${val_metrics['val_mae']:.0f} | "
                f"val_rmse=NZD${val_metrics['val_rmse']:.0f}"
            )

            improved = val_metrics["val_rmse"] < self._best_rmse
            if improved:
                self._best_rmse = val_metrics["val_rmse"]
                self._best_epoch = epoch
                self._es_counter = 0
                ckpt_path = get_checkpoint_path(self.checkpoint_dir, epoch, val_metrics["val_rmse"])
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics["val_rmse"], ckpt_path)
            else:
                self._es_counter += 1
                if not self.save_best_only:
                    ckpt_path = get_checkpoint_path(self.checkpoint_dir, epoch, val_metrics["val_rmse"])
                    save_checkpoint(self.model, self.optimizer, epoch, val_metrics["val_rmse"], ckpt_path)

            if self._es_counter >= self.es_patience:
                logger.info(f"Early stopping at epoch {epoch} (best was epoch {self._best_epoch})")
                break

        self.writer.close()
        logger.info(f"Training finished. Best val_rmse=NZD${self._best_rmse:.0f} at epoch {self._best_epoch}")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def train_one_epoch(self) -> float:
        """One pass over the training dataloader. Returns avg loss."""
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(self.train_loader, desc="train", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("y")

            self.optimizer.zero_grad()
            preds = self.model(**batch)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def eval_one_epoch(self) -> Dict[str, float]:
        """One pass over the val dataloader. Returns loss + NZD MAE/RMSE."""
        self.model.eval()
        running_loss = 0.0
        all_preds: list = []
        all_labels: list = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="val  ", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("y")

                preds = self.model(**batch)
                loss = self.criterion(preds, labels)
                running_loss += loss.item()

                # Reverse log1p → NZD for interpretable metrics
                all_preds.append(torch.expm1(preds).cpu())
                all_labels.append(torch.expm1(labels).cpu())

        preds_nzd = torch.cat(all_preds)
        labels_nzd = torch.cat(all_labels)
        mae = (preds_nzd - labels_nzd).abs().mean().item()
        rmse = ((preds_nzd - labels_nzd) ** 2).mean().sqrt().item()

        return {
            "val_loss": running_loss / len(self.val_loader),
            "val_mae": mae,
            "val_rmse": rmse,
        }
