from __future__ import annotations

from typing import Any

import torch
from hydra.utils import instantiate
from lightning import LightningModule
from monai.metrics import DiceMetric
from torch import nn

from src.losses.soft_cldice import SoftCLDiceLoss


class ValveSegmentationTask(LightningModule):
    def __init__(
        self,
        model_cfg: Any,
        loss_cfg: Any,
        optimizer_cfg: Any,
        scheduler_cfg: Any | None = None,
        threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model_cfg if isinstance(model_cfg, nn.Module) else instantiate(model_cfg)
        self.loss_fn = loss_cfg if isinstance(loss_cfg, nn.Module) else instantiate(loss_cfg)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.threshold = threshold
        self.topology_proxy = SoftCLDiceLoss(iter_=10, smooth=1.0, sigmoid=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        inputs = batch["image"]
        targets = batch["target"]
        mask = batch.get("mask")

        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets, mask=mask)

        probs = torch.sigmoid(outputs)
        pred_bin = (probs > self.threshold).float()
        target_bin = (targets > self.threshold).float()
        self.dice_metric(pred_bin, target_bin)
        dice = self.dice_metric.aggregate()
        self.dice_metric.reset()
        topology_score = 1.0 - self.topology_proxy(outputs, targets, mask=mask)

        self.log(f"{stage}/loss", loss, prog_bar=(stage == "val"), sync_dist=True, batch_size=inputs.shape[0])
        self.log(f"{stage}/dice", dice, sync_dist=True, batch_size=inputs.shape[0])
        self.log(f"{stage}/topology_proxy", topology_score, sync_dist=True, batch_size=inputs.shape[0])
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self) -> Any:
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        if self.scheduler_cfg is None:
            return optimizer

        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
