from __future__ import annotations

import torch
from monai.losses import SoftclDiceLoss
from torch import nn


class SoftCLDiceLoss(nn.Module):
    """Thin adapter around MONAI's SoftclDiceLoss.

    We only keep project glue here: optional sigmoid and optional ROI masking.
    """

    def __init__(self, iter_: int = 3, smooth: float = 1.0, sigmoid: bool = True) -> None:
        super().__init__()
        self.sigmoid = sigmoid
        self.loss = SoftclDiceLoss(iter_=iter_, smooth=smooth)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.sigmoid:
            preds = torch.sigmoid(preds)
        if mask is not None:
            preds = preds * mask
            targets = targets * mask
        return self.loss(preds, targets)
