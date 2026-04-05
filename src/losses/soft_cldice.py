from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations: int = 10, smooth: float = 1.0) -> None:
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def _soft_erode(self, x: torch.Tensor) -> torch.Tensor:
        p1 = -F.max_pool3d(-x, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        p2 = -F.max_pool3d(-x, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
        p3 = -F.max_pool3d(-x, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
        return torch.minimum(torch.minimum(p1, p2), p3)

    def _soft_dilate(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)

    def _soft_open(self, x: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(x))

    def _soft_skeletonize(self, x: torch.Tensor) -> torch.Tensor:
        opened = self._soft_open(x)
        skeleton = torch.relu(x - opened)
        current = x
        for _ in range(self.iterations):
            current = self._soft_erode(current)
            opened = self._soft_open(current)
            skeleton = skeleton + torch.relu(current - opened)
        return skeleton

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preds = torch.sigmoid(preds)
        if mask is not None:
            preds = preds * mask
            targets = targets * mask

        pred_skel = self._soft_skeletonize(preds)
        target_skel = self._soft_skeletonize(targets)

        tprec = (torch.sum(pred_skel * targets) + self.smooth) / (torch.sum(pred_skel) + self.smooth)
        tsens = (torch.sum(target_skel * preds) + self.smooth) / (torch.sum(target_skel) + self.smooth)
        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice

