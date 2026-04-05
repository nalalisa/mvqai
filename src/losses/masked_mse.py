from __future__ import annotations

import torch
from torch import nn


class MaskedMSELoss(nn.Module):
    def __init__(self, foreground_weight: float = 200.0, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.foreground_weight = foreground_weight
        self.eps = eps

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        preds = torch.sigmoid(preds)
        weights = torch.ones_like(targets)
        weights = weights + (targets > 0).float() * (self.foreground_weight - 1.0)

        if mask is not None:
            weights = weights * mask

        loss = ((preds - targets) ** 2) * weights
        denom = weights.sum().clamp_min(self.eps)
        return loss.sum() / denom

