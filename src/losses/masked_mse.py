from __future__ import annotations

import torch
from torch import nn


class MaskedMSELoss(nn.Module):
    """Minimal custom loss retained because MONAI doesn't provide foreground-weighted masked MSE."""

    def __init__(
        self,
        foreground_weight: float = 200.0,
        eps: float = 1.0e-6,
        sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.foreground_weight = foreground_weight
        self.eps = eps
        self.sigmoid = sigmoid
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.sigmoid:
            preds = torch.sigmoid(preds)

        weights = torch.where(
            targets > 0,
            torch.full_like(targets, self.foreground_weight),
            torch.ones_like(targets),
        )
        loss = self.mse(preds, targets) * weights

        if mask is not None:
            loss = loss * mask
            weights = weights * mask

        denom = weights.sum().clamp_min(self.eps)
        return loss.sum() / denom
