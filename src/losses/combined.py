from __future__ import annotations

import torch
from hydra.utils import instantiate
from torch import nn


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float, masked_mse: nn.Module, soft_cldice: nn.Module) -> None:
        super().__init__()
        self.alpha = alpha
        self.masked_mse = instantiate(masked_mse) if not isinstance(masked_mse, nn.Module) else masked_mse
        self.soft_cldice = instantiate(soft_cldice) if not isinstance(soft_cldice, nn.Module) else soft_cldice

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mse = self.masked_mse(preds, targets, mask=mask)
        cldice = self.soft_cldice(preds, targets, mask=mask)
        return self.alpha * mse + (1.0 - self.alpha) * cldice

