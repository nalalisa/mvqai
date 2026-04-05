from __future__ import annotations

from torch import nn
from monai.networks.nets import AttentionUnet, DynUNet, UNet


class MonaiUNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = UNet(**kwargs)

    def forward(self, x):
        return self.net(x)


class MonaiAttentionUnet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = AttentionUnet(**kwargs)

    def forward(self, x):
        return self.net(x)


class MonaiDynUNet(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = DynUNet(**kwargs)

    def forward(self, x):
        return self.net(x)

