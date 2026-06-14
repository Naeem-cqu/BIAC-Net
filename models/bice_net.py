"""BICE-Net implementation with bidirectional inter-attention communication."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .attention import CBAM, DepthwiseSeparableConv


class SharedBottleneck(nn.Module):
    """Shared projection used by both branches."""

    def __init__(self, channels: int, ratio: int = 4) -> None:
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.project = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


class BICEBlock(nn.Module):
    """Bidirectional inter-attention communication."""

    def __init__(self, channels: int, steps: int = 1, bottleneck_ratio: int = 4) -> None:
        super().__init__()
        self.global_attn = CBAM(channels)
        self.local_attn = DepthwiseSeparableConv(channels)
        self.shared_proj = SharedBottleneck(channels, ratio=bottleneck_ratio)
        self.steps = steps
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fa = self.global_attn(x)
        fb = self.local_attn(x)

        for _ in range(self.steps):
            ga = self.shared_proj(fa)
            gb = self.shared_proj(fb)
            fa = self.global_attn(x + self.alpha * gb)
            fb = self.local_attn(x + self.beta * ga)

        return fa, fb


class MultiScaleBICE(nn.Module):
    """Apply BICE at multiple spatial scales using shared weights."""

    def __init__(self, channels: int, scales: int = 2, steps: int = 1) -> None:
        super().__init__()
        self.scales = scales
        self.shared_bice = BICEBlock(channels, steps=steps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fa, fb = self.shared_bice(x)

        if self.scales <= 1:
            return fa, fb

        down = F.avg_pool2d(x, kernel_size=2, stride=2)
        fa_down, fb_down = self.shared_bice(down)
        fa = fa + F.interpolate(fa_down, size=x.shape[-2:], mode="bilinear", align_corners=False)
        fb = fb + F.interpolate(fb_down, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return fa, fb


class GatedFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([a, b], dim=1))
        return gate * a + (1.0 - gate) * b


class BICENet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet101",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        backbone = backbone.lower()
        self.backbone_name = backbone
        if backbone == "vit_b_16":
            self.backbone = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.feature_channels = self.backbone.hidden_dim
        elif backbone == "resnet101":
            self.backbone = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            )
            self.feature_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.bice = MultiScaleBICE(self.feature_channels, scales=2, steps=1)
        self.fusion = GatedFusion(self.feature_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.feature_channels, num_classes)

    def _vit_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone._process_input(x)
        batch_size, num_tokens, _ = x.shape
        cls_token = self.backbone.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.backbone.encoder(x)
        x = x[:, 1:, :]
        grid_size = int(math.isqrt(num_tokens))
        return x.transpose(1, 2).reshape(
            batch_size, self.feature_channels, grid_size, grid_size
        )

    def _resnet_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "vit_b_16":
            features = self._vit_features(x)
        else:
            features = self._resnet_features(x)

        global_refined, local_refined = self.bice(features)
        fused = self.fusion(global_refined, local_refined)
        pooled = self.pool(fused).flatten(1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

