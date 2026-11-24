"""DeepLabV3+ with a ResNet-50 and output stride 16.

Chen et al., ECCV 2018. The decoder fuses stride-4 features with ASPP output.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from .unet import conv_bn_relu, load_rgb_backbone


def separable_block(in_channels: int, out_channels: int, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        conv_bn_relu(in_channels, out_channels, 1),
    )


class PooledBatchNorm(nn.BatchNorm2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.shape[0] == 1:
            # Per-channel batch variance is undefined for a single pooled
            # pixel. Use the running estimates on that microbatch.
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([conv_bn_relu(2048, 256, 1)])
        self.projections.extend(separable_block(2048, 256, rate) for rate in (6, 12, 18))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 256, 1, bias=False),
            PooledBatchNorm(256),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(conv_bn_relu(1280, 256, 1), nn.Dropout(0.1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.global_pool(features)
        pooled = F.interpolate(pooled, size=features.shape[-2:], mode="bilinear", align_corners=False)
        contexts = [projection(features) for projection in self.projections]
        contexts.append(pooled)
        return self.fuse(torch.cat(contexts, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, pretrained: bool, weights_path: str | None):
        super().__init__()
        self.backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        load_rgb_backbone(self.backbone, pretrained, weights_path)
        self.aspp = ASPP()
        self.low_level = conv_bn_relu(256, 48, 1)
        self.decoder = nn.Sequential(separable_block(304, 256), separable_block(256, 256), nn.Conv2d(256, 2, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        stem = backbone.relu(backbone.bn1(backbone.conv1(image)))
        low_level = backbone.layer1(backbone.maxpool(stem))
        features = backbone.layer4(backbone.layer3(backbone.layer2(low_level)))
        context = self.aspp(features)
        context = F.interpolate(context, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(torch.cat((self.low_level(low_level), context), dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def build(pretrained: bool, weights_path: str | None) -> nn.Module:
    return DeepLabV3Plus(pretrained, weights_path)
