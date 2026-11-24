"""U-Net decoder with a standard torchvision ResNet-18 encoder."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


def load_rgb_backbone(backbone: nn.Module, pretrained: bool, weights_path: str | None) -> None:
    if pretrained:
        if weights_path is None or not Path(weights_path).is_file():
            raise FileNotFoundError(f"Expected a local ImageNet backbone checkpoint: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)

        # Keep the red and green filters in their own channels. Initialize the
        # new near-infrared input using the average pretrained RGB filter.
        with torch.no_grad():
            rgb = backbone.conv1.weight.detach().clone()
            backbone.conv1.weight.copy_(torch.cat((rgb.mean(dim=1, keepdim=True), rgb[:, :2]), dim=1))

    backbone.fc = nn.Identity()


def conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.convs = nn.Sequential(
            conv_bn_relu(in_channels + skip_channels, out_channels),
            conv_bn_relu(out_channels, out_channels),
        )

    def forward(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convs(torch.cat((features, skip), dim=1))


class ResNetUNet(nn.Module):
    def __init__(self, pretrained: bool, weights_path: str | None):
        super().__init__()
        self.backbone = resnet18(weights=None)
        load_rgb_backbone(self.backbone, pretrained, weights_path)
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.full_resolution = nn.Sequential(conv_bn_relu(32, 16), conv_bn_relu(16, 16), nn.Conv2d(16, 2, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        stem = backbone.relu(backbone.bn1(backbone.conv1(image)))
        stage1 = backbone.layer1(backbone.maxpool(stem))
        stage2 = backbone.layer2(stage1)
        stage3 = backbone.layer3(stage2)
        stage4 = backbone.layer4(stage3)

        decoded = self.up4(stage4, stage3)
        decoded = self.up3(decoded, stage2)
        decoded = self.up2(decoded, stage1)
        decoded = self.up1(decoded, stem)
        decoded = F.interpolate(decoded, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.full_resolution(decoded)


def build(pretrained: bool, weights_path: str | None) -> nn.Module:
    return ResNetUNet(pretrained, weights_path)
