"""GeoSeg UNetFormer decoder with a supervised ImageNet-1K ResNet-18."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from vendor.geoseg.UNetFormer import Decoder

from .unet import load_rgb_backbone


LOSS_NAME = "SoftCE (smoothing 0.05) + Dice (smooth 0.05) + 0.4 × auxiliary SoftCE"


class UNetFormer(nn.Module):
    def __init__(self, pretrained: bool, weights_path: str | None):
        super().__init__()
        self.backbone = resnet18(weights=None)
        load_rgb_backbone(self.backbone, pretrained, weights_path)
        self.decoder = Decoder(encoder_channels=(64, 128, 256, 512), num_classes=2)

    def forward(self, image: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        backbone = self.backbone
        stem = backbone.relu(backbone.bn1(backbone.conv1(image)))
        stage1 = backbone.layer1(backbone.maxpool(stem))
        stage2 = backbone.layer2(stage1)
        stage3 = backbone.layer3(stage2)
        stage4 = backbone.layer4(stage3)
        height, width = image.shape[-2:]
        return self.decoder(stage1, stage2, stage3, stage4, height, width)


def build(pretrained: bool, weights_path: str | None) -> nn.Module:
    return UNetFormer(pretrained, weights_path)


def inference_logits(model_outputs):
    return model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs


def training_loss(model_outputs, labels, criterion):
    main_logits, aux_logits = (
        F.interpolate(output.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
        for output in model_outputs
    )
    # GeoSeg and PyTorch both distribute the smoothing mass across all classes.
    main_ce = F.cross_entropy(main_logits, labels, label_smoothing=0.05)
    aux_ce = F.cross_entropy(aux_logits, labels, label_smoothing=0.05)

    probabilities = main_logits.log_softmax(dim=1).exp()
    targets = F.one_hot(labels, num_classes=2).movedim(-1, 1)
    dims = (0, 2, 3)
    target_count = targets.sum(dims)
    intersection = (probabilities * targets).sum(dims)
    denominator = probabilities.sum(dims) + target_count
    dice = (2 * intersection + 0.05) / (denominator + 0.05).clamp_min(1e-7)
    dice_loss = ((1 - dice) * (target_count > 0)).mean()
    return main_ce + dice_loss + 0.4 * aux_ce
