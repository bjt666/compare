"""MSEONet-R101 for two-class GF16 data with a PyTorch training interface.

The authors' MSCA and EPFO layer definitions are fetched during setup from
HuangWBill/MSEONet at the pinned revision. This adapter implements only the
small OpenMMLab interface used by those layers and the cascade's point loss
and scatter. The upstream EPFO predict call has mismatched arguments and its
fallback point selection can reuse another image's mask. Point selection here
prioritizes boundary points, fills from other points and never repeats a point.
"""

import ast
import functools
import hashlib
import io
from pathlib import Path
from typing import NamedTuple
from urllib.request import Request, urlopen
from zipfile import ZipFile

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet101


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "feb759b61213563f97b00553aacc783b5c933230"
SOURCE_DIR = ROOT / "vendor" / "mseonet"
LOSS_NAME = "MSEONet: MSCA dense CE + EPFO sampled-point CE"
SOURCE_HASHES = {
    "msca_head.py": "ec1cbadbdedf80a75ad7b7b3619d27dff4b5d6ec1c9e9b6e0d6bf21b2399c6bc",
    "epfo_head.py": "ebac3d7b727dc130a34dba569e907ff756e05de8e6383b54c50667bb70376c97",
}
WEIGHTS = ROOT / "weights" / "resnet101_v1c-e67eebb6.pth"


def prepare_source(destination=SOURCE_DIR) -> Path:
    """Get only the two author heads excluded from the distributed project."""
    destination = Path(destination)
    repository = "HuangWBill/MSEONet"
    folder = "mmseg/decode_head"

    def fetch(url):
        request = Request(url, headers={"User-Agent": "GF16-MSEONet-benchmark"})
        with urlopen(request, timeout=45) as response:
            return response.read()

    archive = None
    destination.mkdir(parents=True, exist_ok=True)
    for filename, checksum in SOURCE_HASHES.items():
        target = destination / filename
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == checksum:
            continue
        url = f"https://raw.githubusercontent.com/{repository}/{UPSTREAM_REVISION}/{folder}/{filename}"
        try:
            payload = fetch(url)
        except OSError:
            if archive is None:
                archive = fetch(f"https://codeload.github.com/{repository}/zip/{UPSTREAM_REVISION}")
            with ZipFile(io.BytesIO(archive)) as saved:
                payload = saved.read(f"MSEONet-{UPSTREAM_REVISION}/{folder}/{filename}")
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError(f"MSEONet source checksum mismatch: {url}")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
    return destination


class _SafeBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        # MSCA has a 1x1 pooled level. An odd-sized data split can leave a
        # singleton tail, so use the learned running statistics for that tail.
        if self.training and x.shape[0] * x.shape[2] * x.shape[3] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                training=False, eps=self.eps)
        return super().forward(x)


class _ConvModule(nn.Sequential):
    """The Conv1d/Conv2d, optional BN, ReLU subset used by upstream heads."""

    def __init__(self, in_channels, out_channels, kernel_size, *, stride=1, padding=0,
                 conv_cfg=None, norm_cfg=None, act_cfg=None):
        convolution = nn.Conv1d if conv_cfg and conv_cfg["type"] == "Conv1d" else nn.Conv2d
        conv = convolution(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, bias=norm_cfg is None)
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
        layers = [conv]
        if norm_cfg is not None:
            layers.append(_SafeBatchNorm2d(out_channels))
        if act_cfg is not None:
            layers.append(nn.ReLU(inplace=act_cfg.get("inplace", True)))
        super().__init__(*layers)


class _BaseHead(nn.Module):
    def __init__(self, *, in_channels, in_index, channels, num_classes,
                 align_corners, dropout_ratio, norm_cfg=None, conv_cfg=None,
                 act_cfg=None, input_transform=None, init_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.in_index = in_index
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.dropout_ratio = dropout_ratio
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = {"type": "ReLU"} if act_cfg is None else act_cfg
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        nn.init.normal_(self.conv_seg.weight, std=0.01)
        nn.init.zeros_(self.conv_seg.bias)

    def cls_seg(self, features):
        if self.dropout is not None:
            features = self.dropout(features)
        return self.conv_seg(features)


def _point_sample(x, points, *, mode="bilinear", align_corners=False):
    # mmcv point_sample is grid_sample on [0,1] coordinates. Use float32 for
    # coordinates and interpolation, independently of CUDA mixed precision.
    grid = points.float().mul(2).sub(1).unsqueeze(2)
    return F.grid_sample(x.float(), grid, mode=mode, align_corners=align_corners).squeeze(3)


def _is_openmmlab_import(node):
    return (isinstance(node, ast.ImportFrom) and node.module is not None
            and node.module.split(".")[0] in {"mmcv", "mmseg"})


@functools.lru_cache(maxsize=2)
def _load_head(filename, name):
    path = SOURCE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned MSEONet source {path}. Run the project setup first.")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_HASHES[filename]:
        raise ValueError(f"{path} differs from MSEONet revision {UPSTREAM_REVISION}.")
    tree = ast.parse(payload, filename=str(path))
    # Supply a local adapter in place of mmcv/mmseg imports. The original
    # MSCA and EPFO layer constructors and forward implementations execute.
    body = []
    for node in tree.body:
        if _is_openmmlab_import(node):
            continue
        if isinstance(node, ast.Try) and len(node.body) == 1 and _is_openmmlab_import(node.body[0]):
            continue
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
        body.append(node)
    tree.body = body
    scope = {
        "__name__": f"mseonet_official_{path.stem}",
        "ConvModule": _ConvModule,
        "BaseDecodeHead": _BaseHead,
        "BaseCascadeDecodeHead": _BaseHead,
        "SampleList": list,
        "resize": F.interpolate,
        "point_sample": _point_sample,
    }
    exec(compile(tree, str(path), "exec"), scope)
    return scope[name]


class _ResNet101V1c(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        # Torchvision stages use the same v1c PyTorch bottleneck convention.
        # The previous dilation in the first block implements contract_dilation.
        resnet = resnet101(weights=None, replace_stride_with_dilation=(False, True, True))
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)["state_dict"]
            state = {name: value for name, value in state.items() if name not in {"fc.weight", "fc.bias"}}
            # Pretrained RGB filters become NIR/R/G. No pretrained blue
            # filter corresponds to NIR, so initialize NIR from their mean.
            rgb = state["stem.0.weight"]
            state["stem.0.weight"] = torch.cat((rgb.mean(dim=1, keepdim=True), rgb[:, :2]), dim=1)
            self.load_state_dict(state, strict=True)
        else:
            for layer in self.stem:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, image):
        f1 = self.layer1(self.maxpool(self.stem(image)))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4


@torch.no_grad()
def _select_points(scores):
    height, width = scores.shape[-2:]
    coarse = scores.float()
    top_two = coarse.topk(2, dim=1).values
    uncertainty = (top_two[:, 1] - top_two[:, 0]).flatten(1)

    # Predicted label transitions within the 5x5 neighborhood, equivalent to
    # upstream boundary extraction for binary logits.
    labels = coarse.argmax(dim=1, keepdim=True).float()
    high = F.max_pool2d(labels, 5, stride=1, padding=2)
    low = -F.max_pool2d(-labels, 5, stride=1, padding=2)
    boundary = (high != low).flatten(1)
    preferred = boundary & (uncertainty != 0)
    counts = boundary.sum(dim=1).tolist()
    point_count = int(min(counts) * 0.75)
    if point_count == 0:
        point_count = int(max(counts) * 0.75)
    point_count = min(max(point_count, 1), height * width)

    # Keep boundary candidates ahead of other points at every score range.
    # A single topk avoids duplicate scatter indices in fallback cases.
    priority = uncertainty - uncertainty.amin(dim=1, keepdim=True)
    margin = priority.amax(dim=1, keepdim=True) + 1.0
    priority = priority + preferred.to(priority.dtype) * margin
    indices = priority.topk(point_count, dim=1).indices
    # Keep the official EPFO normalization at integer feature coordinates.
    xy = torch.stack((indices.remainder(width).float() / width,
                      indices.div(width, rounding_mode="floor").float() / height), dim=-1)
    return indices, xy


class _Output(NamedTuple):
    coarse: torch.Tensor
    refined: torch.Tensor
    point_logits: torch.Tensor
    point_coords: torch.Tensor


class _MSEONet(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        self.backbone = _ResNet101V1c(weights_path)
        coarse_head = _load_head("msca_head.py", "MSCA_Head")
        point_head = _load_head("epfo_head.py", "EPFO_Head")
        self.msca = coarse_head(in_channels=2048, in_index=3, channels=512, num_classes=2,
                                align_corners=False, pool_scales=(1, 2, 3, 6), use_CRF=False,
                                dropout_ratio=0.1, norm_cfg={"type": "BN"})
        self.epfo = point_head(in_channels=[256], in_index=[0], channels=256, num_classes=2,
                               align_corners=False, dropout_ratio=-1, num_fcs=3, theta=5,
                               coarse_pred_each_layer=True)
        nn.init.normal_(self.epfo.fc_seg.weight, std=0.01)
        nn.init.zeros_(self.epfo.fc_seg.bias)

    def forward(self, image):
        features = self.backbone(image)
        coarse = self.msca(features)
        point_grid = F.interpolate(coarse, size=features[0].shape[-2:],
                                   mode="bilinear", align_corners=False)
        indices, coords = _select_points(point_grid)
        point_logits = self.epfo(_point_sample(features[0], coords), _point_sample(point_grid, coords))
        refined = point_grid.flatten(2).scatter(2, indices[:, None].expand(-1, 2, -1),
                                                point_logits.to(point_grid.dtype)).reshape_as(point_grid)
        return _Output(coarse, refined, point_logits, coords)


def build(pretrained: bool = True, weights_path=None) -> nn.Module:
    weights = Path(weights_path) if weights_path else WEIGHTS
    if pretrained and not weights.is_file():
        raise FileNotFoundError(f"MSEONet needs {weights}. Run the project setup first.")
    return _MSEONet(weights if pretrained else None)


def inference_logits(model_outputs):
    return model_outputs.refined


def training_loss(model_outputs, labels, criterion):
    # Original stage-0 dense CE + stage-1 point CE. Resized logits / sampled
    # labels stay aligned to the original GF mask's normalized coordinates.
    coarse = F.interpolate(model_outputs.coarse.float(), size=labels.shape[-2:],
                           mode="bilinear", align_corners=False)
    point_labels = _point_sample(labels[:, None], model_outputs.point_coords, mode="nearest")
    return F.cross_entropy(coarse, labels) + F.cross_entropy(model_outputs.point_logits.float(),
                                                           point_labels.squeeze(1).long())
