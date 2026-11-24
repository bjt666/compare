"""GF16 adapter for the authors' single-temporal PUGNet–PVT-v2-B4."""

from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from urllib.request import Request, urlopen


UPSTREAM_REVISION = "505a33d7f887c7d3bf01e5621ea6bd2ec622098e"
WEIGHTS_URL = "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth"
WEIGHTS_FILENAME = "pvt_v2_b4.pth"
LOSS_NAME = "PUGNet: four cross-entropies + native contrastive loss"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_DIR = _PROJECT_ROOT / "vendor" / "pugnet"
_SOURCE_URL = (
    "https://raw.githubusercontent.com/Henryjiepanli/PU_RS/"
    f"{UPSTREAM_REVISION}/single-temporal_binary_segmentation/geoseg/models/PUGNet/"
)
_SOURCE_HASHES = {
    "PUGNet.py": "edd7e97f9ce03ef8a6f9dfb7544043164d052f7221ff0a8dd18280bbec49d7ab",
    "pvtv2.py": "bad9a96bca9841679b5f7a5a97bf9bbc4239611fb4be6d0ba0783f64e4f9dc8c",
    "resnet.py": "2805fe0788e9ab6f691960f06bb921232db8ed4991d1eea8908740c9124b8c1a",
    "Res2Net_v1b.py": "8b519434e85e1531d2e3188a900915674fc749c6f5d75eb93b47b80bcc69a6fb",
    "ResNext101.py": "30c06a8cb7ef9ff9398ca49e8ff4bbb6ea72292ae63744b7c31c8550ba93cd5a",
}


def _matches(path: Path, sha256: str) -> bool:
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256


def prepare_source(destination: str | Path = _SOURCE_DIR) -> Path:
    """Fetch and verify the authors' files on an Internet-connected host."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, sha256 in _SOURCE_HASHES.items():
        target = destination / name
        if _matches(target, sha256):
            continue
        request = Request(_SOURCE_URL + name, headers={"User-Agent": "GF16-PUGNet-benchmark"})
        with urlopen(request, timeout=60) as response:
            source_bytes = response.read()
        if hashlib.sha256(source_bytes).hexdigest() != sha256:
            raise RuntimeError(f"PUGNet source integrity check failed for {name}")
        temporary = destination / (name + ".download")
        temporary.write_bytes(source_bytes)
        temporary.replace(target)
    return destination


def _load_upstream(source_dir: Path):
    invalid = [name for name, sha256 in _SOURCE_HASHES.items() if not _matches(source_dir / name, sha256)]
    if invalid:
        raise FileNotFoundError(
            f"Missing or mismatched PUGNet source under {source_dir}: {', '.join(invalid)}. "
            "Run `python -m bench_models.pugnet` on a machine with Internet access, "
            "then copy vendor/pugnet to the training server."
        )
    package_name = "_gf16m_pugnet_" + hashlib.sha256(str(source_dir).encode()).hexdigest()[:12]
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(source_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.PUGNet")


def _load_backbone_weights(backbone, path: Path) -> None:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"PVT-v2-B4 ImageNet weights not found at {path}. Download: {WEIGHTS_URL}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping):
        nested = state.get("model", state.get("state_dict"))
        if isinstance(nested, Mapping):
            state = nested
    if not isinstance(state, Mapping):
        raise ValueError(f"Expected a PVT-v2-B4 state dictionary at {path}")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    expected = backbone.state_dict()
    missing = set(expected).difference(state)
    if missing:
        example = ", ".join(sorted(missing)[:4])
        raise ValueError(f"Incompatible PVT-v2-B4 checkpoint at {path}: {len(missing)} missing ({example})")
    state = {key: state[key] for key in expected}
    input_weight = state["patch_embed1.proj.weight"]
    # Source channel order RGB. The GF16 comparison feeds NIR, R, G.
    state["patch_embed1.proj.weight"] = torch.cat(
        (input_weight.mean(dim=1, keepdim=True), input_weight[:, :2]), dim=1
    )
    backbone.load_state_dict(state, strict=True)


def build(
    pretrained: bool = True,
    weights_path: str | Path | None = None,
    *,
    source_dir: str | Path = _SOURCE_DIR,
):
    """Build the official architecture using an explicitly chosen local checkpoint."""
    upstream = _load_upstream(Path(source_dir).resolve())
    path = Path(weights_path) if weights_path else _PROJECT_ROOT / "weights" / WEIGHTS_FILENAME

    class PUGNetGF16(upstream.PUGNet):
        def get_backbone(self, backbone_name):
            if backbone_name != "PVT-v2":
                raise ValueError(f"Unexpected PUGNet backbone: {backbone_name}")
            backbone = upstream.pvt_v2_b4()
            if pretrained:
                _load_backbone_weights(backbone, path)
            return backbone, [64, 128, 320, 512]

    return PUGNetGF16(backbone_name="PVT-v2", channel=32)


def inference_logits(model_outputs):
    """Use the final prediction head, as in the official single-image evaluator."""
    return model_outputs[3]


def training_loss(model_outputs, labels, criterion=None):
    """Preserve the native sum of four cross-entropies and batch contrastive term."""
    from torch.nn import functional as F

    if labels.shape[0] < 2:
        raise ValueError("PUGNet training requires at least two images per physical batch for its contrastive loss.")
    stage_losses = []
    for logits in model_outputs[:4]:
        logits = logits.float()
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        stage_losses.append(F.cross_entropy(logits, labels))
    return sum(stage_losses) + model_outputs[4].float()


if __name__ == "__main__":
    print(f"Prepared PUGNet source at {prepare_source()}")
