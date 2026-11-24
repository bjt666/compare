"""Download classification weights and the model sources required for offline training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = {
    "unet": "resnet18-f37072fd.pth",
    "deeplabv3plus": "resnet50-0676ba61.pth",
    "unetformer": "resnet18-f37072fd.pth",
    "pugnet": "pvt_v2_b4.pth",
    "mseonet": "resnet101_v1c-e67eebb6.pth",
}
WEIGHTS = {
    "resnet18-f37072fd.pth": (
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    ),
    "resnet50-0676ba61.pth": (
        "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
    ),
    "pvt_v2_b4.pth": (
        "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth",
        "0952b4c89f7cb1b75527ffef7665be487911cc9177c03c735d6b64ddcedf88ed",
    ),
    "resnet101_v1c-e67eebb6.pth": (
        "https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth",
        "e67eebb62faab570c7c2268f93f5b68db2cd1730d5397923f34b39820c691d64",
    ),
}


def check_file(path: Path, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size < 1_000_000:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest() == sha256


def download(path: Path, url: str, sha256: str) -> None:
    if check_file(path, sha256):
        print(f"已有 {path.name}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        print(f"下载 {path.name} ({attempt + 1}/3)", flush=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as file:
            temporary = Path(file.name)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "GF16m-benchmark/1.0"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    shutil.copyfileobj(response, file, length=2**20)
                file.close()
                if not check_file(temporary, sha256):
                    raise ValueError(f"Incomplete download or checksum mismatch: {url}")
                temporary.replace(path)
                return
            except (OSError, ValueError) as exc:
                file.close()
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise SystemExit(f"下载失败：{url}\n可以手动下载到 {path}\n{exc}") from exc
        time.sleep(attempt + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", metavar="MODEL", help=f"all or {' / '.join(MODELS)}")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    for name in args.models:
        if name != "all" and name not in MODELS:
            parser.error(f"Unknown model: {name}")
    selected = MODELS if "all" in args.models or not args.models else args.models
    completed = set()
    for model in selected:
        if model in ("pugnet", "mseonet"):
            importlib.import_module(f"bench_models.{model}").prepare_source()
        name = MODELS[model]
        if not args.source_only and name not in completed:
            download(ROOT / "weights" / name, *WEIGHTS[name])
            completed.add(name)


if __name__ == "__main__":
    main()
