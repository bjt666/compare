"""Train one independent model and score its best checkpoint on GF 16 m TIFFs."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark_data.dataset import DEFAULT_BANDS, DEFAULT_CHANNELS, PairedTiffDataset
from benchmark_data.dataset import build_manifest, compute_train_stats, file_sha256
from benchmark_data.lovasz import foreground_lovasz_loss
from benchmark_data.metrics import METRIC_POLICY, metrics_from_confusion


ROOT = Path(__file__).resolve().parent
MODELS = {
    "unet": ("resnet18-f37072fd.pth", 8),
    "deeplabv3plus": ("resnet50-0676ba61.pth", 4),
    "unetformer": ("resnet18-f37072fd.pth", 4),
    "pugnet": ("pvt_v2_b4.pth", 2),
    "mseonet": ("resnet101_v1c-e67eebb6.pth", 2),
}


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def resize_scores(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[:2] != (labels.shape[0], 2):
        raise ValueError(f"Expected B×2×H×W prediction, got {tuple(logits.shape)}")
    logits = logits.float()
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def criterion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = resize_scores(logits, labels)
    loss = F.cross_entropy(logits, labels)
    lovasz = foreground_lovasz_loss(logits, labels)
    return loss if lovasz is None else loss + 0.1 * lovasz


def scale_input(images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(images, scale_factor=2, mode="bilinear", align_corners=False)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def input_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = scale_input(batch["image"].to(device, non_blocking=True))
    labels = batch["mask"].to(device, non_blocking=True)
    return images, labels


def inference_logits(adapter, output):
    if hasattr(adapter, "inference_logits"):
        return adapter.inference_logits(output)
    return output


def update_epoch(model, adapter, loader, optimizer, scaler, device, args, *, check=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, examples, updates = 0.0, 0, 0
    accumulation = args.accumulate
    samples = len(loader.dataset)
    progress = tqdm(loader, desc="训练", leave=False)
    for i, batch in enumerate(progress):
        images, labels = input_batch(batch, device)
        with torch.autocast(device_type=device.type, enabled=args.amp):
            outputs = model(images)
        # Keep the pixel objective in FP32 and let native adapters select their final/auxiliary tensors.
        if hasattr(adapter, "training_loss"):
            loss = adapter.training_loss(outputs, labels, criterion)
        else:
            loss = criterion(inference_logits(adapter, outputs), labels)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite {args.model} training loss at batch {i + 1}")

        actual_batch = labels.shape[0]
        first = i // accumulation * accumulation * args.batch_size
        group_size = min(samples - first, accumulation * args.batch_size)
        scaler.scale(loss * (actual_batch / group_size)).backward()
        total_loss += loss.detach().item() * actual_batch
        examples += actual_batch
        if (i + 1) % accumulation == 0 or i + 1 == len(loader):
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= previous_scale:
                updates += 1
            if check and updates:
                break
        progress.set_postfix(loss=f"{total_loss / examples:.4f}")
    if not updates:
        raise FloatingPointError("No optimizer update completed. Check CUDA precision and gradients.")
    return total_loss / examples, updates


@torch.inference_mode()
def evaluate(model, adapter, loader, device) -> tuple[dict, list[dict]]:
    model.eval()
    matrix = np.zeros((2, 2), dtype=np.int64)
    rows = []
    for batch in tqdm(loader, desc="评估", leave=False):
        images, labels = input_batch(batch, device)
        outputs = model(images)
        scores = resize_scores(inference_logits(adapter, outputs), labels)
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError("Non-finite evaluation predictions")
        codes = (2 * labels + scores.argmax(dim=1)).flatten(1)
        for filename, image_codes in zip(batch["name"], codes.cpu().numpy()):
            counts = np.bincount(image_codes, minlength=4).reshape(2, 2)
            matrix += counts
            rows.append({"filename": filename, **metrics_from_confusion(counts)})
    return metrics_from_confusion(matrix), rows


def make_loader(dataset, args, *, training=False, epoch=0, check=False):
    generator = torch.Generator().manual_seed(args.seed + epoch)
    return DataLoader(
        dataset, batch_size=args.batch_size if training else args.eval_batch_size,
        shuffle=training, num_workers=0 if check else args.workers, pin_memory=args.device == "cuda",
        worker_init_fn=seed_worker, generator=generator,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulate", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-lr", type=float, default=6e-5)
    parser.add_argument("--head-lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action="store_true", help="CUDA FP16 with scaling. Default is FP32.")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/checkpoints/last.pt.")
    parser.add_argument("--check", action="store_true", help="Try an optimizer update with real TIFFs; save nothing.")
    args = parser.parse_args()
    weight_name, default_batch = MODELS[args.model]
    if args.batch_size is None:
        args.batch_size = default_batch
    if args.accumulate is None:
        if args.batch_size not in (1, 2, 4, 8):
            parser.error("Pass --accumulate explicitly for a batch size other than 1, 2, 4 or 8.")
        args.accumulate = 8 // args.batch_size
    if min(args.epochs, args.batch_size, args.accumulate, args.eval_batch_size) < 1 or args.workers < 0:
        parser.error("Epochs and batch sizes must be positive and workers cannot be negative.")
    if not 0 <= args.seed < 2**32 or min(args.encoder_lr, args.head_lr) <= 0 or args.weight_decay < 0:
        parser.error("Invalid seed, learning rate or weight decay.")
    if not all(map(math.isfinite, (args.encoder_lr, args.head_lr, args.weight_decay))):
        parser.error("Learning rates and weight decay must be finite.")
    if args.weights is not None and args.random_init:
        parser.error("Choose either --weights or --random-init.")
    if args.resume and args.check:
        parser.error("Use --resume or --check separately.")
    if args.amp and args.device != "cuda":
        parser.error("--amp requires --device cuda.")
    if args.model == "pugnet" and args.batch_size < 2:
        parser.error("PUGNet's original contrastive loss needs at least two images per physical batch.")
    args.weights = (args.weights or ROOT / "weights" / weight_name).resolve()
    args.data_root = args.data_root.resolve()
    args.output_dir = (args.output_dir or ROOT / "outputs" / f"{args.model}_s{args.seed}").resolve()
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用。请确认实例已分配 GPU，并使用 setup.sh 创建的环境。")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoints = args.output_dir / "checkpoints"
    last_path, best_path = checkpoints / "last.pt", checkpoints / "best.pt"
    pending_best_path = checkpoints / "best_pending.pt"
    previous = None
    if args.resume:
        previous = torch.load(last_path, map_location="cpu", weights_only=False)
    elif not args.check and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"输出目录已有内容：{args.output_dir}。续训请加 --resume，或选择新目录。")

    manifest = build_manifest(args.data_root, ("train", "val", "test")) if not args.check else None
    adapter = importlib.import_module(f"bench_models.{args.model}")
    settings = {
        "model": args.model, "seed": args.seed, "epochs": args.epochs, "amp": args.amp, "device": args.device,
        "physical_batch_size": args.batch_size, "accumulation_steps": args.accumulate,
        "effective_batch_size": args.batch_size * args.accumulate,
        "workers": args.workers, "eval_batch_size": args.eval_batch_size,
        "encoder_lr": args.encoder_lr, "head_lr": args.head_lr, "weight_decay": args.weight_decay,
        "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "pretrained": not args.random_init,
        "loss": getattr(adapter, "LOSS_NAME", "CE + 0.1 × foreground Lovasz"),
        "bands": list(DEFAULT_BANDS), "band_indices_zero_based": list(DEFAULT_CHANNELS),
        "input_scale": 2, "metric_grid": "native", "selected_by": "val_foreground_iou",
        "eval_precision": "fp32",
        "augmentation": "independent horizontal/vertical 0.5 and uniform quarter turns",
    }
    if previous is not None:
        if previous["settings"] != settings or previous["dataset_fingerprint"] != manifest["fingerprint"]:
            raise SystemExit("续训设置或数据文件已改变，请核对原运行参数。")
        mean = np.asarray(previous["mean"], dtype=np.float32)
        std = np.asarray(previous["std"], dtype=np.float32)
        initialization = previous["initialization"]
    else:
        mean, std = compute_train_stats(args.data_root, DEFAULT_CHANNELS)
        initialization = None
        if not args.random_init:
            if not args.weights.is_file():
                raise SystemExit(f"缺少权重 {args.weights}。请先运行 setup.sh，或使用 --weights 指定文件。")
            initialization = {"filename": args.weights.name, "sha256": file_sha256(args.weights)}

    datasets = {name: PairedTiffDataset(args.data_root, name, mean, std, augment=name == "train")
                for name in ("train", "val", "test")}
    if args.model == "pugnet" and (len(datasets["train"]) < 2 or len(datasets["train"]) % args.batch_size == 1):
        raise SystemExit("PUGNet 的最后一批也需要至少两张图。请换物理 batch 大小。")
    print(f"{args.model} / {device} / split sizes: " + "/".join(str(len(d)) for d in datasets.values()))
    model = adapter.build(previous is None and not args.random_init, str(args.weights)).to(device)
    if hasattr(adapter, "backbone_parameters"):
        encoder_parameters = list(adapter.backbone_parameters(model))
    else:
        backbone = getattr(model, "backbone", getattr(model, "encoder", None))
        if backbone is None:
            raise RuntimeError(f"{args.model} adapter did not expose its backbone")
        encoder_parameters = list(backbone.parameters())
    encoder_parameters = [p for p in encoder_parameters if p.requires_grad]
    encoder_ids = {id(p) for p in encoder_parameters}
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": args.encoder_lr},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids],
         "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp, init_scale=1024.0)
    model_summary = {"parameters_total": sum(p.numel() for p in model.parameters()),
                     "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}

    if args.check:
        loader = make_loader(datasets["train"], args, training=True, check=True)
        loss, _ = update_epoch(model, adapter, loader, optimizer, scaler, device, args, check=True)
        with torch.inference_mode():
            model.eval()
            images, labels = input_batch(next(iter(make_loader(datasets["val"], args, check=True))), device)
            outputs = model(images)
            scores = resize_scores(inference_logits(adapter, outputs), labels)
            if not torch.isfinite(scores).all().item():
                raise FloatingPointError("Check produced non-finite evaluation scores.")
        print(f"CHECK OK: input {tuple(images.shape)}, aligned scores {tuple(scores.shape)}, loss {loss:.4f}, "
              f"{model_summary['parameters_total']:,} parameters")
        if device.type == "cuda":
            print(f"CUDA memory peak: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated")
        return

    checkpoints.mkdir(parents=True, exist_ok=True)
    start_epoch, best_epoch, best_iou = 1, 0, -1.0
    history = []
    if previous is not None:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        start_epoch, best_epoch, best_iou = previous["epoch"] + 1, previous["best_epoch"], previous["best_iou"]
        history = previous["history"]
        # Finish promoting a staged best only if last.pt committed its epoch.
        if pending_best_path.is_file():
            pending = torch.load(pending_best_path, map_location="cpu", weights_only=True)
            if pending["epoch"] == best_epoch:
                pending_best_path.replace(best_path)
            else:
                pending_best_path.unlink()
            del pending
        torch.set_rng_state(previous["rng"]["torch"])
        random.setstate(previous["rng"]["python"])
        np.random.set_state(previous["rng"]["numpy"])
        if device.type == "cuda" and previous["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(previous["rng"]["cuda"])
        del previous

    atomic_json(args.output_dir / "config.json", {
        **settings, "initialization": initialization, "data_root": str(args.data_root),
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "dataset_fingerprint": manifest["fingerprint"], "workers": args.workers,
        "eval_batch_size": args.eval_batch_size, "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
    })
    atomic_json(args.output_dir / "dataset_manifest.json", manifest)
    atomic_json(args.output_dir / "model_summary.json", model_summary)
    val_loader = make_loader(datasets["val"], args)
    if history:
        write_csv(args.output_dir / "metrics.csv", history)
    for epoch in range(start_epoch, args.epochs + 1):
        encoder_lr, head_lr = scheduler.get_last_lr()
        train_loss, updates = update_epoch(
            model, adapter, make_loader(datasets["train"], args, training=True, epoch=epoch),
            optimizer, scaler, device, args,
        )
        val_metrics, _ = evaluate(model, adapter, val_loader, device)
        value = val_metrics["foreground_iou"]
        if value is None:
            raise ValueError("Validation foreground IoU is undefined. Check validation masks.")
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "optimizer_steps": updates,
                        "encoder_lr": encoder_lr, "head_lr": head_lr,
                        **{f"val_{key}": value for key, value in val_metrics.items()}})
        improved = value > best_iou
        if improved:
            best_epoch, best_iou = epoch, value
            save_checkpoint(pending_best_path, {
                "model": model.state_dict(), "epoch": best_epoch, "val_iou": best_iou,
            })
        save_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best_epoch": best_epoch, "best_iou": best_iou,
            "settings": settings, "history": history, "dataset_fingerprint": manifest["fingerprint"],
            "initialization": initialization, "mean": mean.tolist(), "std": std.tolist(),
            "rng": {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
        })
        if improved:
            pending_best_path.replace(best_path)
        write_csv(args.output_dir / "metrics.csv", history)
        print(f"Epoch {epoch:3d}/{args.epochs}: loss {train_loss:.4f}  val IoU {value:.4%}  best {best_iou:.4%}",
              flush=True)

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    best_sha = file_sha256(best_path)
    for split in ("val", "test"):
        result, rows = evaluate(model, adapter, make_loader(datasets[split], args), device)
        split_dir = args.output_dir / "evaluation" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(split_dir / "metrics.json", {
            "split": split, "model": args.model, "seed": args.seed, "num_images": len(rows),
            "best_epoch": best["epoch"], "best_val_iou": best["val_iou"], "checkpoint_sha256": best_sha,
            "dataset_fingerprint": manifest["fingerprint"], "policy": METRIC_POLICY, **result,
        })
        write_csv(split_dir / "per_image_metrics.csv", rows)
        if result["foreground_iou"] is not None:
            print(f"Best checkpoint / {split} / IoU {result['foreground_iou']:.4%} / {split_dir}", flush=True)


if __name__ == "__main__":
    main()
