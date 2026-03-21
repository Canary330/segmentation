from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm_backend.data import PromptSegmentationDataset
from vlm_backend.model import (
    PromptableMobileUNetFPN,
    load_pure_visual_checkpoint_into_prompt_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the prompt-driven fetal ultrasound VLM backend.")
    parser.add_argument("--data-root", type=Path, default=Path("prepared/prompt_seg"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/vlm_prompt_seg"))
    parser.add_argument("--text-encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--init-pure-visual-checkpoint", type=Path, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--train-sources",
        type=str,
        default="",
        help="Comma-separated list of sources to keep for training.",
    )
    parser.add_argument(
        "--eval-sources",
        type=str,
        default="",
        help="Comma-separated list of sources to keep for validation/testing.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    dice = (2.0 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
    }


def collate_fn(batch: list[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    labels = [item["label"] for item in batch]
    sources = [item["source"] for item in batch]
    return {"images": images, "masks": masks, "prompts": prompts, "labels": labels, "sources": sources}


def filter_manifest(manifest_path: Path, allowed_sources: set[str], output_path: Path) -> Path:
    if not allowed_sources:
        return manifest_path
    lines = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["source"] in allowed_sources:
            lines.append(line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def run_epoch(
    model: PromptableMobileUNetFPN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    bce_loss = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total = 0
    progress = tqdm(loader, leave=False, desc="train" if training else "eval")

    for batch in progress:
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)
        prompts = batch["prompts"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, prompts)
            loss = 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss_from_logits(logits, masks)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_metrics = compute_metrics(logits.detach(), masks)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_dice += batch_metrics["dice"] * batch_size
        running_iou += batch_metrics["iou"] * batch_size
        total += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_metrics['dice']:.4f}")

    return {
        "loss": running_loss / max(total, 1),
        "dice": running_dice / max(total, 1),
        "iou": running_iou / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    train_sources = {item.strip() for item in args.train_sources.split(",") if item.strip()}
    eval_sources = {item.strip() for item in args.eval_sources.split(",") if item.strip()}

    filtered_dir = args.experiment_dir / "filtered_manifests"
    train_manifest = filter_manifest(args.data_root / "training.jsonl", train_sources, filtered_dir / "training.jsonl")
    val_manifest = filter_manifest(args.data_root / "validation.jsonl", eval_sources, filtered_dir / "validation.jsonl")
    test_manifest = filter_manifest(args.data_root / "testing.jsonl", eval_sources, filtered_dir / "testing.jsonl")

    train_dataset = PromptSegmentationDataset(train_manifest, image_size=args.img_size, train=True)
    val_dataset = PromptSegmentationDataset(val_manifest, image_size=args.img_size, train=False)
    test_dataset = PromptSegmentationDataset(test_manifest, image_size=args.img_size, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = PromptableMobileUNetFPN(
        text_encoder_name=args.text_encoder,
        fpn_channels=args.fpn_channels,
        dropout=args.dropout,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)
    if args.init_pure_visual_checkpoint is not None:
        pure_visual_checkpoint = torch.load(
            args.init_pure_visual_checkpoint,
            map_location=device,
            weights_only=False,
        )
        missing, unexpected = load_pure_visual_checkpoint_into_prompt_model(
            model,
            pure_visual_checkpoint,
        )
        print(
            "Initialized visual branch from pure-visual checkpoint "
            f"{args.init_pure_visual_checkpoint} | missing={len(missing)} | unexpected={len(unexpected)}"
        )
    if args.init_checkpoint is not None:
        init_checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init_checkpoint["model"], strict=False)
        print(f"Initialized VLM weights from {args.init_checkpoint}")

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, use_amp)
        val_metrics = run_epoch(model, val_loader, None, device, use_amp)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        (args.experiment_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, args.experiment_dir / "last.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(checkpoint, args.experiment_dir / "best.pt")

    best_ckpt = torch.load(
        args.experiment_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["model"])
    test_metrics = run_epoch(model, test_loader, None, device, use_amp)
    (args.experiment_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Best checkpoint test metrics | test_dice={test_metrics['dice']:.4f} | test_iou={test_metrics['iou']:.4f}"
    )


if __name__ == "__main__":
    main()
