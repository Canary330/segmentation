from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from backend.pure_visual.mobileunet_fpn import MobileUNetFPN

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileUNet-FPN on the prepared fetal A4C segmentation dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("prepared/a4c13"),
        help="Prepared dataset root created by backend/scripts/prepare_cvat_segmentation.py",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/a4c13_mobileunet_fpn"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fpn-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
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


def load_summary(data_root: Path) -> dict:
    summary_path = data_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run backend/scripts/prepare_cvat_segmentation.py first."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


class A4CSegmentationDataset(Dataset):
    def __init__(self, split_root: Path, image_size: int, train: bool) -> None:
        self.image_dir = split_root / "images"
        self.mask_dir = split_root / "masks"
        self.image_size = image_size
        self.train = train
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.8:
            angle = random.uniform(-12.0, 12.0)
            max_shift = int(round(self.image_size * 0.04))
            translate = (
                random.randint(-max_shift, max_shift),
                random.randint(-max_shift, max_shift),
            )
            scale = random.uniform(0.95, 1.05)
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        if random.random() < 0.3:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.3:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))

        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.train:
            image, mask = self._augment(image, mask)

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor


class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_background: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_background:
            probs = probs[:, 1:]
            one_hot = one_hot[:, 1:]

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


def update_confusion_matrix(
    confusion: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    valid = (targets >= 0) & (targets < num_classes)
    indices = num_classes * targets[valid] + predictions[valid]
    confusion += torch.bincount(indices, minlength=num_classes**2).view(num_classes, num_classes)
    return confusion


def compute_segmentation_metrics(
    confusion: torch.Tensor,
    class_names: Iterable[str],
) -> dict:
    confusion = confusion.float()
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    class_names = list(class_names)

    metrics = {
        "mean_iou_all": iou.mean().item(),
        "mean_dice_all": dice.mean().item(),
        "per_class_iou": {},
        "per_class_dice": {},
    }

    if len(class_names) > 1:
        metrics["mean_iou_fg"] = iou[1:].mean().item()
        metrics["mean_dice_fg"] = dice[1:].mean().item()
    else:
        metrics["mean_iou_fg"] = metrics["mean_iou_all"]
        metrics["mean_dice_fg"] = metrics["mean_dice_all"]

    for idx, class_name in enumerate(class_names):
        metrics["per_class_iou"][class_name] = round(iou[idx].item(), 6)
        metrics["per_class_dice"][class_name] = round(dice[idx].item(), 6)

    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_loss: nn.Module,
    dice_loss: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc="train", leave=False)

    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = 0.5 * ce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    ce_loss: nn.Module,
    dice_loss: nn.Module,
    device: torch.device,
    use_amp: bool,
    class_names: list[str],
) -> dict:
    model.eval()
    total_loss = 0.0
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    progress = tqdm(loader, desc="eval", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = 0.5 * ce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)

        total_loss += loss.item() * images.size(0)
        predictions = torch.argmax(logits, dim=1).cpu()
        confusion = update_confusion_matrix(confusion, predictions, masks.cpu(), num_classes)

    metrics = compute_segmentation_metrics(confusion, class_names)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    args: argparse.Namespace,
    class_names: list[str],
    metrics: dict,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "class_names": class_names,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    summary = load_summary(args.data_root)
    foreground_classes = summary["class_names"]
    class_names = ["background", *foreground_classes]
    num_classes = len(class_names)

    train_dataset = A4CSegmentationDataset(args.data_root / "training", args.img_size, train=True)
    val_dataset = A4CSegmentationDataset(args.data_root / "validation", args.img_size, train=False)
    test_dataset = A4CSegmentationDataset(args.data_root / "testing", args.img_size, train=False)

    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MobileUNetFPN(
        num_classes=num_classes,
        fpn_channels=args.fpn_channels,
        dropout=args.dropout,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = MulticlassDiceLoss(num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_score = -1.0

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            ce_loss=ce_loss,
            dice_loss=dice_loss,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            ce_loss=ce_loss,
            dice_loss=dice_loss,
            device=device,
            use_amp=use_amp,
            class_names=class_names,
        )
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)
        (args.experiment_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mIoU_fg={val_metrics['mean_iou_fg']:.4f} | "
            f"val_mDice_fg={val_metrics['mean_dice_fg']:.4f}"
        )

        latest_metrics = {
            "train_loss": train_loss,
            "validation": val_metrics,
        }
        save_checkpoint(
            args.experiment_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            args,
            class_names,
            latest_metrics,
        )

        if val_metrics["mean_iou_fg"] > best_score:
            best_score = val_metrics["mean_iou_fg"]
            save_checkpoint(
                args.experiment_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                args,
                class_names,
                latest_metrics,
            )

    best_checkpoint = torch.load(
        args.experiment_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        ce_loss=ce_loss,
        dice_loss=dice_loss,
        device=device,
        use_amp=use_amp,
        class_names=class_names,
    )
    (args.experiment_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "Best validation checkpoint test metrics | "
        f"test_mIoU_fg={test_metrics['mean_iou_fg']:.4f} | "
        f"test_mDice_fg={test_metrics['mean_dice_fg']:.4f}"
    )


if __name__ == "__main__":
    main()
