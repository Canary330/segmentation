from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from backend.pure_visual.class_maps import A4C_13_CLASSES
from backend.vlm_backend.model import build_model_from_checkpoint
from backend.vlm_backend.prompt_templates import get_prompts_for_label

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the prompt-driven model on A4C 13-class data.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("prepared/a4c13"))
    parser.add_argument("--split", type=str, default="validation", choices=["training", "validation", "testing"])
    parser.add_argument("--text-encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output", type=Path, default=Path("experiments/vlm_prompt_seg/a4c13_eval.json"))
    return parser.parse_args()


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


def preprocess_image(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0)


def preprocess_mask(mask_path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(mask_path).convert("L")
    mask = TF.resize(mask, [image_size, image_size], interpolation=InterpolationMode.NEAREST)
    return torch.from_numpy(np.array(mask, dtype=np.int64))


def compute_metrics(confusion: torch.Tensor) -> dict:
    confusion = confusion.float()
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    class_names = ["background", *A4C_13_CLASSES]
    return {
        "mean_iou_all": iou.mean().item(),
        "mean_dice_all": dice.mean().item(),
        "mean_iou_fg": iou[1:].mean().item(),
        "mean_dice_fg": dice[1:].mean().item(),
        "per_class_iou": {name: round(iou[idx].item(), 6) for idx, name in enumerate(class_names)},
        "per_class_dice": {name: round(dice[idx].item(), 6) for idx, name in enumerate(class_names)},
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, default_text_encoder=args.text_encoder)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()

    image_dir = args.data_root / args.split / "images"
    mask_dir = args.data_root / args.split / "masks"
    image_paths = sorted(image_dir.glob("*.png"))
    num_classes = len(A4C_13_CLASSES) + 1
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc=f"eval-{args.split}"):
            image_tensor = preprocess_image(image_path, args.img_size).to(device)
            gt_mask = preprocess_mask(mask_dir / image_path.name, args.img_size)

            class_probs = []
            for label in A4C_13_CLASSES:
                prompt = get_prompts_for_label(label)[0]
                logits = model(image_tensor, [prompt])
                probs = torch.sigmoid(logits)[0, 0].cpu()
                class_probs.append(probs)

            stacked = torch.stack(class_probs, dim=0)
            max_probs, pred_idx = stacked.max(dim=0)
            pred_mask = torch.where(max_probs > 0.5, pred_idx + 1, torch.zeros_like(pred_idx))

            gt_flat = gt_mask.reshape(-1)
            pred_flat = pred_mask.reshape(-1)
            indices = num_classes * gt_flat + pred_flat
            confusion += torch.bincount(indices, minlength=num_classes * num_classes).view(num_classes, num_classes)

    metrics = compute_metrics(confusion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
