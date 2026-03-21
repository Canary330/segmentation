from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.vlm_backend.label_spaces import (
    PUBLIC_SECOND_TRIMESTER_LABEL_MAP,
    TARGET_A4C_LABELS,
)
from backend.vlm_backend.prompt_templates import sample_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a unified prompt-conditioned segmentation dataset."
    )
    parser.add_argument("--a4c-root", type=Path, default=Path("prepared/a4c13"))
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path("/Users/mico/Downloads/Fetal Echocardiography Second Trimester"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("prepared/prompt_seg"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists() and force:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_binary_mask(mask_array, path: Path) -> None:
    Image.fromarray(mask_array.astype("uint8") * 255, mode="L").save(path)


def render_polygon_mask(width: int, height: int, points: list[list[float]]) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([tuple(point) for point in points], fill=255)
    return mask


def convert_a4c_split(
    split: str,
    a4c_root: Path,
    output_root: Path,
    manifest_lines: list[str],
    rng: random.Random,
) -> None:
    image_dir = a4c_root / split / "images"
    mask_dir = a4c_root / split / "masks"
    out_image_dir = output_root / split / "images"
    out_mask_dir = output_root / split / "masks"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(image_dir.glob("*.png")):
        source_mask = mask_dir / image_path.name
        if not source_mask.exists():
            continue
        image_target = out_image_dir / f"a4c13__{image_path.name}"
        shutil.copy2(image_path, image_target)

        class_mask = Image.open(source_mask).convert("L")
        class_array = __import__("numpy").array(class_mask)
        for class_id, label in enumerate(TARGET_A4C_LABELS, start=1):
            binary_mask = (class_array == class_id)
            if binary_mask.sum() == 0:
                continue
            binary_name = f"{image_path.stem}__{label}.png"
            binary_target = out_mask_dir / f"a4c13__{binary_name}"
            save_binary_mask(binary_mask, binary_target)
            manifest_lines.append(
                json.dumps(
                    {
                        "split": split,
                        "source": "a4c13",
                        "label": label,
                        "prompt": sample_prompt(label, rng),
                        "image_path": str(image_target.resolve()),
                        "mask_path": str(binary_target.resolve()),
                    },
                    ensure_ascii=False,
                )
            )


def split_for_public(video_name: str, frame_index: int) -> str:
    digest = hashlib.md5(f"{video_name}:{frame_index}".encode("utf-8")).hexdigest()
    key = int(digest[:8], 16) % 10
    if key < 7:
        return "training"
    if key < 9:
        return "validation"
    return "testing"


def convert_public_dataset(
    public_root: Path,
    output_root: Path,
    manifest_per_split: dict[str, list[str]],
    rng: random.Random,
) -> None:
    import numpy as np

    for json_path in sorted(public_root.rglob("*.json")):
        if json_path.name.startswith("._"):
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        shapes = data.get("shapes", [])
        if not shapes:
            continue

        video_name = json_path.parent.name
        frame_name = json_path.stem
        split = split_for_public(video_name, int(frame_name))
        out_image_dir = output_root / split / "images"
        out_mask_dir = output_root / split / "masks"
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        image_path = json_path.with_suffix(".jpg")
        if not image_path.exists():
            continue

        image_target = out_image_dir / f"public__{video_name}__{image_path.name}"
        if not image_target.exists():
            shutil.copy2(image_path, image_target)

        width = int(data["imageWidth"])
        height = int(data["imageHeight"])
        grouped_masks: dict[str, np.ndarray] = {}

        for shape in shapes:
            raw_label = shape.get("label")
            canonical = PUBLIC_SECOND_TRIMESTER_LABEL_MAP.get(raw_label)
            if canonical is None:
                continue
            mask = render_polygon_mask(width, height, shape["points"])
            grouped_masks.setdefault(canonical, np.zeros((height, width), dtype=np.uint8))
            grouped_masks[canonical] = np.maximum(grouped_masks[canonical], np.array(mask, dtype=np.uint8))

        for canonical, mask_array in grouped_masks.items():
            if mask_array.max() == 0:
                continue
            binary_name = f"{video_name}__{frame_name}__{canonical}.png"
            binary_target = out_mask_dir / f"public__{binary_name}"
            save_binary_mask(mask_array > 0, binary_target)
            manifest_per_split[split].append(
                json.dumps(
                    {
                        "split": split,
                        "source": "public_second_trimester",
                        "label": canonical,
                        "prompt": sample_prompt(canonical, rng),
                        "image_path": str(image_target.resolve()),
                        "mask_path": str(binary_target.resolve()),
                    },
                    ensure_ascii=False,
                )
            )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    ensure_clean_dir(args.output_root, args.force)
    manifest_per_split = {"training": [], "validation": [], "testing": []}

    for split in ("training", "validation", "testing"):
        convert_a4c_split(split, args.a4c_root, args.output_root, manifest_per_split[split], rng)

    convert_public_dataset(args.public_root, args.output_root, manifest_per_split, rng)

    summary = {}
    for split, lines in manifest_per_split.items():
        unique_lines = []
        seen = set()
        for line in lines:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)
        manifest_path = args.output_root / f"{split}.jsonl"
        manifest_path.write_text("\n".join(unique_lines) + ("\n" if unique_lines else ""), encoding="utf-8")

        counts = {}
        for line in unique_lines:
            record = json.loads(line)
            key = f"{record['source']}::{record['label']}"
            counts[key] = counts.get(key, 0) + 1
        summary[split] = {
            "samples": len(unique_lines),
            "counts": counts,
        }

    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared prompt segmentation dataset at {args.output_root}")


if __name__ == "__main__":
    main()
