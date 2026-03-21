from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pure_visual.class_maps import PRESETS, build_label_to_id, canonicalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CVAT XML annotations into semantic masks for MobileUNet-FPN training."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="a4c13_poly",
        choices=sorted(PRESETS.keys()),
        help="Dataset preset to prepare.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("FOCUS-dataset"),
        help="Root directory containing training/validation/testing images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prepared/a4c13"),
        help="Output directory for converted images, masks, and metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output directory before regenerating derived data.",
    )
    return parser.parse_args()


def parse_points(raw_points: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for pair in raw_points.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_value, y_value = pair.split(",")
        points.append((float(x_value), float(y_value)))
    return points


def decode_cvat_rle(rle_values: str, width: int, height: int) -> Image.Image:
    flat = [0] * (width * height)
    cursor = 0
    current = 0
    for value in rle_values.split(","):
        run = int(value.strip())
        if run <= 0:
            continue
        end = min(cursor + run, len(flat))
        if current == 1:
            for index in range(cursor, end):
                flat[index] = 255
        cursor = end
        current = 1 - current
    return Image.frombytes("L", (width, height), bytes(flat))


def render_mask(
    image_tag: ET.Element,
    label_to_id: dict[str, int],
    preset_name: str,
) -> tuple[Image.Image, dict[str, int], list[str]]:
    width = int(image_tag.get("width"))
    height = int(image_tag.get("height"))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    instance_counts: dict[str, int] = {}
    ignored_labels: set[str] = set()

    preset = PRESETS[preset_name]

    for polygon in image_tag.findall("polygon"):
        raw_label = polygon.get("label", "")
        canonical_label = canonicalize_label(raw_label, preset)
        if canonical_label is None:
            ignored_labels.add(raw_label)
            continue
        points = parse_points(polygon.get("points", ""))
        if len(points) >= 3:
            draw.polygon(points, fill=label_to_id[canonical_label])
            instance_counts[canonical_label] = instance_counts.get(canonical_label, 0) + 1

    for mask_tag in image_tag.findall("mask"):
        raw_label = mask_tag.get("label", "")
        canonical_label = canonicalize_label(raw_label, preset)
        if canonical_label is None:
            ignored_labels.add(raw_label)
            continue

        crop_width = int(mask_tag.get("width"))
        crop_height = int(mask_tag.get("height"))
        left = int(mask_tag.get("left"))
        top = int(mask_tag.get("top"))
        crop_mask = decode_cvat_rle(mask_tag.get("rle", ""), crop_width, crop_height)
        fill = Image.new("L", crop_mask.size, color=label_to_id[canonical_label])
        mask.paste(fill, (left, top), crop_mask)
        instance_counts[canonical_label] = instance_counts.get(canonical_label, 0) + 1

    for box in image_tag.findall("box"):
        raw_label = box.get("label", "")
        canonical_label = canonicalize_label(raw_label, preset)
        if canonical_label is None:
            ignored_labels.add(raw_label)
            continue
        xtl = float(box.get("xtl"))
        ytl = float(box.get("ytl"))
        xbr = float(box.get("xbr"))
        ybr = float(box.get("ybr"))
        draw.rectangle([xtl, ytl, xbr, ybr], fill=label_to_id[canonical_label])
        instance_counts[canonical_label] = instance_counts.get(canonical_label, 0) + 1

    return mask, instance_counts, sorted(ignored_labels)


def prepare_output_root(output_root: Path, force: bool) -> None:
    if output_root.exists() and force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def convert_split(
    split_name: str,
    xml_path: Path,
    image_root: Path,
    output_root: Path,
    label_to_id: dict[str, int],
    preset_name: str,
) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_tags = root.findall("image")

    split_output = output_root / split_name
    images_output = split_output / "images"
    masks_output = split_output / "masks"
    images_output.mkdir(parents=True, exist_ok=True)
    masks_output.mkdir(parents=True, exist_ok=True)

    label_names = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    pixel_counts = {label: 0 for label in label_names}
    images_with_class = {label: 0 for label in label_names}
    instance_counts = {label: 0 for label in label_names}
    ignored_labels: set[str] = set()
    missing_images: list[str] = []

    for image_tag in image_tags:
        image_name = image_tag.get("name")
        source_image = image_root / image_name
        if not source_image.exists():
            missing_images.append(image_name)
            continue

        derived_mask, mask_instances, image_ignored_labels = render_mask(
            image_tag=image_tag,
            label_to_id=label_to_id,
            preset_name=preset_name,
        )
        ignored_labels.update(image_ignored_labels)

        histogram = derived_mask.histogram()
        for label_name, class_id in label_to_id.items():
            pixels = histogram[class_id]
            pixel_counts[label_name] += pixels
            if pixels > 0:
                images_with_class[label_name] += 1
            instance_counts[label_name] += mask_instances.get(label_name, 0)

        shutil.copy2(source_image, images_output / image_name)
        derived_mask.save(masks_output / image_name)

    return {
        "xml_path": str(xml_path),
        "image_root": str(image_root),
        "images": len(image_tags) - len(missing_images),
        "pixel_counts": pixel_counts,
        "images_with_class": images_with_class,
        "instances": instance_counts,
        "ignored_labels": sorted(ignored_labels),
        "missing_images": missing_images,
    }


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_root.resolve()

    prepare_output_root(output_root, force=args.force)

    label_to_id = build_label_to_id(preset.classes)
    summary = {
        "preset": preset.name,
        "description": preset.description,
        "class_names": list(preset.classes),
        "class_to_id": label_to_id,
        "splits": {},
    }

    for split_name, relative_xml in preset.split_to_xml.items():
        xml_path = ROOT / relative_xml
        image_root = dataset_root / preset.split_to_dataset_subset[split_name] / "images"
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {xml_path}")
        if not image_root.exists():
            raise FileNotFoundError(f"Missing image directory: {image_root}")

        split_summary = convert_split(
            split_name=split_name,
            xml_path=xml_path,
            image_root=image_root,
            output_root=output_root,
            label_to_id=label_to_id,
            preset_name=preset.name,
        )
        summary["splits"][split_name] = split_summary

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared dataset written to {output_root}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
