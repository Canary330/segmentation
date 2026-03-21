from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PromptSegmentationDataset(Dataset):
    def __init__(self, manifest_path: Path, image_size: int = 512, train: bool = True) -> None:
        self.manifest_path = manifest_path
        self.image_size = image_size
        self.train = train
        self.records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"No records found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.8:
            angle = random.uniform(-12.0, 12.0)
            shift = int(round(self.image_size * 0.04))
            translate = (random.randint(-shift, shift), random.randint(-shift, shift))
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")
        mask = Image.open(record["mask_path"]).convert("L")

        if self.train:
            image, mask = self._augment(image, mask)

        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = torch.from_numpy((np.array(mask, dtype=np.uint8) > 0).astype(np.float32)).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "prompt": record["prompt"],
            "label": record["label"],
            "source": record["source"],
            "image_path": record["image_path"],
        }
