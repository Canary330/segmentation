from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    description: str
    classes: tuple[str, ...]
    label_aliases: dict[str, str]
    split_to_xml: dict[str, str]
    split_to_dataset_subset: dict[str, str]


A4C_13_CLASSES = (
    "DAO",
    "LA",
    "RA",
    "LV",
    "RV",
    "VS",
    "IS",
    "SP",
    "RB",
    "LVW",
    "RVW",
    "LL",
    "RL",
)

A4C_13_LABEL_ALIASES = {
    "DAO降主动脉": "DAO",
    "DAo": "DAO",
    "LA左心房": "LA",
    "RA右心房": "RA",
    "LV左心室": "LV",
    "RV右心室": "RV",
    "VS室间隔": "VS",
    "IVS": "VS",
    "IS房间隔": "IS",
    "IAS": "IS",
    "SP脊柱": "SP",
    "RB肋骨": "RB",
    "LVW左心室壁": "LVW",
    "RVW右心室壁": "RVW",
    "LL左肺": "LL",
    "RL右肺": "RL",
}

A4C_13_POLY = DatasetPreset(
    name="a4c13_poly",
    description="FOCUS A4C 13-class setup based on the polygon annotations in training2/validation2/testing2.xml.",
    classes=A4C_13_CLASSES,
    label_aliases=A4C_13_LABEL_ALIASES,
    split_to_xml={
        "training": "标注/training2.xml",
        "validation": "标注/validation2.xml",
        "testing": "标注/testing2.xml",
    },
    split_to_dataset_subset={
        "training": "training",
        "validation": "validation",
        "testing": "testing",
    },
)

PRESETS = {
    A4C_13_POLY.name: A4C_13_POLY,
}


def build_label_to_id(classes: tuple[str, ...]) -> dict[str, int]:
    return {label: index for index, label in enumerate(classes, start=1)}


def canonicalize_label(raw_label: str, preset: DatasetPreset) -> str | None:
    if raw_label in preset.classes:
        return raw_label
    return preset.label_aliases.get(raw_label)
