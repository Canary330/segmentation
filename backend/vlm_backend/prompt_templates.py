from __future__ import annotations

import random

from .label_spaces import CANONICAL_LABEL_METADATA


def get_prompts_for_label(label: str) -> list[str]:
    if label not in CANONICAL_LABEL_METADATA:
        raise KeyError(f"Unknown label: {label}")
    return list(CANONICAL_LABEL_METADATA[label]["prompts"])


def sample_prompt(label: str, rng: random.Random | None = None) -> str:
    prompts = get_prompts_for_label(label)
    rng = rng or random
    return rng.choice(prompts)
