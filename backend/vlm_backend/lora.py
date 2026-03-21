from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base + update


def inject_lora(
    module: nn.Module,
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    replaced: list[str] = []
    for parent_name, parent_module in module.named_modules():
        child_items = list(parent_module.named_children())
        for child_name, child_module in child_items:
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child_module, nn.Linear) and child_name.endswith(target_suffixes):
                setattr(
                    parent_module,
                    child_name,
                    LoRALinear(
                        base_layer=child_module,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
                replaced.append(full_name)
    return replaced
