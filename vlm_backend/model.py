from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from transformers import AutoTokenizer, CLIPTextModel

from .lora import inject_lora


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TextGate(nn.Module):
    def __init__(self, text_dim: int, channels: int) -> None:
        super().__init__()
        self.to_scale = nn.Linear(text_dim, channels)
        self.to_bias = nn.Linear(text_dim, channels)

    def forward(self, feature_map: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        scale = self.to_scale(text_embedding).unsqueeze(-1).unsqueeze(-1)
        bias = self.to_bias(text_embedding).unsqueeze(-1).unsqueeze(-1)
        return feature_map * (1.0 + torch.tanh(scale)) + bias


class DecoderBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(channels * 2, channels),
            DepthwiseSeparableConv(channels, channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class PromptableMobileUNetFPN(nn.Module):
    def __init__(
        self,
        text_encoder_name: str = "openai/clip-vit-base-patch32",
        fpn_channels: int = 128,
        dropout: float = 0.1,
        image_pretrained: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if image_pretrained else None
        backbone = mobilenet_v2(weights=weights).features

        self.stage1 = nn.Sequential(*backbone[:2])
        self.stage2 = nn.Sequential(*backbone[2:4])
        self.stage3 = nn.Sequential(*backbone[4:7])
        self.stage4 = nn.Sequential(*backbone[7:14])
        self.stage5 = nn.Sequential(*backbone[14:19])

        self.lateral1 = ConvBNAct(16, fpn_channels)
        self.lateral2 = ConvBNAct(24, fpn_channels)
        self.lateral3 = ConvBNAct(32, fpn_channels)
        self.lateral4 = ConvBNAct(96, fpn_channels)
        self.lateral5 = ConvBNAct(1280, fpn_channels)

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
        if freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

        self.lora_modules = inject_lora(
            self.text_encoder,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        hidden_size = self.text_encoder.config.hidden_size
        self.text_projection = nn.Linear(hidden_size, fpn_channels)

        self.gate1 = TextGate(fpn_channels, fpn_channels)
        self.gate2 = TextGate(fpn_channels, fpn_channels)
        self.gate3 = TextGate(fpn_channels, fpn_channels)
        self.gate4 = TextGate(fpn_channels, fpn_channels)
        self.gate5 = TextGate(fpn_channels, fpn_channels)

        self.decoder4 = DecoderBlock(fpn_channels)
        self.decoder3 = DecoderBlock(fpn_channels)
        self.decoder2 = DecoderBlock(fpn_channels)
        self.decoder1 = DecoderBlock(fpn_channels)

        self.head = nn.Sequential(
            DepthwiseSeparableConv(fpn_channels, fpn_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(fpn_channels, 1, kernel_size=1),
        )

    def encode_text(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.text_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.text_encoder(**tokens)
        pooled = outputs.pooler_output
        return self.text_projection(pooled)

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        text_embedding = self.encode_text(prompts, images.device)

        s1 = self.stage1(images)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        p5 = self.gate5(self.lateral5(s5), text_embedding)
        p4 = self.gate4(self.lateral4(s4), text_embedding) + self._upsample_like(p5, s4)
        p3 = self.gate3(self.lateral3(s3), text_embedding) + self._upsample_like(p4, s3)
        p2 = self.gate2(self.lateral2(s2), text_embedding) + self._upsample_like(p3, s2)
        p1 = self.gate1(self.lateral1(s1), text_embedding) + self._upsample_like(p2, s1)

        d4 = self.decoder4(p5, p4)
        d3 = self.decoder3(d4, p3)
        d2 = self.decoder2(d3, p2)
        d1 = self.decoder1(d2, p1)

        logits = self.head(F.interpolate(d1, size=images.shape[-2:], mode="bilinear", align_corners=False))
        return logits

    def load_visual_weights(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> tuple[list[str], list[str]]:
        """Load the overlapping visual branch weights from a pure-visual checkpoint."""
        own_state = self.state_dict()
        filtered = {}
        for key, value in state_dict.items():
            if key.startswith("text_") or key.startswith("gate"):
                continue
            if key not in own_state:
                continue
            if own_state[key].shape != value.shape:
                continue
            filtered[key] = value
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        if strict and missing:
            raise RuntimeError(f"Missing visual keys when loading pure-visual weights: {missing}")
        return missing, unexpected


def build_model_from_checkpoint(
    checkpoint: dict,
    default_text_encoder: str = "openai/clip-vit-base-patch32",
) -> PromptableMobileUNetFPN:
    args = checkpoint.get("args", {})
    return PromptableMobileUNetFPN(
        text_encoder_name=args.get("text_encoder", default_text_encoder),
        fpn_channels=int(args.get("fpn_channels", 128)),
        dropout=float(args.get("dropout", 0.1)),
        lora_rank=int(args.get("lora_rank", 8)),
        lora_alpha=float(args.get("lora_alpha", 16.0)),
        lora_dropout=float(args.get("lora_dropout", 0.05)),
    )


def load_pure_visual_checkpoint_into_prompt_model(
    model: PromptableMobileUNetFPN,
    checkpoint: dict,
) -> tuple[list[str], list[str]]:
    pure_state = checkpoint.get("model", checkpoint)
    return model.load_visual_weights(pure_state)
