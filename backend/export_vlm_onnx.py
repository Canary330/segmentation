from __future__ import annotations

import argparse
from pathlib import Path

import torch

from backend.vlm_backend.model import PromptableMobileUNetFPN, build_model_from_checkpoint


class OnnxPromptableWrapper(torch.nn.Module):
    def __init__(self, model: PromptableMobileUNetFPN) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        prompts = ["segment the fetal left ventricle"] * images.shape[0]
        return self.model(images, prompts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the VLM backend to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/vlm_prompt_seg.onnx"))
    parser.add_argument("--text-encoder", type=str, default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint, default_text_encoder=args.text_encoder)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    wrapper = OnnxPromptableWrapper(model)
    dummy = torch.randn(1, 3, 512, 512)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
