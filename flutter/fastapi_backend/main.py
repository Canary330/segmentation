from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import functional as TF

from pure_visual.class_maps import A4C_13_CLASSES
from vlm_backend.model import build_model_from_checkpoint
from vlm_backend.prompt_templates import get_prompts_for_label


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_CHECKPOINT = os.environ.get("SEGMENTATION_MODEL_PATH", "")
TEXT_ENCODER_NAME = os.environ.get("SEGMENTATION_TEXT_ENCODER", "openai/clip-vit-base-patch32")
IMAGE_SIZE = int(os.environ.get("SEGMENTATION_IMAGE_SIZE", "512"))
SINGLE_PROMPT_THRESHOLD = float(os.environ.get("SEGMENTATION_BINARY_THRESHOLD", "0.5"))
A4C13_MULTICLASS_THRESHOLD = float(os.environ.get("SEGMENTATION_MULTICLASS_THRESHOLD", "0.15"))


class PredictResponse(BaseModel):
    prompt: str
    width: int
    height: int
    foreground_pixels: int
    mask_png_base64: str


class MultiClassPredictResponse(BaseModel):
    width: int
    height: int
    labels: list[str]
    mask_png_base64: str


app = FastAPI(title="Fetal Ultrasound VLM Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: PromptableMobileUNetFPN | None = None
_device: torch.device | None = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model() -> PromptableMobileUNetFPN:
    global _model, _device
    if _model is not None:
        return _model

    if not MODEL_CHECKPOINT:
        raise HTTPException(status_code=500, detail="SEGMENTATION_MODEL_PATH is not configured.")

    checkpoint_path = Path(MODEL_CHECKPOINT)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=500, detail=f"Checkpoint not found: {checkpoint_path}")

    _device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, default_text_encoder=TEXT_ENCODER_NAME)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(_device)
    model.eval()
    _model = model
    return _model


def preprocess_image(image_bytes: bytes) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)
    return tensor, original_size


def encode_mask_to_base64(mask: np.ndarray) -> str:
    import base64

    pil_mask = Image.fromarray(mask.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    pil_mask.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def predict_single_prompt(model: PromptableMobileUNetFPN, tensor: torch.Tensor, prompt: str) -> np.ndarray:
    with torch.no_grad():
        logits = model(tensor, [prompt])
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return probs


def predict_label_with_prompt_ensemble(
    model: PromptableMobileUNetFPN,
    tensor: torch.Tensor,
    label: str,
) -> np.ndarray:
    prompt_probs = [predict_single_prompt(model, tensor, prompt) for prompt in get_prompts_for_label(label)]
    return np.mean(np.stack(prompt_probs, axis=0), axis=0)


@app.get("/hello")
async def hello_world():
    return {"message": "Fetal ultrasound VLM backend is running"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "checkpoint_configured": bool(MODEL_CHECKPOINT),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    prompt: str = Form(...),
):
    model = load_model()
    assert _device is not None

    image_bytes = await image.read()
    tensor, (width, height) = preprocess_image(image_bytes)
    tensor = tensor.to(_device)

    probs = predict_single_prompt(model, tensor, prompt)
    mask = (probs > SINGLE_PROMPT_THRESHOLD).astype(np.uint8) * 255
    mask = np.array(Image.fromarray(mask).resize((width, height)))
    return PredictResponse(
        prompt=prompt,
        width=width,
        height=height,
        foreground_pixels=int((mask > 0).sum()),
        mask_png_base64=encode_mask_to_base64(mask),
    )


@app.post("/predict_13class", response_model=MultiClassPredictResponse)
async def predict_13class(image: UploadFile = File(...)):
    model = load_model()
    assert _device is not None

    image_bytes = await image.read()
    tensor, (width, height) = preprocess_image(image_bytes)
    tensor = tensor.to(_device)

    class_probs = []
    for label in A4C_13_CLASSES:
        class_probs.append(predict_label_with_prompt_ensemble(model, tensor, label))

    stacked = np.stack(class_probs, axis=0)
    best_idx = stacked.argmax(axis=0) + 1
    best_prob = stacked.max(axis=0)
    combined = np.where(best_prob > A4C13_MULTICLASS_THRESHOLD, best_idx, 0).astype(np.uint8)
    combined = np.array(Image.fromarray(combined, mode="L").resize((width, height), resample=Image.NEAREST))

    return MultiClassPredictResponse(
        width=width,
        height=height,
        labels=["background", *A4C_13_CLASSES],
        mask_png_base64=encode_mask_to_base64(combined),
    )
