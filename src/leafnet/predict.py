"""Top-k inference from a saved checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .data import IMAGE_EXTENSIONS, build_transforms
from .model import build_model


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["arch"], len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["class_names"], checkpoint.get("image_size", 224)


def iter_images(target: str | Path) -> list[Path]:
    target = Path(target)
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


@torch.no_grad()
def predict_images(model, class_names, images: list[Path], image_size: int, device, top_k: int = 5):
    transform = build_transforms(image_size, augment=False)
    rows = []
    for path in images:
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        scores, indices = probabilities.topk(min(top_k, len(class_names)))
        rows.append(
            {
                "image": str(path),
                "predictions": [
                    {"class": class_names[i], "probability": round(float(s), 4)}
                    for s, i in zip(scores.tolist(), indices.tolist(), strict=True)
                ],
            }
        )
    return rows
