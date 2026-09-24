from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch import nn

COLOURS = {"acer": (200, 40, 40), "betula": (40, 180, 60), "quercus": (40, 60, 200)}


@pytest.fixture
def leaf_folder(tmp_path: Path) -> Path:
    """Three visually separable 'species', eight images each, plus junk files."""
    rng = np.random.default_rng(0)
    root = tmp_path / "leaves"
    for name, colour in COLOURS.items():
        folder = root / name
        folder.mkdir(parents=True)
        for i in range(8):
            noise = rng.integers(-20, 20, size=(40, 40, 3))
            pixels = np.clip(np.array(colour) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(folder / f"{i:02d}.png")
        (folder / "notes.txt").write_text("not an image")
        (folder / ".DS_Store").write_bytes(b"\0")
    return root


class MeanColourBranch(nn.Module):
    """Tiny stand-in for a backbone: global average colour -> small projection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(3, dim)

    def forward(self, images):
        return self.project(images.mean(dim=(2, 3)))


@pytest.fixture
def tiny_factory():
    from leafnet.model import HybridClassifier

    def factory(arch: str, num_classes: int):
        branches = {}
        if arch in ("hybrid", "resnet101"):
            branches["cnn"] = (MeanColourBranch(8), 8)
        if arch in ("hybrid", "vit"):
            branches["vit"] = (MeanColourBranch(4), 4)
        return HybridClassifier(branches, num_classes)

    return factory
