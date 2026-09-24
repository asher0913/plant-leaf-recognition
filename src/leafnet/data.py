"""Dataset discovery, deterministic label mapping, transforms and repeated splits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ImageFolderIndex:
    """Every image under a ``root/<class>/<image>`` tree, with a stable label order."""

    paths: tuple[Path, ...]
    labels: tuple[int, ...]
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(np.asarray(self.labels), minlength=self.num_classes)
        return {name: int(count) for name, count in zip(self.class_names, counts, strict=True)}


def scan_image_folder(root: str | Path) -> ImageFolderIndex:
    """Index an image-per-class folder tree.

    Class folders and files are sorted, so the same dataset produces the same
    label mapping on every operating system. Hidden files and non-image files
    are skipped rather than crashing the loader halfway through an epoch.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root}")

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(class_dirs) < 2:
        raise ValueError(f"expected at least two class folders under {root}")

    paths: list[Path] = []
    labels: list[int] = []
    for label, class_dir in enumerate(class_dirs):
        images = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"class folder has no images: {class_dir}")
        paths.extend(images)
        labels.extend([label] * len(images))

    return ImageFolderIndex(tuple(paths), tuple(labels), tuple(p.name for p in class_dirs))


class LeafDataset(Dataset):
    """Loads RGB images lazily and applies a torchvision transform."""

    def __init__(self, paths: Sequence[Path], labels: Sequence[int], transform=None) -> None:
        if len(paths) != len(labels):
            raise ValueError("paths and labels must have the same length")
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def build_transforms(image_size: int = 224, *, augment: bool = False):
    """Return the preprocessing pipeline.

    Without augmentation this is exactly the reported protocol: resize to a
    square input and normalise with ImageNet statistics, which is what both
    pretrained backbones expect. The optional training augmentation uses
    flips and small rotations because a leaf's identity does not depend on
    its orientation on the scanner.
    """
    normalise = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not augment:
        return transforms.Compose([transforms.Resize((image_size, image_size)), *normalise])
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            *normalise,
        ]
    )


def repeated_splits(
    labels: Sequence[int],
    *,
    repeats: int,
    test_size: float = 0.3,
    seed: int = 0,
    stratify: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated random train/test partitions (Monte Carlo cross-validation).

    Split ``i`` is drawn with ``random_state=seed + i``, so any single split
    can be regenerated independently of the others. Stratification keeps the
    per-class train/test ratio fixed, which matters when a class has only a
    dozen images.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")

    y = np.asarray(labels)
    placeholder = np.zeros(len(y))
    splits = []
    for i in range(repeats):
        splitter_cls = StratifiedShuffleSplit if stratify else ShuffleSplit
        splitter = splitter_cls(n_splits=1, test_size=test_size, random_state=seed + i)
        train_idx, test_idx = next(splitter.split(placeholder, y if stratify else None))
        splits.append((np.sort(train_idx), np.sort(test_idx)))
    return splits
