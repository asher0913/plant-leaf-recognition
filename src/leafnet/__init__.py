"""Hybrid ResNet-101 + ViT-B/16 plant leaf recognition."""

from .data import ImageFolderIndex, LeafDataset, build_transforms, repeated_splits, scan_image_folder
from .metrics import classification_metrics, summarize
from .model import ARCHITECTURES, HybridClassifier, build_model

__all__ = [
    "ARCHITECTURES",
    "HybridClassifier",
    "ImageFolderIndex",
    "LeafDataset",
    "build_model",
    "build_transforms",
    "classification_metrics",
    "repeated_splits",
    "scan_image_folder",
    "summarize",
]
__version__ = "1.0.0"
