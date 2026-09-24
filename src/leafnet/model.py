"""Hybrid CNN + Vision Transformer classifier and single-branch ablations."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

ARCHITECTURES = ("hybrid", "resnet101", "vit")


class HybridClassifier(nn.Module):
    """Concatenate pooled features from several backbones, then classify linearly.

    Each branch maps an image batch ``(N, 3, H, W)`` to a pooled feature
    vector ``(N, D_i)``. The fused representation has ``sum(D_i)`` channels
    and feeds a single linear layer. With one branch this degenerates to an
    ordinary fine-tuned backbone, which is how the single-model ablations are
    built: every architecture shares the same head and training loop.
    """

    def __init__(
        self,
        branches: Mapping[str, tuple[nn.Module, int]],
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not branches:
            raise ValueError("at least one branch is required")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.branches = nn.ModuleDict({name: module for name, (module, _) in branches.items()})
        self.branch_dims = {name: int(dim) for name, (_, dim) in branches.items()}
        self.feature_dim = sum(self.branch_dims.values())
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.feature_dim, num_classes)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        parts = []
        for name, branch in self.branches.items():
            feature = branch(images)
            if feature.ndim != 2 or feature.shape[1] != self.branch_dims[name]:
                raise RuntimeError(
                    f"branch {name!r} returned shape {tuple(feature.shape)}, "
                    f"expected (N, {self.branch_dims[name]})"
                )
            parts.append(feature)
        return torch.cat(parts, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.features(images)))


def resnet101_branch(pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import ResNet101_Weights, resnet101

    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    dim = model.fc.in_features  # 2048
    model.fc = nn.Identity()
    return model, dim


def vit_branch(pretrained: bool, name: str = "vit_base_patch16_224") -> tuple[nn.Module, int]:
    import timm

    # num_classes=0 removes the classification head and returns the pooled
    # [CLS] embedding (768-D for ViT-Base).
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return model, int(model.num_features)


def build_model(
    arch: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> HybridClassifier:
    """Build ``hybrid`` (ResNet-101 + ViT-B/16) or one of its single-branch ablations."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}; choose from {ARCHITECTURES}")
    branches: dict[str, tuple[nn.Module, int]] = {}
    if arch in ("hybrid", "resnet101"):
        branches["cnn"] = resnet101_branch(pretrained)
    if arch in ("hybrid", "vit"):
        branches["vit"] = vit_branch(pretrained)
    return HybridClassifier(branches, num_classes, dropout=dropout)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
