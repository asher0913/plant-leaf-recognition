"""Training and inference loops shared by every experiment."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import classification_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    amp: bool = False
    grad_clip: float | None = None
    # Evaluate on the held-out split after these epochs (1-based). Used to
    # reproduce the accuracy-versus-epochs study without retraining.
    eval_epochs: Sequence[int] = field(default_factory=tuple)


@dataclass
class TrainResult:
    losses: list[float]
    checkpoints: dict[int, dict[str, float]]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = None,
) -> float:
    model.train()
    total, count = 0.0, 0
    use_amp = scaler is not None
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(model(images), labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += loss.item() * labels.size(0)
        count += labels.size(0)
    return total / max(count, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` for a labelled loader."""
    model.eval()
    targets, predictions = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(np.asarray(labels))
    return np.concatenate(targets), np.concatenate(predictions)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    y_true, y_pred = predict(model, loader, device)
    return classification_metrics(y_true, y_pred)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    *,
    eval_loader: DataLoader | None = None,
    log: Callable[[str], None] | None = None,
) -> TrainResult:
    """Fine-tune every parameter with Adam(W) and cross-entropy.

    All backbone weights are updated, not only the new head: on a small,
    visually narrow domain like scanned leaves the ImageNet features need
    adapting, and the study's single-branch ablations use the same setting.
    """
    model.to(device)
    optimizer: torch.optim.Optimizer
    if config.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler(device.type) if config.amp and device.type == "cuda" else None

    losses: list[float] = []
    checkpoints: dict[int, dict[str, float]] = {}
    wanted = set(config.eval_epochs)
    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler, grad_clip=config.grad_clip
        )
        losses.append(loss)
        message = f"epoch {epoch}/{config.epochs} loss={loss:.4f}"
        if eval_loader is not None and epoch in wanted:
            checkpoints[epoch] = evaluate(model, eval_loader, device)
            message += f" top1={checkpoints[epoch]['top1']:.4f}"
        if log:
            log(message)
    return TrainResult(losses, checkpoints)
