"""Repeated stratified hold-out evaluation with resumable result files."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import LeafDataset, build_transforms, repeated_splits, scan_image_folder
from .engine import TrainConfig, evaluate, fit, resolve_device, seed_everything
from .metrics import summarize
from .model import build_model, count_parameters

ModelFactory = Callable[[str, int], nn.Module]


@dataclass
class ExperimentConfig:
    data_dir: str
    output_dir: str = "runs/hybrid"
    arch: str = "hybrid"
    repeats: int = 20
    test_size: float = 0.3
    seed: int = 0
    stratify: bool = True
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    augment: bool = False
    pretrained: bool = True
    device: str = "auto"
    save_models: bool = False
    train: TrainConfig = field(default_factory=TrainConfig)


def _default_factory(pretrained: bool) -> ModelFactory:
    def factory(arch: str, num_classes: int) -> nn.Module:
        return build_model(arch, num_classes, pretrained=pretrained)

    return factory


def _load_existing(path: Path, config: ExperimentConfig) -> list[dict]:
    if not path.exists():
        return []
    previous = json.loads(path.read_text())
    keys = ("arch", "test_size", "seed", "stratify", "image_size")
    if any(previous["config"].get(k) != getattr(config, k) for k in keys):
        raise ValueError(f"{path} was produced with a different protocol; use a new --output-dir")
    return previous["splits"]


def run_repeated_holdout(
    config: ExperimentConfig,
    *,
    model_factory: ModelFactory | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Train a fresh model on each split and score it on that split's test set.

    Results are written after every split. Re-running with the same
    ``output_dir`` skips splits that already finished, so a long 20-split
    campaign survives a preempted GPU job.
    """
    index = scan_image_folder(config.data_dir)
    splits = repeated_splits(
        index.labels,
        repeats=config.repeats,
        test_size=config.test_size,
        seed=config.seed,
        stratify=config.stratify,
    )
    device = resolve_device(config.device)
    factory = model_factory or _default_factory(config.pretrained)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    completed = _load_existing(results_path, config)
    done = {row["split"] for row in completed}

    eval_tf = build_transforms(config.image_size, augment=False)
    train_tf = build_transforms(config.image_size, augment=config.augment)
    loader_kwargs = {"batch_size": config.batch_size, "num_workers": config.num_workers}
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True

    log(f"{index.num_classes} classes, {len(index.paths)} images, {config.repeats} splits, device={device}")
    parameters = None
    for split_id, (train_idx, test_idx) in enumerate(splits):
        if split_id in done:
            log(f"split {split_id}: already complete, skipping")
            continue
        seed_everything(config.seed + split_id)
        paths, labels = index.paths, index.labels
        train_set = LeafDataset([paths[i] for i in train_idx], [labels[i] for i in train_idx], train_tf)
        test_set = LeafDataset([paths[i] for i in test_idx], [labels[i] for i in test_idx], eval_tf)
        generator = torch.Generator().manual_seed(config.seed + split_id)
        train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_kwargs)
        test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

        model = factory(config.arch, index.num_classes)
        parameters = count_parameters(model)
        started = time.perf_counter()
        trained = fit(model, train_loader, device, config.train, eval_loader=test_loader, log=log)
        metrics = evaluate(model, test_loader, device)
        elapsed = time.perf_counter() - started
        log(
            f"split {split_id}: top1={metrics['top1']:.4f} "
            f"f1_weighted={metrics['f1_weighted']:.4f} ({elapsed:.0f}s)"
        )
        completed.append(
            {
                "split": split_id,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                **metrics,
                "losses": trained.losses,
                "checkpoints": {str(k): v for k, v in trained.checkpoints.items()},
                "seconds": elapsed,
            }
        )
        if config.save_models:
            torch.save(
                {
                    "arch": config.arch,
                    "class_names": list(index.class_names),
                    "image_size": config.image_size,
                    "state_dict": model.state_dict(),
                },
                out_dir / f"model_split{split_id:02d}.pt",
            )
        _write(results_path, config, index.class_names, completed, parameters)

    return _write(results_path, config, index.class_names, completed, parameters)


def _write(path: Path, config: ExperimentConfig, class_names, rows: list[dict], parameters) -> dict:
    rows = sorted(rows, key=lambda row: row["split"])
    report = {
        "config": {**asdict(config), "train": asdict(config.train)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "num_classes": len(class_names),
        "parameters": parameters,
        "splits": rows,
        "summary": {
            metric: summarize([row[metric] for row in rows]) for metric in ("top1", "f1_weighted", "f1_macro")
        }
        if rows
        else {},
    }
    path.write_text(json.dumps(report, indent=2))
    return report
