"""Command-line entry point: ``leafnet run | summarize | predict``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import TrainConfig, resolve_device
from .experiment import ExperimentConfig, run_repeated_holdout
from .model import ARCHITECTURES


def _run(args: argparse.Namespace) -> int:
    eval_epochs = tuple(args.eval_epochs or ())
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        arch=args.arch,
        repeats=args.repeats,
        test_size=args.test_size,
        seed=args.seed,
        stratify=not args.no_stratify,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        pretrained=not args.no_pretrained,
        device=args.device,
        save_models=args.save_models,
        train=TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            amp=args.amp,
            eval_epochs=eval_epochs,
        ),
    )
    report = run_repeated_holdout(config)
    _print_summary(report)
    return 0


def _print_summary(report: dict) -> None:
    summary = report["summary"]
    print(f"\n{report['config']['arch']} over {summary['top1']['n']} splits")
    for metric in ("top1", "f1_weighted", "f1_macro"):
        s = summary[metric]
        print(
            f"  {metric:<12} {100 * s['mean']:.2f} ± {100 * s['std']:.2f}  "
            f"(range {100 * s['min']:.2f}–{100 * s['max']:.2f})"
        )


def _summarize(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.results).read_text())
    _print_summary(report)
    return 0


def _predict(args: argparse.Namespace) -> int:
    from .predict import iter_images, load_checkpoint, predict_images

    device = resolve_device(args.device)
    model, class_names, image_size = load_checkpoint(args.checkpoint, device)
    images = iter_images(args.input)
    if not images:
        print(f"no images found under {args.input}", file=sys.stderr)
        return 1
    rows = predict_images(model, class_names, images, image_size, device, top_k=args.top_k)
    print(json.dumps(rows, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leafnet", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="repeated hold-out training and evaluation")
    run.add_argument("--data-dir", required=True, help="root/<species>/<image> folder")
    run.add_argument("--output-dir", default="runs/hybrid")
    run.add_argument("--arch", choices=ARCHITECTURES, default="hybrid")
    run.add_argument("--repeats", type=int, default=20)
    run.add_argument("--test-size", type=float, default=0.3)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-stratify", action="store_true", help="plain random splits")
    run.add_argument("--epochs", type=int, default=10)
    run.add_argument("--eval-epochs", type=int, nargs="*", help="extra test evaluations, e.g. 10 20 30")
    run.add_argument("--lr", type=float, default=1e-4)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--label-smoothing", type=float, default=0.0)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--image-size", type=int, default=224)
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--augment", action="store_true", help="flip/rotate/crop training augmentation")
    run.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    run.add_argument("--no-pretrained", action="store_true")
    run.add_argument("--device", default="auto")
    run.add_argument("--save-models", action="store_true")
    run.set_defaults(func=_run)

    summarize = sub.add_parser("summarize", help="print the summary of a results.json")
    summarize.add_argument("results")
    summarize.set_defaults(func=_summarize)

    predict = sub.add_parser("predict", help="top-k species for an image or folder")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(func=_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
