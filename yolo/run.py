#!/usr/bin/env python3
"""Unified CLI to train any YOLO checkpoint listed in ``models/``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backends import DEFAULT_OUTPUT_ROOT, run_train
from registry import discover_models, list_models, list_regression_models, resolve_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YOLO checkpoints on mini VOC (or a custom dataset YAML)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train one checkpoint (model id = weights stem).")
    train_p.add_argument(
        "--model",
        required=True,
        help="Model id, e.g. yolov5n, yolov8s-seg, yolov10n (see 'models').",
    )
    train_p.add_argument("--data", type=Path, default=None, help="Dataset YAML override.")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--device", default="0", help="cuda device id, 'cpu', or 'cuda'.")
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--weights", default=None, help="Override initial weights path.")
    train_p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    models_p = sub.add_parser("models", help="List discovered checkpoints.")
    models_p.add_argument(
        "--trainable-only",
        action="store_true",
        help="Only list models with a training backend in this repo.",
    )

    sub.add_parser("backends", help="Summarize training backends and model counts.")

    return parser


def _cmd_models(trainable_only: bool) -> int:
    items = list_models(trainable_only=trainable_only)
    for model_id, desc in items:
        print(f"{model_id:20}  {desc}")
    print(f"\nTotal: {len(items)}", file=sys.stderr)
    return 0


def _cmd_backends() -> int:
    catalog = discover_models()
    counts: dict[str, int] = {}
    for spec in catalog.values():
        key = spec.backend.value
        counts[key] = counts.get(key, 0) + 1
    print("Training backends:")
    for backend, count in sorted(counts.items()):
        print(f"  {backend:16}  {count} checkpoint(s)")
    trainable = sum(1 for s in catalog.values() if s.trainable)
    print(f"\nTrainable: {trainable} / {len(catalog)} checkpoints in models/")
    print(f"Regression default set: {', '.join(list_regression_models())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)

    if args.command == "models":
        return _cmd_models(args.trainable_only)

    if args.command == "backends":
        return _cmd_backends()

    if args.command == "train":
        try:
            spec = resolve_model(args.model)
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        print(f"[INFO] Training {spec.model_id} via {spec.backend.value}", file=sys.stderr)
        return run_train(
            args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            device=args.device,
            batch_size=args.batch_size,
            weights=args.weights,
            output_root=args.output_root,
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
