"""DamageScope command-line interface."""

from __future__ import annotations

import argparse

import numpy as np

from .config import get_settings
from .generate import make_dataset, make_sample
from .localize import localize, mean_best_iou
from .model import DamageClassifier
from .service import InspectionService


def cmd_train(args) -> None:
    settings = get_settings()
    images, labels, _ = make_dataset(n=args.n, size=settings.image_size, seed=42)
    model = DamageClassifier(size=settings.image_size)
    report = model.train(images, labels)
    path = model.save(settings.model_dir)
    print(f"version {report.version}  metrics {report.metrics}  -> {path}")


def cmd_export(args) -> None:
    settings = get_settings()
    model = DamageClassifier.load(settings.model_dir)
    out = model.export_onnx(f"{settings.model_dir}/model.onnx")
    print(f"exported ONNX model -> {out}")


def cmd_demo(args) -> None:
    settings = get_settings()
    print("1) Training a damage classifier on synthetic panels ...")
    images, labels, samples = make_dataset(n=args.n, size=settings.image_size, seed=42)
    model = DamageClassifier(size=settings.image_size, params={"n_estimators": 150})
    report = model.train(images, labels)
    print(f"   version {report.version}  metrics {report.metrics}")

    service = InspectionService(settings=settings, model=model)

    print("2) Inspecting a damaged panel and a clean panel ...")
    for damaged in (True, False):
        sample = make_sample(seed=9000 + int(damaged), size=settings.image_size, damaged=damaged)
        result = service.inspect(sample.image)
        label = "DAMAGED" if damaged else "CLEAN"
        print(f"   [{label}] -> damaged={result['damaged']} conf={result['confidence']} "
              f"severity={result['severity']} regions={result['n_regions']} "
              f"cost={result['estimated_cost_cop']:,.0f} COP")

    print("3) Localization quality (mean IoU over damaged samples) ...")
    ious = []
    for sample in [s for s in samples if s.label == 1][:30]:
        boxes, _b, _a = localize(sample.image, settings.residual_sigma, settings.min_region_area)
        ious.append(mean_best_iou(boxes, sample.boxes))
    print(f"   mean IoU = {np.mean(ious):.3f} over {len(ious)} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="DamageScope CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train and save the classifier")
    p.add_argument("--n", type=int, default=500)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="export the trained model to ONNX")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("demo", help="train, inspect, and score localization")
    p.add_argument("--n", type=int, default=300)
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
