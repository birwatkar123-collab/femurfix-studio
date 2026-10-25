#!/usr/bin/env python3
"""Train YOLO11 on the FracAtlas binary fracture dataset and export ONNX.

Examples
--------
GPU (Kaggle/Colab) full run::

    python ml/train/train_yolo.py --device 0 --epochs 50 --imgsz 640 --batch 16

CPU smoke test::

    python ml/train/train_yolo.py --device cpu --epochs 1 --imgsz 320 --batch 8 \
        --fraction 0.02 --name smoke --no-export-onnx --no-eval-test
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from ultralytics import YOLO

ML_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolo11n.pt", help="weights or model YAML")
    parser.add_argument("--data", type=Path, default=ML_ROOT / "data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0 if os.name == "nt" else 8)
    parser.add_argument("--project", type=Path, default=ML_ROOT / "runs")
    parser.add_argument("--name", default="fracatlas_yolo11n")
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of training images to use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", action="store_true", help="cache images in RAM")
    parser.add_argument("--exist-ok", action="store_true", help="reuse an existing run directory")
    parser.add_argument("--export-onnx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-test", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    if not data.is_file():
        raise SystemExit(f"Data YAML not found: {data}")

    if str(args.device).lower() == "cpu":
        print("WARNING: CPU training selected. A full 50-epoch run will take many hours.")
    elif torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        fraction=args.fraction,
        cache=args.cache,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        optimizer="auto",
        amp=True,
        plots=True,
        save=True,
        val=True,
        close_mosaic=min(10, max(0, args.epochs - 1)),
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.0,
        flipud=0.0,
        mosaic=1.0,
        auto_augment=None,
        erasing=0.0,
    )

    trainer = model.trainer
    run_dir = Path(trainer.save_dir)
    best = Path(trainer.best)
    if not best.is_file():
        raise SystemExit(f"Training finished without a checkpoint in {run_dir}")

    summary: dict[str, object] = {"run_dir": str(run_dir), "best_pt": str(best)}

    if args.eval_test:
        metrics = model.val(
            data=str(data),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(args.project.resolve()),
            name=f"{args.name}_test",
            exist_ok=True,
            plots=True,
        )
        summary["test"] = {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        }
        print(f"Test metrics: {summary['test']}")

    if args.export_onnx:
        try:
            onnx_path = YOLO(str(best)).export(
                format="onnx",
                imgsz=args.imgsz,
                opset=12,
                simplify=True,
                dynamic=False,
            )
            summary["onnx"] = str(onnx_path)
            print(f"ONNX exported: {onnx_path}")
        except Exception as exc:  # export failure should not discard a completed training run
            print(f"ONNX export failed (checkpoint is still valid): {exc}")

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")
    if "onnx" in summary:
        print(f"Next: copy {summary['onnx']} to backend/models/femur_fracture.onnx")
    else:
        print("Next: export ONNX (enable --export-onnx) before backend integration.")


if __name__ == "__main__":
    main()
