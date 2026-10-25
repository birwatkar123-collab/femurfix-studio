#!/usr/bin/env python3
"""Score the local ONNX detector on a FracAtlas split at deployment thresholds."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ml_detector import OnnxFractureDetector  # noqa: E402
from app.utils.config import LOCAL_ONNX_MODEL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


def load_labels(path: Path) -> np.ndarray:
    boxes = []
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                _, cx, cy, w, h = map(float, parts)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model", type=Path, default=LOCAL_ONNX_MODEL)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    detector = OnnxFractureDetector(args.model, conf=0.01)
    images_dir = PROJECT_ROOT / "ml" / "data" / "images" / args.split
    labels_dir = PROJECT_ROOT / "ml" / "data" / "labels" / args.split
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    samples = []
    for image_path in image_paths:
        image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
        tensor, scale, pad = detector._preprocess(image)
        pred = detector.session.run(None, {detector.input_name: tensor})[0]
        boxes = detector._postprocess(pred, image.shape[:2], scale, pad)
        labels = load_labels(labels_dir / f"{image_path.stem}.txt")
        samples.append((boxes, labels))

    positives = sum(1 for _, labels in samples if len(labels))
    total_boxes = sum(len(labels) for _, labels in samples)
    print(f"split={args.split} images={len(samples)} positives={positives} boxes={total_boxes}")

    best = None
    for threshold in THRESHOLDS:
        img_tp = img_fp = img_fn = 0
        box_tp = box_fp = box_fn_total = 0
        for boxes, labels in samples:
            kept = [b for b in boxes if b["confidence"] >= threshold]
            if labels.size and kept:
                img_tp += 1
            elif labels.size:
                img_fn += 1
            elif kept:
                img_fp += 1

            used = set()
            for box in sorted(kept, key=lambda b: -b["confidence"]):
                overlaps = [iou(box["bbox_norm"], label) for label in labels]
                match = int(np.argmax(overlaps)) if overlaps else -1
                if match >= 0 and overlaps[match] >= args.iou and match not in used:
                    used.add(match)
                    box_tp += 1
                else:
                    box_fp += 1
            box_fn = len(labels) - len(used)
            box_fn_total += box_fn

        precision = box_tp / max(1, box_tp + box_fp)
        recall = box_tp / max(1, box_tp + box_fn_total)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        img_precision = img_tp / max(1, img_tp + img_fp)
        img_recall = img_tp / max(1, img_tp + img_fn)
        print(
            f"conf {threshold:.2f} | box P {precision:.3f} R {recall:.3f} F1 {f1:.3f}"
            f" | image P {img_precision:.3f} R {img_recall:.3f}"
            f" | TP {box_tp} FP {box_fp} FN {box_fn_total}"
        )
        if best is None or f1 > best[1]:
            best = (threshold, f1)

    print(f"best F1 threshold on {args.split}: {best[0]:.2f} (F1 {best[1]:.3f})")


if __name__ == "__main__":
    main()
