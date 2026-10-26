#!/usr/bin/env python3
"""Measure ROI-cropped detection vs whole-image detection on a FracAtlas split.

Hypothesis: Feeding the detector a zoom-in of the segmented bone instead of the
whole image makes the small fracture boxes (median 41 px at full frame) several
times larger after letterboxing, improving recall.

For every image the femur/bone silhouette is segmented with the production
`XRayAnalyzer._segment_bone`, a margin-padded crop is taken, run through the
same ONNX detector, and detections are mapped back to the original frame so
box-level metrics stay comparable with ground truth.

Usage:
    python backend/tests/eval_roi_crop.py --split test
    python backend/tests/eval_roi_crop.py --split val --margin 0.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ml_detector import OnnxFractureDetector  # noqa: E402
from app.services.xray_analyzer import XRayAnalyzer  # noqa: E402
from app.utils.config import LOCAL_ONNX_MODEL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]


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


def segment_bbox(image: np.ndarray, analyzer: XRayAnalyzer, margin: float) -> Optional[Tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) of a padded bone-mask crop, or None when segmentation fails."""
    try:
        mask = analyzer._segment_bone(image)
    except Exception:
        mask = None
    if mask is None or not np.any(mask):
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    height, width = image.shape[:2]
    pad_y = int(max(1.0, (rows[-1] - rows[0]) * margin))
    pad_x = int(max(1.0, (cols[-1] - cols[0]) * margin))
    x0 = max(0, int(cols[0]) - pad_x)
    x1 = min(width, int(cols[-1]) + 1 + pad_x)
    y0 = max(0, int(rows[0]) - pad_y)
    y1 = min(height, int(rows[-1]) + 1 + pad_y)
    if (y1 - y0) < 16 or (x1 - x0) < 16:
        return None
    return (x0, y0, x1, y1)


def detections_in_frame(det: OnnxFractureDetector, image: np.ndarray, frame: Optional[Tuple[int, int, int, int]]):
    """All detector boxes for `image`; if `frame` given, map them back to the full image."""
    tensor, scale, pad = det._preprocess(image)
    pred = det.session.run(None, {det.input_name: tensor})[0]
    boxes = det._postprocess(pred, image.shape[:2], scale, pad)
    if frame is None:
        return boxes
    x0, y0, x1, y1 = frame
    cw, ch = x1 - x0, y1 - y0
    height, width = image.shape[:2]
    mapped = []
    for box in boxes:
        nx0, ny0, nx1, ny1 = box["bbox_norm"]
        mapped.append({
            "confidence": box["confidence"],
            "bbox_norm": [
                (nx0 * cw + x0) / width,
                (ny0 * ch + y0) / height,
                (nx1 * cw + x0) / width,
                (ny1 * ch + y0) / height,
            ],
        })
    return mapped


class Sweep:
    """Accumulates per-image predictions vs GT across a threshold sweep."""

    def __init__(self, iou_thr: float) -> None:
        self.iou_thr = iou_thr
        # per image: (scores sorted desc, boxes) vs gt boxes
        self.images: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add(self, scores: List[float], boxes: np.ndarray, gt: np.ndarray) -> None:
        order = np.argsort(scores)[::-1]
        self.images.append((
            np.asarray(scores)[order],
            boxes[order] if len(order) else boxes.reshape(0, 4),
            gt,
        ))

    def metrics(self, conf: float) -> Dict[str, float]:
        tp_img = fp_img = fn_img = 0
        tp_box = fp_box = fn_box = 0
        for scores, boxes, gt in self.images:
            keep = scores >= conf
            preds = boxes[keep] if keep.any() else boxes[:0]
            has_pos = len(gt) > 0
            if len(preds) > 0:
                if has_pos:
                    tp_img += 1
                else:
                    fp_img += 1
            elif has_pos:
                fn_img += 1
            # box-level greedy matching
            matched = np.zeros(len(gt), dtype=bool)
            for p in preds:
                best_i, best_v = -1, 0.0
                for gi in range(len(gt)):
                    if matched[gi]:
                        continue
                    v = iou(p, gt[gi])
                    if v > best_v:
                        best_v, best_i = v, gi
                if best_i >= 0 and best_v >= self.iou_thr:
                    matched[best_i] = True
                    tp_box += 1
                else:
                    fp_box += 1
            fn_box += int((~matched).sum())
        return {
            "box_P": tp_box / (tp_box + fp_box) if tp_box + fp_box else 0.0,
            "box_R": tp_box / (tp_box + fn_box) if tp_box + fn_box else 0.0,
            "box_F1": 2 * tp_box / (2 * tp_box + fp_box + fn_box) if 2 * tp_box + fp_box + fn_box else 0.0,
            "img_P": tp_img / (tp_img + fp_img) if tp_img + fp_img else 0.0,
            "img_R": tp_img / (tp_img + fn_img) if tp_img + fn_img else 0.0,
            "img_F1": 2 * tp_img / (2 * tp_img + fp_img + fn_img) if 2 * tp_img + fp_img + fn_img else 0.0,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--margin", type=float, default=0.15, help="crop padding as fraction of mask extent")
    ap.add_argument("--model", default=str(LOCAL_ONNX_MODEL))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="only first N images (debug)")
    args = ap.parse_args()

    img_dir = PROJECT_ROOT / "ml" / "data" / "images" / args.split
    lbl_dir = PROJECT_ROOT / "ml" / "data" / "labels" / args.split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        images = images[: args.limit]

    det = OnnxFractureDetector(model_path=args.model, conf=0.01)
    analyzer = XRayAnalyzer()
    whole_sweep = Sweep(args.iou)
    roi_sweep = Sweep(args.iou)
    roi_hits = 0
    crop_shrink = []

    for idx, img_path in enumerate(images):
        image = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
        gt = load_labels(lbl_dir / (img_path.stem + ".txt"))
        frame = segment_bbox(image, analyzer, args.margin)
        if frame is not None:
            roi_hits += 1
            x0, y0, x1, y1 = frame
            crop_shrink.append((x1 - x0) * (y1 - y0) / (image.shape[0] * image.shape[1]))
            crop = image[y0:y1, x0:x1]
            roi_dets = detections_in_frame(det, crop, frame)
        else:
            roi_dets = []  # no segmentation -> ROI variant abstains; counted as missed positives
            crop_shrink.append(1.0)
        whole_dets = detections_in_frame(det, image, None)

        ws = [float(d["confidence"]) for d in whole_dets]
        wb = np.array([d["bbox_norm"] for d in whole_dets], dtype=np.float32).reshape(-1, 4)
        whole_sweep.add(ws, wb, gt)
        rs = [float(d["confidence"]) for d in roi_dets]
        rb = np.array([d["bbox_norm"] for d in roi_dets], dtype=np.float32).reshape(-1, 4)
        roi_sweep.add(rs, rb, gt)

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(images)} processed", flush=True)

    print(f"\nSplit: {args.split} | {len(images)} images | ROI segmentation usable on {roi_hits} "
          f"({100 * roi_hits / len(images):.0f}%), mean crop area {100 * np.mean(crop_shrink):.0f}% of full frame")

    print(f"\nWhole-image (baseline):")
    print("conf |  box P / R / F1      |  img P / R / F1")
    for t in THRESHOLDS:
        m = whole_sweep.metrics(t)
        print(f"{t:.2f} | {m['box_P']:.3f} / {m['box_R']:.3f} / {m['box_F1']:.3f} | "
              f"{m['img_P']:.3f} / {m['img_R']:.3f} / {m['img_F1']:.3f}")

    print(f"\nROI-cropped (margin {args.margin}):")
    print("conf |  box P / R / F1      |  img P / R / F1")
    for t in THRESHOLDS:
        m = roi_sweep.metrics(t)
        print(f"{t:.2f} | {m['box_P']:.3f} / {m['box_R']:.3f} / {m['box_F1']:.3f} | "
              f"{m['img_P']:.3f} / {m['img_R']:.3f} / {m['img_F1']:.3f}")

    # side-by-side at the operating gate 0.15
    print("\nAt gate 0.15 (whole -> ROI):")
    for name, key in [("img recall", "img_R"), ("img precision", "img_P"), ("box recall", "box_R")]:
        print(f"  {name}: {whole_sweep.metrics(0.15)[key]:.3f} -> {roi_sweep.metrics(0.15)[key]:.3f}")


if __name__ == "__main__":
    main()
