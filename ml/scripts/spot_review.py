#!/usr/bin/env python3
"""Clinical spot-review: draw boxes from review_manifest.csv onto images.

Usage:
  python ml/scripts/spot_review.py --samples 20 --out ml/data/region/spot_review
  # opens the output folder; inspect *_boxed.jpg images

Then delete the folder when review is done.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REGIONS = ["femoral_neck", "intertrochanteric", "subtrochanteric", "femoral_shaft", "distal_femur"]
COLORS = {
    "femoral_neck": (0, 215, 255),        # orange
    "intertrochanteric": (0, 255, 128),    # green
    "subtrochanteric": (255, 128, 0),      # blue
    "femoral_shaft": (255, 0, 255),        # magenta
    "distal_femur": (255, 255, 0),         # cyan
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("ml/data/region/review_manifest.csv"))
    parser.add_argument("--images-root", type=Path, default=Path("ml/data/region/images"))
    parser.add_argument("--out", type=Path, default=Path("ml/data/region/spot_review"))
    parser.add_argument("--samples", type=int, default=20, help="Random positives to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"Manifest not found: {args.manifest}")

    rows = []
    with args.manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["negative"] == "1" or not row["boxes"]:
                continue
            rows.append(row)

    if not rows:
        sys.exit("No positive images in manifest")

    random.seed(args.seed)
    picked = random.sample(rows, min(args.samples, len(rows)))
    args.out.mkdir(parents=True, exist_ok=True)

    for row in picked:
        rel = row["file"]
        # manifest stores e.g. "images/train/..."; images_root is the parent of images/
        img_path = args.images_root.parent / rel
        if not img_path.exists():
            print(f"missing: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"unreadable: {img_path}")
            continue

        h, w = img.shape[:2]
        classes = row["classes"].split(",")

        for cls in classes:
            if cls not in REGIONS:
                continue
            # Find the box for this class (manifest doesn't store coords; re-read label file)
            label_path = (args.images_root.parent / "labels" / rel).with_suffix(".txt")
            if not label_path.exists():
                continue
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                if REGIONS[cid] != cls:
                    continue
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                color = COLORS.get(cls, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{cls} ({cid})"
                cv2.putText(img, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_path = args.out / f"{Path(rel).stem}_boxed.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"  {rel} -> {out_path.name}  classes: {classes}")

    print(f"\nWrote {len(picked)} boxed images to {args.out}")
    print("Open the folder and verify: does the orange box sit on the femoral neck?")
    print("                        does green sit on the intertrochanteric region?")
    print("                        does blue sit on the subtrochanteric region?")

if __name__ == "__main__":
    main()