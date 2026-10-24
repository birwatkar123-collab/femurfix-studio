#!/usr/bin/env python3
"""Generate a synthetic AP femur radiograph (with an optional fracture) for E2E tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WIDTH, HEIGHT = 620, 900
TOP, BOTTOM = 70, 830


def _radio_soft_tissue() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=np.float32)


def build_xray(fracture: bool = True, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((HEIGHT, WIDTH), 12.0, dtype=np.float32)

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    cx = WIDTH / 2
    length = BOTTOM - TOP

    # Soft-tissue envelope around the thigh.
    soft = ((xx - cx) / 150.0) ** 2 + ((yy - (TOP + length / 2)) / (length / 2 + 40)) ** 2 < 1.0
    img[soft] += 28.0

    # Bone mask: shaft + condyles + neck/head + trochanter.
    shaft = (np.abs(xx - cx) < 26) & (yy > TOP + 80) & (yy < BOTTOM - 30)
    distal = ((xx - cx) / 92.0) ** 2 + ((yy - (BOTTOM - 55)) / 62.0) ** 2 < 1.0
    head = ((xx - (cx + 74)) / 46.0) ** 2 + ((yy - (TOP + 78)) / 46.0) ** 2 < 1.0
    neck = (np.abs((yy - (TOP + 110)) - 0.55 * (xx - cx)) < 26) & (xx > cx - 10) & (xx < cx + 80)
    troch = ((xx - (cx - 40)) / 40.0) ** 2 + ((yy - (TOP + 70)) / 34.0) ** 2 < 1.0
    bone = shaft | distal | head | neck | troch

    # Cortical rim is brighter than the medullary canal.
    rim = bone & ~(
        (np.abs(xx - cx) < 17)
        & (yy > TOP + 100)
        & (yy < BOTTOM - 70)
    )
    img[bone] += 120.0
    img[rim] += 55.0
    img[distal] += 30.0
    img[head] += 35.0

    # Simulated fracture: oblique radiolucent line across the mid-shaft.
    fracture_y = TOP + int(length * 0.58)
    if fracture:
        draw = ImageDraw.Draw(Image.fromarray(np.zeros((HEIGHT, WIDTH), dtype=np.uint8)))
        overlay = Image.new("L", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(overlay)
        draw.line(
            [(int(cx - 34), fracture_y - 26), (int(cx + 34), fracture_y + 22)],
            fill=255,
            width=7,
        )
        gap = np.array(overlay, dtype=np.float32) / 255.0
        img = img * (1.0 - 0.92 * gap)
        # Mild displacement of the distal fragment.
        shift = 9
        distal_mask = np.zeros_like(img)
        distal_mask[fracture_y + 20 :, :] = 1.0
        shifted = np.roll(img, shift, axis=1)
        img = img * (1 - distal_mask) + shifted * distal_mask

    # Scatter / quantum noise and blur typical of radiographs.
    img += rng.normal(0.0, 7.0, img.shape)
    img = np.clip(img, 0, 255)
    pil = Image.fromarray(img.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.1))
    return np.array(pil, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="synthetic_femur_xray.png")
    parser.add_argument("--no-fracture", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    image = build_xray(fracture=not args.no_fracture, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(out)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.1f} KB) fracture={not args.no_fracture}")


if __name__ == "__main__":
    main()
