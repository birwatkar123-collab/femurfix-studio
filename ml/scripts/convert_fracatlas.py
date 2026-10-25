#!/usr/bin/env python3
"""Build a YOLO detection dataset from FracAtlas using its official splits.

Positive (fractured) images follow ``Utilities/Fracture Split/{train,valid,test}.csv``.
Negative images (``images/Non_fractured``) are sampled per split at ``--neg-ratio``
times the number of positives so every split keeps the same class balance.

Output layout::

    ml/data/images/{train,val,test}/*.jpg
    ml/data/labels/{train,val,test}/*.txt   # empty label = negative image
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageFile

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ML_ROOT / "data" / "raw" / "fracatlas-original-dataset" / "FracAtlas"
DEFAULT_DST = ML_ROOT / "data"
SPLIT_SOURCES = {"train": "train.csv", "val": "valid.csv", "test": "test.csv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def read_split(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "image_id" not in rows[0]:
        raise SystemExit(f"Unexpected split CSV format: {csv_path}")
    return [row["image_id"] for row in rows]


def check_inputs(src: Path) -> tuple[Path, Path, Path]:
    fractured = src / "images" / "Fractured"
    negatives = src / "images" / "Non_fractured"
    labels = src / "Annotations" / "YOLO"
    for path in (fractured, negatives, labels, src / "Utilities" / "Fracture Split"):
        if not path.is_dir():
            raise SystemExit(f"Missing required directory: {path}")
    return fractured, negatives, labels


def copy_image(src: Path, dst: Path) -> str:
    """Copy a JPEG, repairing truncated files or missing EOI markers.

    Ultralytics treats a JPEG that does not end in ``\\xff\\xd9`` as corrupt and
    rewrites it. Repairing here keeps rebuilds deterministic.
    """
    needs_repair = False
    try:
        with Image.open(src) as image:
            image.load()
        with src.open("rb") as handle:
            handle.seek(-2, 2)
            needs_repair = handle.read() != b"\xff\xd9"
        if not needs_repair:
            shutil.copy2(src, dst)
            return "ok"
    except Exception:
        needs_repair = True

    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(src) as image:
            image.convert("RGB").save(dst, format="JPEG", quality=95, subsampling=0)
        return "repaired"
    except Exception as exc:
        print(f"  could not repair {src.name}: {exc}")
        return "failed"
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def clear_split(dst: Path) -> None:
    for split in SPLIT_SOURCES:
        for kind in ("images", "labels"):
            out_dir = dst / kind / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for path in out_dir.iterdir():
                if path.is_file():
                    path.unlink()
    for cache in (dst / "labels").glob("*.cache"):
        cache.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="FracAtlas root")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="output dataset root")
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=3.0,
        help="negative images per positive image in each split",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fractured_dir, negative_dir, label_dir = check_inputs(args.src)
    positive_names = {
        split: read_split(args.src / "Utilities" / "Fracture Split" / csv_name)
        for split, csv_name in SPLIT_SOURCES.items()
    }

    for split, names in positive_names.items():
        for name in names:
            image = fractured_dir / name
            label = label_dir / f"{Path(name).stem}.txt"
            if not image.is_file():
                raise SystemExit(f"[{split}] missing fractured image: {image}")
            if not label.is_file() or not label.read_text().strip():
                raise SystemExit(f"[{split}] missing or empty annotation: {label}")

    negative_names = sorted(
        path.name for path in negative_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    random.Random(args.seed).shuffle(negative_names)
    if not negative_names:
        raise SystemExit(f"No negative images found in {negative_dir}")

    cursor = 0
    plan: dict[str, tuple[list[str], list[str]]] = {}
    for split, names in positive_names.items():
        wanted = round(len(names) * args.neg_ratio)
        available = len(negative_names) - cursor
        if wanted > available:
            raise SystemExit(
                f"Not enough negative images: need {wanted} for {split}, only {available} left. "
                "Use a smaller --neg-ratio."
            )
        plan[split] = (names, negative_names[cursor : cursor + wanted])
        cursor += wanted

    total_pos = sum(len(pos) for pos, _ in plan.values())
    total_neg = sum(len(neg) for _, neg in plan.values())
    print(f"Positives: {total_pos} | negatives: {total_neg} | unused negatives: {len(negative_names) - cursor}")
    for split, (pos, neg) in plan.items():
        print(f"  {split}: {len(pos)} positive, {len(neg)} negative, {len(pos) + len(neg)} total")

    if args.dry_run:
        return

    clear_split(args.dst)
    box_count = 0
    copied = 0
    repaired: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    for split, (positives, negatives) in plan.items():
        images_out = args.dst / "images" / split
        labels_out = args.dst / "labels" / split
        for name in positives:
            status = copy_image(fractured_dir / name, images_out / name)
            if status == "failed":
                failed.append((split, name))
                continue
            if status == "repaired":
                repaired.append((split, name))
            label_src = label_dir / f"{Path(name).stem}.txt"
            text = label_src.read_text().strip()
            box_count += len(text.splitlines())
            (labels_out / f"{Path(name).stem}.txt").write_text(text + "\n")
            copied += 1
        for name in negatives:
            status = copy_image(negative_dir / name, images_out / name)
            if status == "failed":
                failed.append((split, name))
                continue
            if status == "repaired":
                repaired.append((split, name))
            (labels_out / f"{Path(name).stem}.txt").write_text("")
            copied += 1

    print(f"Copied {copied} images, {box_count} boxes to {args.dst}")
    if repaired:
        print(f"Repaired {len(repaired)} truncated JPEGs: {repaired}")
    if failed:
        print(f"Skipped {len(failed)} unreadable images (and their labels): {failed}")


if __name__ == "__main__":
    main()
