#!/usr/bin/env python3
"""Convert one or more region-labelled YOLO sources into the FemurFix region schema.

Region track (see ``ml/README.md``): a YOLO dataset whose single class replaces
both the binary fracture detector and the geometric region assignment. Target
classes::

    femoral_neck, intertrochanteric, subtrochanteric, femoral_shaft, distal_femur

This script re-maps source class ids, keeps explicit negatives optionally,
splits images into train/val/test, writes ``data.yaml``, and emits a
``review_manifest.csv`` for clinical review of every box and class.

Input expected per ``--source`` (repeatable): a directory with ``images/`` and
``labels/`` (labels flat or split by train/val/test). Class names either come
from a ``classes.txt`` next to the labels or via ``--classes`` (repeatable,
ordered, must match every source).

Example (Roboflow proximal-femur export)::

    python ml/scripts/prepare_region_dataset.py \\
        --source ml/data/raw/proximal-femur-fracture/yolo \\
        --classes neck intertrochanteric subtrochanteric \\
            greater_trochanter lesser_trochanter dislocation normal \\
        --dst ml/data/region --keep-negatives
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Default class -> region mapping. Sources lacking a listed class simply skip
# that entry. `greater/lesser trochanter` are AO sub-codes of intertrochanteric.
DEFAULT_MAPPING = {
    "neck": "femoral_neck",
    "femoral_neck": "femoral_neck",
    "intertrochanteric": "intertrochanteric",
    "greater_trochanter": "intertrochanteric",
    "greater trochanter": "intertrochanteric",
    "grater trochanter": "intertrochanteric",  # dataset's own spelling
    "lesser_trochanter": "intertrochanteric",
    "lesser trochanter": "intertrochanteric",
    "subtrochanteric": "subtrochanteric",
    "femoral_shaft": "femoral_shaft",
    "shaft": "femoral_shaft",
    "distal_femur": "distal_femur",
    "distal": "distal_femur",
}
NEGATIVE_CLASSES = {"normal", "dislocation", "non_fractured"}
REGIONS = ["femoral_neck", "intertrochanteric", "subtrochanteric", "femoral_shaft", "distal_femur"]


def find_files(source: Path) -> list[Path]:
    if (source / "images").is_dir():
        return sorted(p for p in (source / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    files = []
    for split in ("train", "valid", "val", "test"):
        img_dir = source / split / "images"
        if img_dir.is_dir():
            files.extend(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if files:
        return sorted(files)
    return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def label_for(source: Path, image: Path) -> Path:
    rel = image.relative_to(source)
    stem = rel.with_suffix("").as_posix()
    # Roboflow layout: <split>/images/<stem>.<ext> beside <split>/labels/<stem>.txt
    if (
        len(rel.parts) >= 2
        and rel.parts[-2] == "images"
        and (source / rel.parts[0] / "labels").is_dir()
    ):
        return source / rel.parts[0] / "labels" / f"{image.stem}.txt"
    candidates = [
        source / "labels" / f"{stem}.txt",
        source / "labels" / f"{image.stem}.txt",
    ]
    return next((c for c in candidates if c.is_file()), source / "labels" / f"{image.stem}.txt")


def parse_label_line(parts: list[str]) -> tuple[int, float, float, float, float]:
    """Return (class_id, cx, cy, w, h) from a YOLO bbox or segmentation line.

    YOLO bbox: ``class cx cy w h``. YOLO segmentation (Roboflow exports):
    ``class x1 y1 x2 y2 ...`` (polygon, already normalised); the bounding box
    is computed from the polygon extremes.
    """
    idx = int(parts[0])
    coords = [float(v) for v in parts[1:]]
    if len(parts) == 5:
        cx, cy, w, h = coords
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        w, h = max(xs) - min(xs), max(ys) - min(ys)
    else:
        raise ValueError(f"unrecognised label line with {len(parts)} values")
    return idx, cx, cy, w, h


def choose_split(rng: random.Random, fractions: tuple[float, float, float]) -> str:
    cut1 = fractions[0]
    cut2 = fractions[0] + fractions[1]
    roll = rng.random()
    if roll < cut1:
        return "train"
    if roll < cut2:
        return "val"
    return "test"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True, type=Path, help="YOLO source dir (repeatable)")
    parser.add_argument("--classes", action="append",
                        help="Source class names (repeatable; comma-separated list)")
    parser.add_argument("--dst", type=Path, default=root / "data" / "region")
    parser.add_argument("--keep-negatives", action="store_true",
                        help="Keep normal/dislocation images as negatives (empty labels)")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--prefix-chars", type=int, default=3,
                        help="Filename prefix length used as the patient/group key; "
                             "0 = treat every file as its own group")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fractions = (args.train, args.val, 1.0 - args.train - args.val)
    if abs(fractions[2]) < 1e-6:
        raise SystemExit("train + val must not exceed 1.0")

    src_name_list: list[str] = []
    for chunk in args.classes or []:
        src_name_list.extend(n.strip() for n in chunk.split(",") if n.strip())
    args.classes = src_name_list
    mapping = dict(DEFAULT_MAPPING)
    target_ids = {name: i for i, name in enumerate(REGIONS)}

    for dst_sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (args.dst / dst_sub / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    group_to_split: dict[str, str] = {}
    manifest_rows: list[dict[str, str]] = []
    stats = {"kept": 0, "negatives": 0, "dropped": 0, "unmapped": {}}
    counters = {split: 0 for split in ("train", "val", "test")}

    for source in args.source:
        source = source.resolve()
        files = find_files(source)
        class_file = source / "classes.txt"
        if not args.classes and class_file.is_file():
            src_names = [n.strip() for n in class_file.read_text().splitlines() if n.strip()]
        elif args.classes:
            src_names = list(args.classes)
        else:
            raise SystemExit(f"No class names: pass --classes or a classes.txt in {source}")

        src_ids = {name: i for i, name in enumerate(src_names)}
        print(f"source={source.name} images={len(files)} classes={src_names}")

        for image in files:
            label_src = label_for(source, image)
            boxes = []
            positives = []
            unmapped = []
            if label_src.is_file():
                skipped = 0
                for line in label_src.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        idx, cx, cy, w, h = parse_label_line(parts)
                    except (ValueError, IndexError):
                        skipped += 1
                        continue
                    if not 0 <= idx < len(src_names):
                        unmapped.append(f"id{idx}")
                        continue
                    name = src_names[idx]
                    if name.lower() in NEGATIVE_CLASSES:
                        continue  # ignore negative annotations
                    target = mapping.get(name.lower())
                    if target is None:
                        unmapped.append(name)
                        continue
                    positives.append(
                        f"{target_ids[target]} {cx:.7f} {cy:.7f} {w:.7f} {h:.7f}"
                    )
                if skipped:
                    stats["unmapped"][f"malformed lines: {skipped}"] = (
                        stats["unmapped"].get(f"malformed lines: {skipped}", 0) + skipped
                    )
            if unmapped:
                stats["unmapped"][" ".join(sorted(unmapped))] = (
                    stats["unmapped"].get(" ".join(sorted(unmapped)), 0) + 1)
            is_negative = not positives
            if is_negative and not args.keep_negatives:
                stats["dropped"] += 1
                continue

            group = (image.stem[: max(1, args.prefix_chars)].lower()
                     if args.prefix_chars > 0 else image.stem.lower())
            split = group_to_split.get(group) or choose_split(rng, fractions)
            group_to_split[group] = split
            counters[split] += 1

            if is_negative:
                stats["negatives"] += 1
                text = ""
            else:
                text = "\n".join(positives) + "\n"
                stats["kept"] += 1

            used_stem = f"{source.name}_{image.stem}".replace(" ", "_")
            dst_image = args.dst / "images" / split / f"{used_stem}{image.suffix.lower()}"
            shutil.copy2(image, dst_image)
            (args.dst / "labels" / split / f"{used_stem}.txt").write_text(text)

            region_names = sorted({REGIONS[int(p.split(" ")[0])] for p in positives})
            manifest_rows.append({
                "file": dst_image.relative_to(args.dst).as_posix(),
                "source": source.name,
                "split": split,
                "group": group,
                "boxes": len(positives),
                "classes": ",".join(region_names),
                "negative": "1" if is_negative else "",
            })
            print(f"  {image.name} -> {split} ({'neg' if is_negative else f'{len(positives)} boxes'})")

    (args.dst / "data.yaml").write_text(
        f"path: {args.dst.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(REGIONS))
    )
    with (args.dst / "review_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "source", "split", "group", "boxes", "classes", "negative"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (args.dst / "mapping.json").open("w") as handle:
        import json
        json.dump({"source_classes": src_names, "regions": REGIONS, "mapping": mapping}, handle, indent=2)

    print(f"\nkept {stats['kept']} positives / {stats['negatives']} negatives "
          f"(dropped {stats['dropped']}) -> {args.dst}")
    print(f"splits: {counters}")
    if stats["unmapped"]:
        print("UNMAPPED source classes (excluded from output):")
        for name, count in stats["unmapped"].items():
            print(f"  {name}: {count} boxes")
    print("Next: clinical review of review_manifest.csv, then validate with "
          "ml/scripts/validate_labels.py")


if __name__ == "__main__":
    main()
