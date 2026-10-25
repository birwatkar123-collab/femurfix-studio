#!/usr/bin/env python3
"""
Validate YOLO labels and images.
"""
import argparse
from pathlib import Path

CLASSES = {0:'femoral_neck',1:'intertrochanteric',2:'subtrochanteric',3:'femoral_shaft',4:'distal_femur',5:'no_fracture'}


def validate(images_dir: Path, labels_dir: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
    errors = []
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            errors.append(f'Missing label: {lbl}')
            continue
        content = lbl.read_text().strip()
        if not content:
            continue  # empty is ok for negatives
        for line in content.splitlines():
            parts = line.split()
            if len(parts) != 5:
                errors.append(f'Bad format {lbl}: {line}')
                continue
            cls = int(parts[0])
            if cls not in CLASSES:
                errors.append(f'Bad class {lbl}: {cls}')
                continue
            vals = [float(x) for x in parts[1:]]
            if any(v < 0 or v > 1 for v in vals):
                errors.append(f'Out of range {lbl}: {line}')
    if errors:
        for e in errors[:50]:
            print(e)
        print(f'\nTotal errors: {len(errors)}')
    else:
        print('All labels valid')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--labels-dir', required=True)
    args = parser.parse_args()
    validate(Path(args.images_dir), Path(args.labels_dir))
