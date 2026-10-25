#!/usr/bin/env python3
"""
Split dataset into train/val/test with patient-level awareness.
"""
import argparse
import random
import shutil
from pathlib import Path

random.seed(42)


def split(images_in: Path, labels_in: Path, out_root: Path, train=0.7, val=0.15, test=0.15, patient_regex=None):
    out_root.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [p for p in images_in.iterdir() if p.suffix.lower() in exts]
    
    # Group by patient if regex provided (simple heuristic)
    if patient_regex:
        import re
        pat = re.compile(patient_regex)
        groups = {}
        for img in images:
            m = pat.search(img.stem)
            key = m.group(0) if m else img.stem
            groups.setdefault(key, []).append(img)
        items = list(groups.items())
        random.shuffle(items)
        # flatten
        grouped_images = []
        for k, imgs in items:
            grouped_images.extend(imgs)
        images = grouped_images
    else:
        random.shuffle(images)
    
    n = len(images)
    n_train = int(n * train)
    n_val = int(n * val)
    
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    
    for split_name, img_list in splits.items():
        for img in img_list:
            dst_img = out_root / 'images' / split_name / img.name
            shutil.copy2(img, dst_img)
            lbl = labels_in / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, out_root / 'labels' / split_name / lbl.name)
            else:
                # create empty label for negative
                (out_root / 'labels' / split_name / f"{img.stem}.txt").write_text('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-in', required=True)
    parser.add_argument('--labels-in', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--train', type=float, default=0.7)
    parser.add_argument('--val', type=float, default=0.15)
    parser.add_argument('--test', type=float, default=0.15)
    parser.add_argument('--patient-regex', default=None)
    args = parser.parse_args()
    split(Path(args.images_in), Path(args.labels_in), Path(args.out_root), args.train, args.val, args.test, args.patient_regex)
    print('Done')
