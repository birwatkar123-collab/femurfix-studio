#!/usr/bin/env python3
"""
Convert COCO JSON annotations to YOLO format.
Maps source class names to FemurFix classes.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List

# Map common names to our classes
CLASS_MAP = {
    # FracAtlas/common variations
    'femoral_neck': 0, 'neck': 0, 'femur_neck': 0, 'proximal_femur_neck': 0,
    'intertrochanteric': 1, 'inter-trochanteric': 1, 'trochanteric': 1, 'pertrochanteric': 1,
    'subtrochanteric': 2, 'sub-trochanteric': 2,
    'femoral_shaft': 3, 'shaft': 3, 'diaphyseal': 3, 'femur_shaft': 3,
    'distal_femur': 4, 'distal': 4, 'supracondylar': 4,
    'no_fracture': 5, 'normal': 5, 'negative': 5,
}


def convert(coco_json: Path, images_dir: Path, labels_out: Path, class_map: Dict[str, int] = None):
    class_map = class_map or CLASS_MAP
    labels_out.mkdir(parents=True, exist_ok=True)
    
    with open(coco_json) as f:
        data = json.load(f)
    
    # Build image id -> info
    img_info = {img['id']: img for img in data['images']}
    # Build category id -> name
    cat_name = {cat['id']: cat['name'].lower().replace(' ', '_') for cat in data['categories']}
    
    # Group annotations by image
    anns_by_img: Dict[int, List] = {}
    for ann in data['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)
    
    for img_id, img in img_info.items():
        filename = img['file_name']
        img_w = img['width']
        img_h = img['height']
        stem = Path(filename).stem
        label_path = labels_out / f"{stem}.txt"
        
        lines = []
        for ann in anns_by_img.get(img_id, []):
            cat_id = ann['category_id']
            name = cat_name.get(cat_id, '')
            if name not in class_map:
                # try fuzzy match
                matched = None
                for k, v in class_map.items():
                    if k in name or name in k:
                        matched = v
                        break
                if matched is None:
                    print(f"Warning: unknown class '{name}' in {filename}, skipping")
                    continue
                cls = matched
            else:
                cls = class_map[name]
            
            bbox = ann['bbox']  # [x,y,w,h]
            x, y, w, h = bbox
            # Convert to YOLO format (normalized center x,y,w,h)
            cx = (x + w/2) / img_w
            cy = (y + h/2) / img_h
            nw = w / img_w
            nh = h / img_h
            # Clamp
            cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
            nw, nh = max(0, min(1, nw)), max(0, min(1, nh))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        if lines:
            label_path.write_text('\n'.join(lines) + '\n')
        else:
            # Write empty file for negative samples if image exists
            if (images_dir / filename).exists() or (images_dir / Path(filename).name).exists():
                label_path.write_text('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-json', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--labels-out', required=True)
    args = parser.parse_args()
    convert(Path(args.coco_json), Path(args.images_dir), Path(args.labels_out))
    print('Done')
