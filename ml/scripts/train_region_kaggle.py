#!/usr/bin/env python3
"""FemurFix 5-class region fine-tune @960 — Kaggle kernel entry point.

Attach datasets:
- birwatkar/femurfix-region-dataset (data.yaml + images/labels)
- birwatkar/femurfix-v3-checkpoint (best_v3.pt starting weights)

Outputs (in /kaggle/working):
- femur_region.onnx
- region_summary.json
"""
from __future__ import annotations

import subprocess
import sys

# The Kaggle script image does not ship ultralytics; internet must be enabled.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "ultralytics>=8.3,<9"],
    check=True,
)

from pathlib import Path  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATA_ROOT = Path("/kaggle/input")
print("Input mounts:")
for p in sorted(DATA_ROOT.glob("*")):
    if p.is_dir():
        print(f"  {p.name}: {[c.name for c in sorted(p.iterdir())][:12]}")

def _find(name: str) -> list[Path]:
    hits = sorted(DATA_ROOT.rglob(name))
    if not hits:
        print(f"file not found under {DATA_ROOT}: {name}")
        for p in sorted(DATA_ROOT.rglob("*")):
            if p.is_file():
                print(f"  {p}")
    return hits

yaml_hits = _find("data.yaml")
# Pick the data.yaml that actually sits next to images/val (a stray copy of the
# same file may exist at another level, extracted folder etc.).
yaml_hits = [p for p in yaml_hits if (p.parent / "images" / "val").is_dir()]
assert yaml_hits, "no data.yaml found with images/val alongside it"
DATA_YAML_RAW = yaml_hits[0]

ckpt_hits = _find("best_v3.pt")
assert ckpt_hits, "best_v3.pt not found — checkpoint attachment missing"
CKPT = ckpt_hits[0]

# The uploaded data.yaml carries the authoring machine's absolute paths
# (D:/New folder (3)/...). Rewrite it against the Kaggle mount.
import yaml as _yaml  # noqa: E402

_cfg = _yaml.safe_load(DATA_YAML_RAW.read_text())
_cfg["path"] = DATA_YAML_RAW.parent.as_posix()
DATA_YAML = Path("/kaggle/working/data.yaml")
DATA_YAML.write_text(_yaml.safe_dump(_cfg, sort_keys=False))
print(f"rewrote data.yaml -> {DATA_YAML} (path={_cfg['path']})")

print(f"Data: {DATA_YAML}")
print(f"Checkpoint: {CKPT}")

model = YOLO(str(CKPT))  # start from v3 backbone

model.train(
    data=str(DATA_YAML),
    epochs=60,
    patience=20,
    imgsz=960,
    batch=8,
    device=0,
    seed=0,
    deterministic=True,
    cache=True,
    fliplr=0.5,
    degrees=10.0,
    close_mosaic=12,
    project="/kaggle/working/runs",
    name="region_yolo11s_960",
)

run_dir = Path("/kaggle/working/runs/region_yolo11s_960")
best = run_dir / "weights" / "best.pt"
if not best.is_file():
    # Ultralytics may increment the run name if it reuses the folder.
    candidates = sorted(Path("/kaggle/working/runs").glob("region_yolo11s_960*/weights/best.pt"))
    assert candidates, "no best.pt produced"
    best = candidates[-1]
print(f"best: {best}")

metrics = YOLO(str(best)).val(data=str(DATA_YAML), split="test", imgsz=960, batch=8, device=0, plots=True)

# Ultralytics 8.4 returns per-class AP as numpy arrays (not dicts). NOTE: `box.maps`
# is the per-class mAP@50-95 vector; `box.maps50` is per-class mAP@50. Reading the
# wrong one produced a region_summary.json whose "per_class_ap50" was really AP@50-95.
def _ravel(values):
    return [float(v) for v in np.ravel(np.asarray(values))]

names = getattr(metrics, "names", {}) or {}
ap50 = _ravel(getattr(metrics.box, "maps50", metrics.box.maps))
ap50_95 = _ravel(metrics.box.maps)
per_class = {
    str(names.get(i, i)): {"AP50": ap50[i] if i < len(ap50) else None,
                           "AP50_95": ap50_95[i] if i < len(ap50_95) else None}
    for i in range(len(ap50_95))
}
summary = {
    "per_class": per_class,
    "mAP50": float(metrics.box.map50),
    "mAP50-95": float(metrics.box.map),
    "imgsz": 960,
    "checkpoint": str(best),
}
Path("/kaggle/working/region_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

import shutil  # noqa: E402

shutil.copy2(best, "/kaggle/working/best_region.pt")

onnx_path = Path(YOLO(str(best)).export(format="onnx", imgsz=960, opset=12, simplify=True, dynamic=False))
target = Path("/kaggle/working/femur_region.onnx")
target.write_bytes(onnx_path.read_bytes())
print(f"download-ready: {target} ({target.stat().st_size} bytes)")
