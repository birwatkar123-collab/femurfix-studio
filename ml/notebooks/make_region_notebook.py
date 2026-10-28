#!/usr/bin/env python3
"""Emit the region fine-tuning notebook skeleton (kept as JSON for reproducibility)."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "region_yolo_finetune_gpu.ipynb"


def code(source: str) -> dict:
    return {"cell_type": "code", "id": f"c{abs(hash(source)) % 10**8:08x}", "metadata": {},
            "execution_count": None, "outputs": [], "source": [l + "\n" for l in source.splitlines()]}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "id": f"m{abs(hash(source)) % 10**8:08x}", "metadata": {},
            "source": [l + "\n" for l in source.splitlines()]}


cells = [
    markdown("""# FemurFix — region fine-tuning (5-class) @ 960 px

**SKELETON** — run only after `ml/data/region/` exists (built by
`ml/scripts/prepare_region_dataset.py` from a licensed region-labelled source, e.g. the
Roboflow `proximal-femur-fracture` export) and after clinical review of `review_manifest.csv`.

Classes: `femoral_neck`, `intertrochanteric`, `subtrochanteric`, `femoral_shaft`, `distal_femur`.

Plan (from `ml/README.md`):
- fine-tune from the current checkpoint (`best.pt` of v2 if it beat v1, else the v1 best) at
  `imgsz=960`
- split by patient (the prep script groups filenames by prefix; verify before training)
- hold out a hip-only test set and report **per-class AP** + confusion matrix
"""),
    code("""import torch
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))"""),
    code("""%pip install -q "ultralytics>=8.3,<9" "albumentations>=1.4.21,<2" "onnx>=1.16" "onnxslim>=0.1.31\""""),
    code("""from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path("/kaggle/input/REGION-DATASET/data.yaml")  # attach prep output as a Kaggle dataset
assert DATA_YAML.is_file(), "attach the prepared region dataset"

# fine-tune from the deployed binary checkpoint? Region classes differ, so start
# from a COCO/YOLO checkpoint (or the v2 best.pt backbone is transferred automatically
# by shape-mismatch handling). Prefer: YOLO("/kaggle/input/.../best_v2.pt") when attached.
model = YOLO("yolo11s.pt")
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
)"""),
    code("""import json
from pathlib import Path
from ultralytics import YOLO

run_dir = Path("/kaggle/working/runs/region_yolo11s_960")
best = run_dir / "weights" / "best.pt"
assert best.is_file()
metrics = YOLO(str(best)).val(data=str(DATA_YAML), split="test", imgsz=960, batch=8, device=0, plots=True)
summary = {
    "per_class_ap50": {k: float(v) for k, v in metrics.box.maps.items()},
    "mAP50": float(metrics.box.map50),
    "mAP50-95": float(metrics.box.map),
}
Path("/kaggle/working/region_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))"""),
    code("""from pathlib import Path
from ultralytics import YOLO

best = Path("/kaggle/working/runs/region_yolo11s_960/weights/best.pt")
onnx_path = Path(YOLO(str(best)).export(format="onnx", imgsz=960, opset=12, simplify=True, dynamic=False))
target = Path("/kaggle/working/femur_region.onnx")
target.write_bytes(onnx_path.read_bytes())
print("download-ready:", target, target.stat().st_size, "bytes")"""),
    markdown("""## Outputs
- `/kaggle/working/femur_region.onnx` — region detector candidate
- `/kaggle/working/region_summary.json` — per-class AP on the held-out set

Integration requires the backend changes in `ml_detector.py` (region class names read from ONNX
metadata already work — see `_read_names`) plus `xray_analyzer.py` to consume the class of the
top-scoring box instead of the geometric region fallback. Do **not** ship a region model fine-tuned
solely on non-femur anatomy."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", NB)
