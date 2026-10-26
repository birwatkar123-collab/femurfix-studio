"""Score the user-upload sample set against a detector ONNX model.

Reuses the backend's OnnxFractureDetector (letterbox + postprocess identical to
production). Run from the repo root:

    python backend/tests/sample_baseline.py
    python backend/tests/sample_baseline.py --model .kaggle_staging/v2-output/femur_fracture_v2.onnx

Emits a compact per-sample report and a summary line, so v1/v2 runs can be
diffed side by side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.services.image_loader import ImageLoader
from app.services.ml_detector import OnnxFractureDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, help="ONNX path (default: configured v1 model)")
    parser.add_argument("--conf", type=float, default=0.15, help="app threshold (default 0.15)")
    args = parser.parse_args()

    det = OnnxFractureDetector(model_path=args.model, conf=args.conf)

    samples_dir = REPO / "samples"
    files = sorted(
        p for p in samples_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    rows = []
    hits = 0
    for path in files:
        image, _meta = ImageLoader().load_xray(path)
        out = det.detect(image)
        detected = bool(out["detected"])
        hits += detected
        calibrated = out.get("calibrated_confidence")
        rows.append({
            "file": path.name,
            "detected": detected,
            "confidence": out.get("confidence", 0.0),
            "calibrated_confidence": calibrated,
            "num_detections": out.get("num_detections", 0),
        })
        flag = "+" if detected else "-"
        cal = f"{calibrated:.3f}" if calibrated is not None else "n/a"
        print(f"  [{flag}] {path.name:44s} conf={out.get('confidence', 0.0):.3f} "
              f"cal={cal} (n={out.get('num_detections', 0)})")

    print(f"\nmodel={det.path.name} imgsz={det.imgsz} conf={det.conf} "
          f"| detected {hits}/{len(files)} samples")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
