"""Verify the two-model path: fracture detector + anatomical-zone classifier.

Runs the production analyzer over the user-upload samples and reports, per film,
which model supplied the region label. Run from the repo root:

    python backend/tests/region_classifier_check.py
    python backend/tests/region_classifier_check.py --json

The expectation is that detection parity with v3 is unchanged (5/9) and that
`region_source` is `zone_model` wherever the zone model was confident.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

from app.services.image_loader import ImageLoader
from app.services.ml_detector import get_detector, get_region_classifier
from app.services.xray_analyzer import XRayAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="dump rows as JSON")
    args = parser.parse_args()

    detector = get_detector()
    classifier = get_region_classifier()
    if detector is None:
        print("FAIL: fracture detector unavailable")
        sys.exit(1)
    print(f"fracture detector : {detector.path.name} "
          f"classes={list(detector.names.values())} calibrated={bool(detector.calibration)}")
    if classifier is None:
        print("zone classifier   : unavailable (region will stay geometric)")
    else:
        print(f"zone classifier   : {classifier.path.name} "
              f"classes={list(classifier.names.values())} gate={classifier.conf}")

    loader = ImageLoader()
    analyzer = XRayAnalyzer()
    rows = []
    detected_count = 0
    for path in sorted(
        p for p in (REPO / "samples").iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ):
        image, meta = loader.load_xray(path)
        result = analyzer.analyze(image, meta, image_path=str(path))
        fracture = result["fracture"]
        detected = bool(fracture.get("detected"))
        detected_count += detected
        rows.append({
            "file": path.name,
            "detected": detected,
            "confidence": fracture.get("confidence"),
            "detector_confidence": fracture.get("detector_confidence"),
            "region": fracture.get("region"),
            "region_source": fracture.get("region_source"),
            "model_class": fracture.get("model_class"),
            "region_score": fracture.get("region_score"),
            "view_hint": fracture.get("view_hint"),
            "weak_signal": fracture.get("weak_signal"),
        })
        if not args.json:
            hint = fracture.get("view_hint")
            suffix = f" view={hint['class_name']}@{hint['score']:.2f}" if hint else ""
            print(f"  {'[+]' if detected else '[-]'} {path.name:46s} "
                  f"conf={fracture.get('confidence', 0.0):.3f} "
                  f"region={fracture.get('region')} "
                  f"via={fracture.get('region_source') or '-'} "
                  f"class={fracture.get('model_class') or '-'}{suffix}")

    print(f"\ndetected {detected_count}/{len(rows)} samples "
          f"(v3 reference: 5/9) | engine={analyzer.last_engine}")
    if args.json:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
