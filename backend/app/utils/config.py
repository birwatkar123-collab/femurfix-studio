"""Central configuration: paths, anatomical presets, HU thresholds, safety copy."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # -> backend/
APP_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
CASE_DIR = BASE_DIR / "cases"
MODELS_DIR = BASE_DIR / "models"

for directory in (UPLOAD_DIR, OUTPUT_DIR, CASE_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DISCLAIMER = (
    "EDUCATIONAL SIMULATION ONLY. This application does not provide medical advice, "
    "diagnosis, or treatment. The 3D anatomy is an estimated reconstruction derived from a "
    "2D radiograph and is not suitable for measurement, surgical planning, or implant sizing. "
    "Always consult a qualified orthopaedic surgeon."
)

# X-ray fracture regions -> AO/OTA families used by the recommender.
FRACTURE_REGIONS: Dict[str, str] = {
    "proximal": "31",
    "shaft": "32",
    "distal": "33",
}

BODY_REGION_LABELS = {
    "proximal": "Proximal femur (neck / trochanteric)",
    "shaft": "Femoral shaft",
    "distal": "Distal femur",
}

# Imaging / segmentation thresholds
HU_BONE_MIN = 200
HU_CORTICAL_MIN = 700
HU_METAL_MIN = 1500
HU_SOFT_TISSUE_MIN = -100
HU_SOFT_TISSUE_MAX = 200

# Canonical femur dimensions (mm), used by the parametric shape prior.
FEMUR_CANONICAL_LENGTH_MM = 450.0
FEMUR_CANONICAL_SHAFT_DIAMETER_MM = 30.0
FEMUR_CANONICAL_HEAD_DIAMETER_MM = 48.0

# API behaviour
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 250 * 1024 * 1024))
HF_SPACE_ID = os.environ.get("HF_SPACE_ID", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}

# Local fracture detector
LOCAL_ONNX_MODEL = Path(
    os.environ.get("FRACTURE_ONNX_MODEL", str(MODELS_DIR / "femur_fracture.onnx"))
).expanduser()
USE_LOCAL_ONNX = os.environ.get("USE_LOCAL_ONNX", "1").strip().lower() not in {"0", "false", "no"}
LOCAL_ONNX_CONF = float(os.environ.get("LOCAL_ONNX_CONF", "0.15"))
# Below LOCAL_ONNX_CONF but above this, the detector produced faint signals
# worth surfacing to the user as "weak signal", not a flat negative.
LOCAL_ONNX_WEAK_CONF = float(os.environ.get("LOCAL_ONNX_WEAK_CONF", "0.08"))
LOCAL_ONNX_IOU = float(os.environ.get("LOCAL_ONNX_IOU", "0.45"))
# Optional isotonic calibration of the detector score to an image-level fracture
# probability. Ignored when the file is absent or its model hash does not match.
LOCAL_ONNX_CALIBRATION = Path(
    os.environ.get(
        "FRACTURE_ONNX_CALIBRATION", str(MODELS_DIR / "femur_fracture.calibration.json")
    )
).expanduser()

SUPPORTED_DEVICE_FAMILIES: List[str] = [
    "cannulated_screws",
    "dynamic_hip_screw",
    "cephalomedullary_nail",
    "proximal_locking_plate",
    "intramedullary_nail",
    "locking_plate",
    "distal_femur_plate",
]
