"""HF Space app (Gradio) for FemurFix Studio.

Deploy to Hugging Face Spaces with hardware=zero-a10g or zero-h100 and call from the backend
using gradio_client and the HF_SPACE_ID / HF_TOKEN environment variables.

This is the integration point for a fine-tuned fracture detector + bone segmentation model.
When real weights are not present the endpoints return an explicit 'unavailable' status so the
backend falls back safely to its heuristic engine.
"""
from __future__ import annotations

from typing import Any, Dict

import gradio as gr
import numpy as np
from PIL import Image

__version__ = "0.1.0"


def _load_model() -> Dict[str, Any]:
    # Replace with a real detector (e.g. YOLO/RT-DETR) + segmentation (e.g. SAM/MedSAM/U-Net).
    return {"ready": False, "model": None, "message": "No trained weights loaded"}


MODEL_STATE = _load_model()


def analyze_xray(image: Image.Image) -> Dict[str, Any]:
    if image is None:
        return {"status": "error", "error": "No image provided"}

    arr = np.array(image.convert("RGB"))
    height, width = arr.shape[:2]

    if not MODEL_STATE.get("ready"):
        return {
            "status": "unavailable",
            "message": MODEL_STATE.get("message"),
            "bone_mask": None,
            "fracture": None,
            "note": "Configure model weights to enable HF ZeroGPU inference.",
        }

    # TODO: run detector -> fracture bbox/class/conf, segmenter -> bone_mask (H,W,1) or bool.
    return {
        "status": "ok",
        "engine": "zerogpu_finetuned",
        "image_size": [width, height],
        "bone_mask": None,
        "fracture": {
            "detected": False,
            "region": "shaft",
            "ratio": 0.5,
            "bbox_norm": [0.15, 0.4, 0.85, 0.6],
            "pattern": "unknown",
            "confidence": 0.0,
        },
    }


def segment_femur(image: Image.Image) -> Dict[str, Any]:
    if image is None:
        return {"status": "error", "error": "No image provided"}
    if not MODEL_STATE.get("ready"):
        return {"status": "unavailable", "bone_mask": None}
    return {"status": "ok", "bone_mask": None}


demo = gr.Interface(
    fn=analyze_xray,
    inputs=gr.Image(type="pil", label="Femur AP/Lateral X-ray"),
    outputs=gr.JSON(label="Analysis"),
    title="FemurFix — Fracture Analysis (HF ZeroGPU)",
    description=(
        "Educational space endpoint. Returns schema-compatible JSON consumed by the FemurFix Studio backend. "
        "Set HF_SPACE_ID in the backend to route requests here."
    ),
)

with gr.Blocks(title="FemurFix Space") as app:
    gr.Markdown("# FemurFix Studio — HF ZeroGPU Space")
    gr.Markdown(
        "API endpoint `/analyze_xray` is available for `gradio_client`. "
        "This space is a stub until fine-tuned weights are mounted."
    )
    gr.JSON(value=MODEL_STATE, label="Model state")
    gr.Interface(fn=analyze_xray, inputs=gr.Image(type="pil"), outputs=gr.JSON())
    gr.Interface(fn=segment_femur, inputs=gr.Image(type="pil"), outputs=gr.JSON())


if __name__ == "__main__":
    app.launch()
