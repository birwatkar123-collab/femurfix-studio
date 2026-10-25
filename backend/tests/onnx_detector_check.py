#!/usr/bin/env python3
"""Validate the local ONNX fracture detector with a real model or a generated stub."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def build_stub(path: Path, imgsz: int = 64) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Three anchors: two overlapping high-score boxes and one below threshold.
    data = np.array(
        [
            [32.0, 32.0, 20.0, 20.0, 0.90],
            [33.0, 33.0, 20.0, 20.0, 0.80],
            [10.0, 10.0, 8.0, 8.0, 0.05],
        ],
        dtype=np.float32,
    ).T[None]
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["output0"],
                value=numpy_helper.from_array(data, name="stub_output"),
            )
        ],
        "stub",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 3])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 12)], producer_name="femurfix-stub"
    )
    model.ir_version = 13  # onnxruntime 1.31 rejects IR 14
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to a real YOLO ONNX export")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from make_synthetic_xray import build_xray

    if args.model:
        os.environ["FRACTURE_ONNX_MODEL"] = str(Path(args.model).resolve())
        stub = False
    else:
        stub_path = Path(tempfile.mkdtemp(prefix="femurfix_stub_")) / "stub.onnx"
        build_stub(stub_path)
        os.environ["FRACTURE_ONNX_MODEL"] = str(stub_path)
        stub = True

    from app.services.ml_detector import get_detector

    detector = get_detector()
    if detector is None:
        raise SystemExit("detector did not load")

    input_meta = detector.session.get_inputs()[0]
    output_meta = detector.session.get_outputs()[0]
    print(f"model: {detector.path}")
    print(f"input: {input_meta.name} {input_meta.shape}")
    print(f"output: {output_meta.name} {output_meta.shape}")
    print(f"classes: {detector.names}")

    image = build_xray(fracture=True).astype(np.uint8)
    detection = detector.detect(image)
    print(f"detections: {detection['num_detections']} best={detection['confidence']}")
    if detection["bbox_norm"]:
        print(f"bbox_norm: {detection['bbox_norm']} box_pixels: {detection['box_pixels']}")

    assert 0.0 <= detection["confidence"] <= 1.0
    if detection["bbox_norm"]:
        x1, y1, x2, y2 = detection["bbox_norm"]
        assert 0.0 <= x1 < x2 <= 1.0, detection["bbox_norm"]
        assert 0.0 <= y1 < y2 <= 1.0, detection["bbox_norm"]

    if stub:
        assert detection["detected"] and detection["num_detections"] == 1, detection
        assert abs(detection["confidence"] - 0.90) < 1e-3, detection
        _, scale, (pad_x, pad_y) = detector._preprocess(image)
        height, width = image.shape
        expected = [
            (22.0 - pad_x) / scale,
            (22.0 - pad_y) / scale,
            (42.0 - pad_x) / scale,
            (42.0 - pad_y) / scale,
        ]
        assert np.allclose(
            detection["box_pixels"], np.clip(expected, 0, [width, height, width, height]), atol=1.0
        ), (detection["box_pixels"], expected)

    print("ONNX DETECTOR OK")


if __name__ == "__main__":
    main()
