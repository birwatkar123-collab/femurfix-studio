"""Local ONNX YOLO fracture detector."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..utils.config import (
    LOCAL_ONNX_CALIBRATION,
    LOCAL_ONNX_CONF,
    LOCAL_ONNX_IOU,
    LOCAL_ONNX_MODEL,
    USE_LOCAL_ONNX,
)


class DetectorUnavailable(RuntimeError):
    pass


class OnnxFractureDetector:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf: float = LOCAL_ONNX_CONF,
        iou: float = LOCAL_ONNX_IOU,
    ) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise DetectorUnavailable(f"onnxruntime unavailable: {exc}") from exc

        self.path = Path(model_path or LOCAL_ONNX_MODEL)
        if not self.path.is_file():
            raise DetectorUnavailable(f"ONNX model not found: {self.path}")

        try:
            self.session = ort.InferenceSession(
                str(self.path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise DetectorUnavailable(f"Could not load {self.path.name}: {exc}") from exc

        inputs = self.session.get_inputs()
        if not inputs:
            raise DetectorUnavailable("ONNX model has no inputs")
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) != 4 or not all(isinstance(v, int) for v in shape[2:]):
            raise DetectorUnavailable("ONNX model must have a static NCHW input shape")
        self.imgsz = int(shape[2])
        self.names = self._read_names()
        self.conf = float(conf)
        self.iou = float(iou)
        self.calibration = self._load_calibration()

    def _read_names(self) -> Dict[int, str]:
        raw = self.session.get_modelmeta().custom_metadata_map.get("names", "")
        if raw:
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return {int(key): str(value) for key, value in parsed.items()}
            except (ValueError, SyntaxError):
                pass
        return {0: "fracture"}

    def _model_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_calibration(self) -> Optional[Dict[str, Any]]:
        path = Path(LOCAL_ONNX_CALIBRATION)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("model_sha256") != self._model_sha256():
                print(f"[detector] {path.name} does not match this model; using raw scores")
                return None
            points = sorted(
                (float(x), float(y)) for x, y in data.get("points", [])
            )
            if len(points) < 2:
                return None
        except Exception as exc:
            print(f"[detector] could not load {path.name}: {exc}")
            return None
        return {
            "source": data.get("source"),
            "method": data.get("method", "isotonic"),
            "xs": np.array([p[0] for p in points], dtype=np.float64),
            "ys": np.array([p[1] for p in points], dtype=np.float64),
        }

    def _calibration_meta(self) -> Optional[Dict[str, Any]]:
        if not self.calibration:
            return None
        return {
            "source": self.calibration["source"],
            "method": self.calibration["method"],
        }

    def _calibrate(self, score: float) -> Optional[float]:
        if not self.calibration:
            return None
        xs, ys = self.calibration["xs"], self.calibration["ys"]
        index = int(np.searchsorted(xs, float(score), side="right")) - 1
        index = max(0, min(index, len(ys) - 1))
        return float(ys[index])

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        tensor, scale, pad = self._preprocess(image)
        pred = self.session.run(None, {self.input_name: tensor})
        pred0 = np.asarray(pred[0], dtype=np.float32)
        raw = pred0.copy()
        while raw.ndim > 2 and raw.shape[0] == 1:
            raw = raw[0]  # e.g. (1,5,8400) -> (5,8400)
        # Top scores before the confidence gate, so callers can surface a
        # "weak signal" state instead of a flat negative.
        score_row = raw[4] if raw.ndim == 2 and raw.shape[0] >= 5 else None
        raw_scores = (
            [round(float(v), 4) for v in np.sort(score_row)[-8:][::-1] if float(v) >= 0.04]
            if score_row is not None
            else []
        )
        raw_top = round(float(raw_scores[0]) if raw_scores else 0.0, 3)
        calibrated_raw_top = (
            round(self._calibrate(raw_top), 3) if self.calibration else None
        )
        boxes = self._postprocess(pred0, image.shape[:2], scale, pad)
        if not boxes:
            return {
                "detected": False,
                "confidence": 0.0,
                "calibrated_confidence": None,
                "raw_top": raw_top,
                "calibrated_raw_top": calibrated_raw_top,
                "raw_scores": raw_scores,
                "bbox_norm": None,
                "num_detections": 0,
                "calibration": self._calibration_meta(),
                "model": self.path.name,
            }

        best = boxes[0]
        confidence = float(best["confidence"])
        return {
            "detected": True,
            "confidence": round(confidence, 3),
            "calibrated_confidence": (
                round(self._calibrate(confidence), 3) if self.calibration else None
            ),
            "raw_top": raw_top,
            "calibrated_raw_top": calibrated_raw_top,
            "raw_scores": raw_scores,
            "bbox_norm": [round(float(v), 4) for v in best["bbox_norm"]],
            "box_pixels": [round(float(v), 1) for v in best["bbox_pixels"]],
            "num_detections": len(boxes),
            "class_name": self.names.get(int(best["class_id"]), "fracture"),
            "calibration": self._calibration_meta(),
            "model": self.path.name,
        }

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        if image.ndim != 2:
            raise ValueError("Expected a 2D grayscale radiograph")
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        gray = Image.fromarray(array).convert("RGB")
        width, height = gray.size
        scale = min(self.imgsz / width, self.imgsz / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = gray.resize((new_w, new_h), Image.BILINEAR)

        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114))
        canvas.paste(resized, (pad_x, pad_y))

        tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(tensor), scale, (pad_x, pad_y)

    def _postprocess(
        self,
        prediction: np.ndarray,
        image_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        pred = np.asarray(prediction, dtype=np.float32)
        if pred.ndim != 3:
            raise ValueError(f"Unexpected ONNX output shape: {pred.shape}")
        pred = pred[0]
        channels = 4 + len(self.names)
        if pred.shape[0] == channels:
            pred = pred.T
        elif pred.shape[1] != channels:
            raise ValueError(f"Unexpected ONNX output shape: {prediction.shape}")

        raw_boxes = pred[:, :4]
        scores = pred[:, 4:]
        if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
            scores = 1.0 / (1.0 + np.exp(-scores))

        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        keep = confidences >= self.conf
        if not np.any(keep):
            return []

        raw_boxes = raw_boxes[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        cx, cy, bw, bh = raw_boxes.T
        boxes = np.stack(
            (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), axis=1
        )
        order = np.argsort(-confidences)
        boxes = boxes[order]
        confidences = confidences[order]
        class_ids = class_ids[order]

        indices = self._nms(boxes, confidences)
        height, width = image_shape
        pad_x, pad_y = pad
        results: List[Dict[str, Any]] = []
        for index in indices:
            x1, y1, x2, y2 = boxes[index]
            x1 = float(np.clip((x1 - pad_x) / scale, 0.0, width))
            x2 = float(np.clip((x2 - pad_x) / scale, 0.0, width))
            y1 = float(np.clip((y1 - pad_y) / scale, 0.0, height))
            y2 = float(np.clip((y2 - pad_y) / scale, 0.0, height))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                {
                    "bbox_pixels": [x1, y1, x2, y2],
                    "bbox_norm": [x1 / width, y1 / height, x2 / width, y2 / height],
                    "confidence": float(confidences[index]),
                    "class_id": int(class_ids[index]),
                }
            )
        return results

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
        order = np.argsort(-scores)
        keep: List[int] = []

        while order.size:
            index = int(order[0])
            keep.append(index)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[index], x1[rest])
            yy1 = np.maximum(y1[index], y1[rest])
            xx2 = np.minimum(x2[index], x2[rest])
            yy2 = np.minimum(y2[index], y2[rest])
            inter = np.maximum(xx2 - xx1, 0.0) * np.maximum(yy2 - yy1, 0.0)
            union = areas[index] + areas[rest] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            order = rest[iou <= self.iou]
        return keep


_DETECTOR: Optional[OnnxFractureDetector] = None
_LOAD_ERROR: Optional[str] = None


def get_detector() -> Optional[OnnxFractureDetector]:
    global _DETECTOR, _LOAD_ERROR
    if not USE_LOCAL_ONNX:
        return None
    if _DETECTOR is not None:
        return _DETECTOR
    if _LOAD_ERROR is not None:
        return None
    try:
        _DETECTOR = OnnxFractureDetector()
    except DetectorUnavailable as exc:
        _LOAD_ERROR = str(exc)
        print(f"[detector] local ONNX disabled: {exc}")
        return None
    print(f"[detector] loaded {_DETECTOR.path.name} at {_DETECTOR.imgsz}px")
    return _DETECTOR
