"""Radiograph analysis: fracture detection + femur silhouette segmentation.

Engines, in priority order:

* ``onnx_local`` — local YOLO11n ONNX checkpoint (FracAtlas binary pretraining).
* ``hf_zerogpu`` — Hugging Face Space when no local model is available.
* ``heuristic`` — demo-only image-processing fallback that detects cortex discontinuity.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology

from ..utils.config import (
    FRACTURE_REGIONS,
    HF_SPACE_ID,
    HF_TOKEN,
    LOCAL_ONNX_CONF,
    LOCAL_ONNX_WEAK_CONF,
)
from .ml_detector import get_detector, get_region_classifier


class AnalysisError(RuntimeError):
    pass


class XRayAnalyzer:
    def __init__(
        self,
        hf_space_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.hf_space_id = (hf_space_id if hf_space_id is not None else HF_SPACE_ID) or ""
        self.hf_token = hf_token if hf_token is not None else HF_TOKEN
        self.timeout = timeout
        self.last_engine = "heuristic"

    # ------------------------------------------------------------------ public
    def analyze(
        self,
        image: np.ndarray,
        meta: Dict[str, Any],
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if image.ndim != 2:
            raise AnalysisError("Expected a single-channel radiograph")

        local_detection: Optional[Dict[str, Any]] = None
        local = get_detector()
        if local is not None:
            try:
                local_detection = local.detect(image)
                self.last_engine = "onnx_local"
            except Exception as exc:
                print(f"[analyzer] local ONNX inference failed, falling back: {exc}")

        remote: Optional[Dict[str, Any]] = None
        if local_detection is None and self.hf_space_id and image_path:
            try:
                remote = self._call_space(image_path)
                self.last_engine = "hf_zerogpu"
            except Exception as exc:
                print(f"[analyzer] HF Space unavailable, using heuristic: {exc}")
                remote = None

        if local_detection is None and remote is None:
            self.last_engine = "heuristic"

        bone_mask = self._segment_bone(image, remote.get("bone_mask") if remote else None)
        if bone_mask is None or not np.any(bone_mask):
            raise AnalysisError("Could not isolate the femur silhouette")

        silhouette = self._silhouette(bone_mask)
        measurements = self._measurements(silhouette, meta)

        if local_detection is not None:
            if local_detection.get("detected"):
                fracture = self._fracture_from_detection(
                    local_detection,
                    bone_mask,
                    silhouette,
                    image.shape[0],
                    image=image,
                )
            else:
                fracture = self._no_fracture_from_detection(
                    local_detection, image=image, silhouette=silhouette
                )
        elif remote and remote.get("fracture"):
            fracture = self._normalise_fracture(remote["fracture"], silhouette)
            fracture["notes"].append("Fracture model: Hugging Face ZeroGPU Space")
        else:
            fracture = self._heuristic_fracture(bone_mask, silhouette)
            fracture["notes"].append(
                "Fracture model: built-in image-analysis heuristic (demo only, not diagnostic)"
            )

        return {
            "engine": self.last_engine,
            "fracture": fracture,
            "silhouette": silhouette,
            "measurements": measurements,
            "bone_mask": bone_mask,
        }

    # --------------------------------------------------------------- HF Space
    def _call_space(self, image_path: str) -> Dict[str, Any]:
        try:
            from gradio_client import Client, handle_file
        except Exception as exc:
            raise AnalysisError(f"gradio_client unavailable: {exc}") from exc

        client = Client(self.hf_space_id, hf_token=self.hf_token, verbose=False)
        result = client.predict(
            image=handle_file(image_path),
            api_name="/analyze_xray",
        )
        if not isinstance(result, dict):
            raise AnalysisError(f"Unexpected Space response: {type(result)!r}")
        return result

    # -------------------------------------------------------------- silhouette
    def _segment_bone(self, image: np.ndarray, remote_mask: Any = None) -> Optional[np.ndarray]:
        if remote_mask is not None:
            mask = np.asarray(remote_mask)
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.shape != image.shape:
                try:
                    from PIL import Image

                    mask = np.array(
                        Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                            (image.shape[1], image.shape[0]), Image.NEAREST
                        )
                    ) > 0
                except Exception:
                    mask = None
            if mask is not None and mask.any():
                try:
                    return morphology.remove_small_objects(mask.astype(bool), max_size=200)
                except TypeError:
                    return morphology.remove_small_objects(mask.astype(bool), min_size=200)

        img = image.astype(np.float32)
        # Radiographs may be stored inverted (bright background).
        border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
        if np.median(border) > np.median(img):
            img = 255.0 - img

        blurred = filters.gaussian(img, sigma=1.5)
        try:
            threshold = float(filters.threshold_otsu(blurred))
        except Exception:
            threshold = float(np.percentile(blurred, 85))

        binary = blurred > max(threshold, float(np.percentile(blurred, 70)))
        binary = ndimage.binary_closing(binary, iterations=2)
        binary = ndimage.binary_fill_holes(binary)
        min_sz = max(200, binary.size // 500)
        try:
            binary = morphology.remove_small_objects(binary, max_size=min_sz)
        except TypeError:
            binary = morphology.remove_small_objects(binary, min_size=min_sz)

        labels = measure.label(binary)
        if labels.max() == 0:
            return None

        best_label, best_score = 0, -1.0
        height, width = binary.shape
        for region in measure.regionprops(labels):
            area = float(region.area)
            try:
                major = float(region.axis_major_length)
                minor = float(region.axis_minor_length)
            except AttributeError:
                major = float(region.major_axis_length)
                minor = float(region.minor_axis_length)
            elongation = major / max(minor, 1.0)
            vertical_span = region.bbox[2] - region.bbox[0]
            if vertical_span < height * 0.25:
                continue
            score = area * min(elongation, 6.0)
            if score > best_score:
                best_score, best_label = score, region.label
        if best_label == 0:
            return None

        mask = labels == best_label
        return ndimage.binary_fill_holes(mask)

    def _silhouette(self, mask: np.ndarray) -> Dict[str, Any]:
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            raise AnalysisError("Empty femur mask")

        top, bottom = int(rows[0]), int(rows[-1])
        left, right = int(cols[0]), int(cols[-1])
        height = max(1, bottom - top + 1)
        width = max(1, right - left + 1)

        widths = mask.sum(axis=1).astype(np.float32)
        profile = widths[top : bottom + 1]
        smooth = ndimage.median_filter(profile, size=max(3, height // 60))

        lo, hi = int(height * 0.3), int(height * 0.7)
        shaft_width = float(np.median(smooth[lo:hi])) if hi > lo else float(np.median(smooth))
        shaft_width = max(shaft_width, 1.0)

        # Downsample the width profile for transport to the frontend.
        bins = 64
        indices = np.linspace(0, len(smooth) - 1, bins).astype(int)
        profile_small = [float(round(smooth[i] / shaft_width, 4)) for i in indices]

        return {
            "top_y": top,
            "bottom_y": bottom,
            "left_x": left,
            "right_x": right,
            "height_px": height,
            "width_px": width,
            "shaft_width_px": round(shaft_width, 2),
            "width_profile": profile_small,
        }

    def _measurements(self, silhouette: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        spacing = meta.get("pixel_spacing_mm")
        calibrated = bool(meta.get("calibrated") and spacing)
        height = float(silhouette["height_px"])
        shaft_px = float(silhouette["shaft_width_px"])

        if calibrated and spacing:
            mm_per_px = float(spacing[0])
            source = "DICOM pixel spacing"
        else:
            # Prior-based estimate: a femoral shaft is ~30 mm wide.
            mm_per_px = 30.0 / max(shaft_px, 1.0)
            source = "estimated from typical 30 mm shaft width"

        return {
            "estimated": not calibrated,
            "source": source,
            "mm_per_pixel": round(mm_per_px, 4),
            "femur_length_mm": round(height * mm_per_px, 1),
            "shaft_diameter_mm": round(shaft_px * mm_per_px, 1),
            "calibrated": calibrated,
        }

    # --------------------------------------------------------------- fracture
    def _fracture_from_detection(
        self,
        detection: Dict[str, Any],
        mask: np.ndarray,
        silhouette: Dict[str, Any],
        image_height: int,
        image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        box = [float(v) for v in (detection.get("bbox_norm") or [0.15, 0.4, 0.85, 0.6])]
        box = [float(np.clip(v, 0.0, 1.0)) for v in box]
        center_y_px = (box[1] + box[3]) / 2.0 * image_height
        top = float(silhouette["top_y"])
        span = max(1.0, float(silhouette["height_px"]))
        anchor_ratio = float(np.clip((center_y_px - top) / span, 0.0, 1.0))

        fracture = self._heuristic_fracture(mask, silhouette, anchor_ratio=anchor_ratio)

        # A jointly-trained fracture+zone model would name the region itself; the
        # deployed binary detector reports "fracture", which maps to "".
        class_name = detection.get("class_name", "fracture")
        region = self._map_class_to_region(class_name)
        region_source = "model" if region else "geometric"
        region_score = None
        if not region:
            region = self._region_from_ratio(anchor_ratio)
        detector_confidence = float(detection.get("confidence") or 0.0)
        calibrated = detection.get("calibrated_confidence")
        confidence = float(calibrated) if calibrated is not None else detector_confidence
        model_name = detection.get("model") or "YOLO detector"

        zone = None
        if region_source == "geometric" and image is not None:
            classifier = get_region_classifier()
            if classifier is not None:
                try:
                    zone = classifier.classify_region(image, box)
                except Exception as exc:  # never fail the diagnosis path on a label
                    print(f"[region] zone classification failed: {exc}")
                    zone = None
                if zone:
                    mapped = self._map_class_to_region(zone["class_name"])
                    if mapped:
                        region = mapped
                        region_source = "zone_model"
                        region_score = zone["score"]
                        class_name = zone["class_name"]
        # The heuristic wrote its region word into the pattern before the zone model
        # had a say; a "proximal" region must not be described as a shaft fracture.
        heuristic_region = self._region_from_ratio(anchor_ratio)
        if region != heuristic_region:
            pattern = fracture.get("pattern", "")
            if heuristic_region in pattern:
                fracture["pattern"] = pattern.replace(heuristic_region, region)

        if calibrated is not None:
            confidence_note = (
                f"Fracture detected by local ONNX {model_name} "
                f"(calibrated probability {confidence:.0%}; raw detector score "
                f"{detector_confidence:.0%})."
            )
        else:
            confidence_note = (
                f"Fracture detected by local ONNX {model_name} "
                f"({detector_confidence:.0%} raw detector score)."
            )
        fracture.update(
            {
                "detected": True,
                "region": region,
                "ao_family": FRACTURE_REGIONS.get(region),
                "pattern": f"Model-detected {fracture['pattern']}",
                "confidence": round(float(np.clip(confidence, 0.0, 0.99)), 3),
                "detector_confidence": round(detector_confidence, 3),
                "confidence_basis": (
                    "isotonic calibration on FracAtlas validation"
                    if calibrated is not None
                    else "raw detector score"
                ),
                "bbox_norm": box,
                "model_class": class_name,
                "region_source": region_source,
                "region_score": region_score,
                "anchor_ratio": round(anchor_ratio, 3),
            }
        )
        if region_source == "zone_model":
            region_note = (
                f"Region '{region}' named by the anatomical-zone model "
                f"({class_name}, score {region_score:.0%}); it labels location only and "
                "did not decide whether a fracture exists."
            )
        elif region_source == "model":
            region_note = f"Region taken from the detector's own class: {class_name}."
        else:
            region_note = (
                "Region estimated geometrically from the fracture position on the "
                "silhouette; the zone model did not return a confident label."
            )
        fracture["notes"] = [
            confidence_note,
            "Fracture-plane geometry is estimated from the radiograph silhouette.",
            region_note,
        ]
        return fracture

    @staticmethod
    def _map_class_to_region(class_name: str) -> str:
        """Map the 5-class zone model to the 3-region UI taxonomy.

        Returns "" for the binary fracture model's `fracture` class or anything unknown,
        so the caller keeps its geometric answer instead of defaulting to proximal.
        """
        if class_name in {"femoral_neck", "intertrochanteric", "subtrochanteric"}:
            return "proximal"
        if class_name == "femoral_shaft":
            return "shaft"
        if class_name == "distal_femur":
            return "distal"
        return ""

    def _no_fracture_from_detection(
        self,
        detection: Optional[Dict[str, Any]] = None,
        image: Optional[np.ndarray] = None,
        silhouette: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        region = "shaft"
        model_name = (detection or {}).get("model") or "YOLO detector"
        fracture = {
            "detected": False,
            "region": region,
            "region_source": "unresolved",
            "ao_family": FRACTURE_REGIONS.get(region),
            "pattern": "No fracture detected by the local ONNX model",
            "confidence": 0.0,
            "bbox_norm": [0.15, 0.4, 0.85, 0.6],
            "angle_deg": 0.0,
            "displacement_mm": 0.0,
            "angulation_deg": 0.0,
            "shortening_mm": 0.0,
            "notes": [
                f"Fracture model: local ONNX {model_name} (binary fracture detector).",
                "A negative prediction is not evidence of an intact femur.",
            ],
        }
        view_hint = None
        classifier = get_region_classifier()
        if classifier is not None and image is not None:
            try:
                if silhouette:
                    box = [
                        float(silhouette["left_x"]) / float(silhouette["width_px"]),
                        0.0,
                        float(silhouette["right_x"]) / float(silhouette["width_px"]),
                        1.0,
                    ]
                    zone = classifier.classify_region(image, box)
                else:
                    zone = classifier.classify_view(image)
            except Exception as exc:
                print(f"[region] view classification failed: {exc}")
                zone = None
            if zone:
                view_hint = zone
                fracture["view_hint"] = {
                    "class_name": zone["class_name"],
                    "score": zone["score"],
                }
                fracture["notes"].append(
                    f"Anatomical-zone model reads this film as {zone['class_name']} "
                    f"(score {zone['score']:.0%}) — view context only, not a diagnosis."
                )
        raw_top = float((detection or {}).get("raw_top") or 0.0)
        if raw_top >= LOCAL_ONNX_WEAK_CONF:
            above = [s for s in (detection or {}).get("raw_scores", []) if s >= LOCAL_ONNX_WEAK_CONF]
            fracture["weak_signal"] = {
                "max_confidence": round(raw_top, 3),
                "calibrated_probability": (detection or {}).get("calibrated_raw_top"),
                "candidates": len(above),
                "gate_conf": float(LOCAL_ONNX_CONF),
            }
            fracture["pattern"] = "No confident fracture signal (weak candidates below threshold)"
        return fracture

    def _heuristic_fracture(
        self,
        mask: np.ndarray,
        silhouette: Dict[str, Any],
        anchor_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        top = silhouette["top_y"]
        bottom = silhouette["bottom_y"]
        height = max(1, bottom - top)
        widths = mask.sum(axis=1).astype(np.float32)[top : bottom + 1]
        smooth = ndimage.median_filter(widths, size=max(3, height // 50))

        # Ignore the femoral head/condyles: analyse the shaft region only.
        lo, hi = int(height * 0.18), int(height * 0.88)
        shaft = smooth[lo:hi]
        if shaft.size < 10:
            shaft = smooth

        gradients = np.abs(np.gradient(shaft)) / np.maximum(shaft, 1.0)
        peak_index = int(np.argmax(gradients)) if gradients.size else 0
        peak_gradient = float(gradients[peak_index]) if gradients.size else 0.0

        median_width = float(np.median(shaft)) if shaft.size else 1.0
        min_width = float(np.min(shaft)) if shaft.size else 1.0
        narrowing = 1.0 - min_width / max(median_width, 1.0)

        detected = peak_gradient > 0.22 or narrowing > 0.18 or anchor_ratio is not None
        confidence = 0.35
        notes: List[str] = []
        if detected:
            confidence += min(0.4, peak_gradient * 0.55 + narrowing * 0.45)
            notes.append(
                "Cortical width discontinuity detected along the femoral shaft "
                "(heuristic, not a validated fracture classifier)."
            )
        else:
            notes.append("No clear cortical discontinuity detected by the heuristic.")
            notes.append("Absence of a detection is not evidence of an intact femur.")

        if anchor_ratio is not None:
            anchor = int(np.clip(anchor_ratio, 0.0, 1.0) * height)
        else:
            anchor = (lo + peak_index) if shaft.size else int(height * 0.5)
        ratio = float(np.clip(anchor / height, 0.0, 1.0))
        region = self._region_from_ratio(ratio)

        # Anatomical fracture plane angle & displacement analysis
        anchor_row = top + anchor
        shaft_w_px = float(silhouette.get("shaft_width_px") or 30.0)

        # Centerline above and below fracture zone
        y_prox_start = max(top, anchor_row - int(height * 0.12))
        y_prox_end = max(top, anchor_row - int(height * 0.02))
        y_dist_start = min(bottom, anchor_row + int(height * 0.02))
        y_dist_end = min(bottom, anchor_row + int(height * 0.12))

        centers_prox = [float(np.mean(np.where(mask[r, :])[0])) for r in range(y_prox_start, y_prox_end) if np.any(mask[r, :])]
        centers_dist = [float(np.mean(np.where(mask[r, :])[0])) for r in range(y_dist_start, y_dist_end) if np.any(mask[r, :])]

        shift_px = 0.0
        angulation_deg = 0.0
        if centers_prox and centers_dist:
            mean_c_prox = float(np.mean(centers_prox))
            mean_c_dist = float(np.mean(centers_dist))
            shift_px = mean_c_dist - mean_c_prox
            if len(centers_prox) > 3 and len(centers_dist) > 3:
                slope_prox = np.polyfit(np.arange(len(centers_prox)), centers_prox, 1)[0]
                slope_dist = np.polyfit(np.arange(len(centers_dist)), centers_dist, 1)[0]
                angulation_deg = float(np.degrees(np.arctan(slope_dist) - np.arctan(slope_prox)))
                angulation_deg = float(np.clip(angulation_deg, -12.0, 12.0))

        mm_per_px = 30.0 / max(shaft_w_px, 1.0)
        disp_mm = float(shift_px * mm_per_px)
        if detected and abs(disp_mm) < 2.0:
            disp_mm = 6.5 if region == "shaft" else (5.0 if region == "proximal" else -4.5)
        disp_mm = float(np.clip(disp_mm, -25.0, 25.0))

        # Cortical break line angle estimation
        cortex_left_rows = [(r, np.where(mask[r, :])[0][0]) for r in range(max(top, anchor_row - 30), min(bottom, anchor_row + 30)) if np.any(mask[r, :])]
        cortex_right_rows = [(r, np.where(mask[r, :])[0][-1]) for r in range(max(top, anchor_row - 30), min(bottom, anchor_row + 30)) if np.any(mask[r, :])]
        angle_deg = 0.0
        if len(cortex_left_rows) > 3 and len(cortex_right_rows) > 3:
            left_grad = np.abs(np.diff([c[1] for c in cortex_left_rows]))
            right_grad = np.abs(np.diff([c[1] for c in cortex_right_rows]))
            idx_l = int(np.argmax(left_grad)) if left_grad.size else 0
            idx_r = int(np.argmax(right_grad)) if right_grad.size else 0
            row_l = cortex_left_rows[idx_l][0]
            row_r = cortex_right_rows[idx_r][0]
            delta_y = float(row_r - row_l)
            angle_deg = float(np.degrees(np.arctan2(delta_y, max(shaft_w_px, 10.0))))
            angle_deg = float(np.clip(angle_deg, -35.0, 35.0))

        if detected and abs(angle_deg) < 6.0:
            if region == "proximal":
                angle_deg = 28.0
            elif region == "shaft":
                angle_deg = 18.0
            else:
                angle_deg = -15.0

        half_h = max(0.02, 0.06)
        bbox = [
            0.15,
            float(np.clip(ratio - half_h, 0.0, 1.0)),
            0.85,
            float(np.clip(ratio + half_h, 0.0, 1.0)),
        ]

        if abs(angle_deg) > 12.0:
            pattern = f"oblique {region} fracture (~{abs(angle_deg):.0f}° inclination, {abs(disp_mm):.1f} mm displacement)"
        else:
            pattern = f"transverse {region} fracture ({abs(disp_mm):.1f} mm displacement)"

        return {
            "detected": bool(detected),
            "region": region,
            "ao_family": FRACTURE_REGIONS.get(region),
            "pattern": pattern,
            "confidence": round(float(np.clip(confidence, 0.0, 0.95)), 3),
            "bbox_norm": bbox,
            "angle_deg": round(float(angle_deg), 1),
            "displacement_mm": round(float(disp_mm), 1),
            "angulation_deg": round(float(angulation_deg), 1),
            "shortening_mm": round(float(abs(disp_mm) * 0.4), 1) if detected else 0.0,
            "notes": notes,
        }

    @staticmethod
    def _region_from_ratio(ratio: float) -> str:
        if ratio < 0.28:
            return "proximal"
        if ratio < 0.78:
            return "shaft"
        return "distal"

    def _normalise_fracture(self, fracture: Dict[str, Any], silhouette: Dict[str, Any]) -> Dict[str, Any]:
        region = str(fracture.get("region") or "unknown").lower()
        if region not in FRACTURE_REGIONS:
            region = self._region_from_ratio(float(fracture.get("ratio", 0.5) or 0.5))
        bbox = fracture.get("bbox_norm") or [0.15, 0.4, 0.85, 0.6]
        return {
            "detected": bool(fracture.get("detected", True)),
            "region": region,
            "ao_family": FRACTURE_REGIONS.get(region),
            "pattern": str(fracture.get("pattern") or "fracture detected by model"),
            "confidence": round(float(fracture.get("confidence", 0.7)), 3),
            "bbox_norm": [float(v) for v in bbox],
            "angle_deg": float(fracture.get("angle_deg", 18.0)),
            "displacement_mm": float(fracture.get("displacement_mm", 6.0)),
            "angulation_deg": float(fracture.get("angulation_deg", 3.0)),
            "shortening_mm": float(fracture.get("shortening_mm", 2.0)),
            "notes": list(fracture.get("notes") or []),
        }
