"""Radiograph analysis: fracture detection + femur silhouette segmentation.

Two engines:

* ``hf_zerogpu`` — calls the Hugging Face Space (fine-tuned detector + bone segmentation).
* ``heuristic`` — deterministic image-processing fallback used when the Space is not
  configured or unreachable. Its output is explicitly marked as a demo heuristic.

The heuristic is NOT a trained diagnostic model. It detects cortex discontinuity and
silhouette narrowing so the rest of the pipeline can be exercised end-to-end.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import filters, measure, morphology

from ..utils.config import FRACTURE_REGIONS, HF_SPACE_ID, HF_TOKEN


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

        bone_mask: Optional[np.ndarray] = None
        remote: Optional[Dict[str, Any]] = None

        if self.hf_space_id and image_path:
            try:
                remote = self._call_space(image_path)
                self.last_engine = "hf_zerogpu"
            except Exception as exc:
                print(f"[analyzer] HF Space unavailable, using heuristic: {exc}")
                remote = None

        if remote is None:
            self.last_engine = "heuristic"

        bone_mask = self._segment_bone(image, remote.get("bone_mask") if remote else None)
        if bone_mask is None or not np.any(bone_mask):
            raise AnalysisError("Could not isolate the femur silhouette")

        silhouette = self._silhouette(bone_mask)
        measurements = self._measurements(silhouette, meta)

        if remote and remote.get("fracture"):
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
    def _heuristic_fracture(
        self, mask: np.ndarray, silhouette: Dict[str, Any]
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

        detected = peak_gradient > 0.22 or narrowing > 0.18
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

        anchor = (lo + peak_index) if shaft.size else int(height * 0.5)
        ratio = float(np.clip(anchor / height, 0.0, 1.0))
        region = self._region_from_ratio(ratio)

        half_h = max(0.02, 0.06)
        bbox = [
            0.15,
            float(np.clip(ratio - half_h, 0.0, 1.0)),
            0.85,
            float(np.clip(ratio + half_h, 0.0, 1.0)),
        ]

        pattern = "cortex discontinuity / possible transverse or oblique fracture"
        if narrowing > 0.22 and peak_gradient <= 0.28:
            pattern = "focal cortical narrowing / possible impacted fracture"

        return {
            "detected": bool(detected),
            "region": region,
            "ao_family": FRACTURE_REGIONS.get(region),
            "pattern": pattern,
            "confidence": round(float(np.clip(confidence, 0.0, 0.95)), 3),
            "bbox_norm": bbox,
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
            "notes": list(fracture.get("notes") or []),
        }
