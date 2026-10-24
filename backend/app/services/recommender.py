"""Explainable fixation recommender.

This is decision support for discussion, not a prescription. Every option is returned with the
reasons it was ranked and the caveats that limit its use.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .implant_catalog import AO_FIT, CATALOG


def _ao_code(fracture: Dict[str, Any]) -> str:
    region = str(fracture.get("region") or "shaft")
    pattern = str(fracture.get("pattern") or "").lower()
    confidence = float(fracture.get("confidence") or 0.0)

    if region == "proximal":
        if "narrow" in pattern or "impact" in pattern or confidence < 0.45:
            return "31-B"
        if "subtrochanter" in pattern:
            return "31-A3"
        return "31-A2"
    if region == "distal":
        if "articular" in pattern or "comminut" in pattern:
            return "33-C"
        return "33-A"
    if "comminut" in pattern:
        return "32-C"
    return "32-A"


class Recommender:
    def recommend(
        self,
        fracture: Dict[str, Any],
        measurements: Dict[str, Any],
        patient: Optional[Dict[str, Any]] = None,
        top_n: int = 4,
    ) -> Dict[str, Any]:
        patient = patient or {}
        code = _ao_code(fracture)
        shortlist = list(AO_FIT.get(code) or AO_FIT.get(code[:2] + "-A") or [])

        region = str(fracture.get("region") or "shaft")
        for device_id, meta in CATALOG.items():
            if region in meta["regions"] and device_id not in shortlist:
                shortlist.append(device_id)

        confidence = float(fracture.get("confidence") or 0.0)
        age = patient.get("age")
        options: List[Dict[str, Any]] = []

        for rank, device_id in enumerate(shortlist[:top_n]):
            meta = CATALOG.get(device_id)
            if not meta:
                continue
            base = 1.0 - rank * 0.09
            score = base * (0.65 + 0.35 * confidence)
            rationale = [f"Matched AO/OTA family {code} for a {region} femoral fracture."]

            if device_id == "cephalomedullary_nail" and region == "proximal":
                score += 0.05
                rationale.append("Load-sharing intramedullary device for unstable proximal patterns.")
            if device_id == "cannulated_screws" and code == "31-B":
                rationale.append("Screw-only fixation preserves the femoral head blood supply in undisplaced neck fractures.")
            if device_id == "dynamic_hip_screw" and code == "31-A2":
                rationale.append("DHS allows controlled collapse at the fracture site.")
            if device_id == "locking_plate" and code == "32-C":
                rationale.append("Plate fixation is preferred when the shaft is comminuted and cannot share load through a nail.")
            if isinstance(age, (int, float)) and age >= 65 and device_id == "cephalomedullary_nail":
                score += 0.04
                rationale.append("Age ≥ 65 favours a load-sharing nail in proximal fractures.")

            caveats = [
                f"Generic educational model — not a specific commercial {meta['name']}.",
                "Implant sizing requires calibrated imaging and surgeon judgement.",
            ]
            if measurements.get("estimated"):
                caveats.append("Measurements are estimated from a 2D radiograph without calibration.")

            options.append(
                {
                    "id": device_id,
                    "name": meta["name"],
                    "family": meta["family"],
                    "score": round(float(min(score, 1.0)), 3),
                    "confidence": round(float(min(0.55 + confidence * 0.4, 0.95)), 3),
                    "rationale": rationale,
                    "caveats": caveats,
                    "patient_text": meta["patient_text"],
                    "surgeon_text": meta["surgeon_text"],
                    "description": meta["description"],
                    "insertion": meta["insertion"],
                }
            )

        options.sort(key=lambda item: item["score"], reverse=True)
        return {
            "ao_family": code,
            "fracture_region": region,
            "options": options,
            "note": "Ranked options are for clinician discussion only and are not a treatment recommendation.",
        }
