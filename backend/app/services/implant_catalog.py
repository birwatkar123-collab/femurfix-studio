"""Parametric, in-house generic fixation catalog.

Every device is generated procedurally in the canonical femur frame (millimetres, z = 0 at the
distal end, z = 450 at the proximal tip, +x = medial). No proprietary CAD is used.

The catalog is intentionally generic and explicitly *educational*: these are representative
constructs, not specific commercial implants.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh

Point = Tuple[float, float, float]


def _cylinder_between(p1: Point, p2: Point, radius: float, sections: int = 24) -> trimesh.Trimesh:
    a = np.array(p1, dtype=float)
    b = np.array(p2, dtype=float)
    vector = b - a
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        raise ValueError("Cannot build a zero-length cylinder")
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    transform = trimesh.geometry.align_vectors([0, 0, 1], vector / length)
    cylinder.apply_transform(transform)
    cylinder.apply_translation((a + b) / 2.0)
    return cylinder


def _box(center: Point, extents: Point) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    return box


def _merge(parts: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


# --------------------------------------------------------------------------- devices
def _cannulated_screws() -> Dict[str, trimesh.Trimesh]:
    parts: Dict[str, trimesh.Trimesh] = {}
    specs = [
        ("screw_1", (-20, -6, 424), (40, 2, 438), 3.6),
        ("screw_2", (-20, 2, 410), (38, 0, 420), 3.6),
        ("screw_3", (-20, 8, 394), (34, -2, 402), 3.4),
    ]
    for name, a, b, r in specs:
        parts[name] = _cylinder_between(a, b, r)
    parts["washers"] = _merge(
        [_cylinder_between((-21, y, z - 1), (-21, y, z + 1), 5.0, sections=20) for y, z in [(-6, 424), (2, 410), (8, 394)]]
    )
    return parts


def _dynamic_hip_screw() -> Dict[str, trimesh.Trimesh]:
    parts = {
        "lag_screw": _cylinder_between((-26, 0, 402), (44, 4, 428), 6.2, sections=28),
        "lag_screw_tip": _cylinder_between((44, 4, 428), (50, 4, 432), 3.6, sections=20),
        "side_plate": _box((-24, 0, 342), (7.0, 20.0, 120.0)),
    }
    for index, z in enumerate((380, 352, 324, 296), start=1):
        parts[f"plate_screw_{index}"] = _cylinder_between((-30, 0, z), (-2, 0, z + 4), 3.2)
    return parts


def _cephalomedullary_nail() -> Dict[str, trimesh.Trimesh]:
    parts = {
        "nail": _cylinder_between((4, 0, 132), (8, 0, 402), 6.4, sections=28),
        "lag_screw": _cylinder_between((4, 0, 396), (44, 4, 422), 6.0, sections=28),
        "set_screw": _cylinder_between((4, 0, 372), (-16, 0, 376), 3.2),
    }
    for index, z in enumerate((176, 148), start=1):
        parts[f"distal_locking_screw_{index}"] = _cylinder_between((-30, 0, z), (30, 0, z), 3.5)
    return parts


def _intramedullary_nail() -> Dict[str, trimesh.Trimesh]:
    parts = {
        "nail": _cylinder_between((0, 0, 84), (0, 0, 404), 6.2, sections=28),
        "proximal_locking_screw": _cylinder_between((-34, 0, 384), (34, 0, 384), 3.6),
        "distal_locking_screw_1": _cylinder_between((-32, 0, 132), (32, 0, 132), 3.6),
        "distal_locking_screw_2": _cylinder_between((-32, 0, 106), (32, 0, 106), 3.6),
    }
    return parts


def _proximal_locking_plate() -> Dict[str, trimesh.Trimesh]:
    parts = {"plate": _box((-27, 0, 366), (6.5, 18.0, 128.0))}
    for index, (y, z) in enumerate(((4, 412), (6, 396), (2, 378)), start=1):
        parts[f"proximal_screw_{index}"] = _cylinder_between((-28, y, z), (36, y - 2, z + 14), 3.4)
    for index, z in enumerate((350, 326, 302), start=1):
        parts[f"shaft_screw_{index}"] = _cylinder_between((-30, 0, z), (-2, 0, z), 3.3)
    return parts


def _locking_plate() -> Dict[str, trimesh.Trimesh]:
    parts = {"plate": _box((-26, 0, 280), (6.5, 17.0, 180.0))}
    for index, z in enumerate((356, 322, 288, 254, 220), start=1):
        parts[f"locking_screw_{index}"] = _cylinder_between((-29, 0, z), (2, 0, z), 3.4)
    return parts


def _distal_femur_plate() -> Dict[str, trimesh.Trimesh]:
    parts = {
        "plate": _box((-26, 0, 66), (6.5, 19.0, 128.0)),
        "condylar_screw_1": _cylinder_between((-29, -8, 30), (24, -6, 24), 3.6),
        "condylar_screw_2": _cylinder_between((-29, 8, 30), (24, 6, 24), 3.6),
    }
    for index, z in enumerate((56, 82, 108), start=1):
        parts[f"shaft_screw_{index}"] = _cylinder_between((-29, 0, z), (0, 0, z), 3.4)
    return parts


def _scaled(parts: Dict[str, trimesh.Trimesh], scale: Tuple[float, float, float]) -> Dict[str, trimesh.Trimesh]:
    out: Dict[str, trimesh.Trimesh] = {}
    for name, mesh in parts.items():
        clone = mesh.copy()
        clone.apply_scale(scale)
        out[name] = clone
    return out


# --------------------------------------------------------------------------- catalog
CATALOG: Dict[str, Dict[str, Any]] = {
    "cannulated_screws": {
        "name": "Cannulated Screws (3)",
        "family": "cannulated_screws",
        "regions": ["proximal"],
        "description": "Three partially threaded cannulated screws for undisplaced or minimally displaced femoral neck fractures.",
        "patient_text": "Three small screws hold the ball of the hip in place while the bone heals.",
        "surgeon_text": "Standard 6.5/7.3 mm cannulated screw construct; suitable for valgus-impacted or undisplaced transcervical fractures.",
        "builder": _cannulated_screws,
        "insertion": "lateral",
    },
    "dynamic_hip_screw": {
        "name": "Dynamic Hip Screw (DHS)",
        "family": "dynamic_hip_screw",
        "regions": ["proximal"],
        "description": "Sliding lag screw with a lateral side plate, allowing controlled impaction across an intertrochanteric fracture.",
        "patient_text": "A metal plate on the outside of the thigh with a large screw into the hip ball, letting the bone settle as it heals.",
        "surgeon_text": "95-degree DHS with a 4-hole side plate; appropriate for stable intertrochanteric patterns.",
        "builder": _dynamic_hip_screw,
        "insertion": "lateral",
    },
    "cephalomedullary_nail": {
        "name": "Cephalomedullary Nail",
        "family": "cephalomedullary_nail",
        "regions": ["proximal"],
        "description": "Intramedullary nail with a lag screw into the femoral head and distal locking screws.",
        "patient_text": "A rod inside the thigh bone with a screw into the hip ball; a strong option for unstable fractures.",
        "surgeon_text": "Short or long cephalomedullary nail with a helical/lag element; indicated for unstable intertrochanteric and subtrochanteric fractures.",
        "builder": _cephalomedullary_nail,
        "insertion": "axial",
    },
    "proximal_locking_plate": {
        "name": "Proximal Femoral Locking Plate",
        "family": "proximal_locking_plate",
        "regions": ["proximal"],
        "description": "Contoured lateral plate with proximal locking screws into the head/neck and shaft screws.",
        "patient_text": "A shaped plate on the side of the upper thigh with screws into the hip area.",
        "surgeon_text": "Proximal femoral locking plate for complex or comminuted proximal fractures where nailing is unfavourable.",
        "builder": _proximal_locking_plate,
        "insertion": "lateral",
    },
    "intramedullary_nail": {
        "name": "Intramedullary Nail",
        "family": "intramedullary_nail",
        "regions": ["shaft"],
        "description": "Load-sharing rod inside the medullary canal with proximal and distal locking screws.",
        "patient_text": "A metal rod inside the thigh bone, held by screws at each end.",
        "surgeon_text": "Antegrade reamed intramedullary nail for simple and wedge femoral shaft fractures.",
        "builder": _intramedullary_nail,
        "insertion": "axial",
    },
    "locking_plate": {
        "name": "Shaft Locking Plate",
        "family": "locking_plate",
        "regions": ["shaft"],
        "description": "Lateral locking plate spanning the femoral shaft.",
        "patient_text": "A plate on the outside of the thigh bone with a row of screws.",
        "surgeon_text": "Lateral locking compression plate; useful for comminuted or periprosthetic shaft fractures.",
        "builder": _locking_plate,
        "insertion": "lateral",
    },
    "distal_femur_plate": {
        "name": "Distal Femoral Locking Plate",
        "family": "distal_femur_plate",
        "regions": ["distal"],
        "description": "Contoured distal plate with condylar and shaft locking screws.",
        "patient_text": "A plate shaped to the lower thigh with screws above the knee.",
        "surgeon_text": "Distal femoral locking plate for extra-articular and select intra-articular distal fractures.",
        "builder": _distal_femur_plate,
        "insertion": "lateral",
    },
}

# Region fit used by the recommender (per fracture pattern nuance).
AO_FIT: Dict[str, List[str]] = {
    "31-A1": ["dynamic_hip_screw", "cannulated_screws"],
    "31-A2": ["dynamic_hip_screw", "cephalomedullary_nail"],
    "31-A3": ["cephalomedullary_nail", "proximal_locking_plate"],
    "31-B": ["cannulated_screws", "cephalomedullary_nail"],
    "31-C": ["cephalomedullary_nail", "proximal_locking_plate"],
    "32-A": ["intramedullary_nail", "locking_plate"],
    "32-B": ["intramedullary_nail", "locking_plate"],
    "32-C": ["locking_plate", "intramedullary_nail"],
    "33-A": ["distal_femur_plate", "cephalomedullary_nail"],
    "33-B": ["distal_femur_plate"],
    "33-C": ["distal_femur_plate"],
}


def build_device(device_id: str, scale: Tuple[float, float, float]) -> Dict[str, trimesh.Trimesh]:
    if device_id not in CATALOG:
        raise KeyError(f"Unknown implant '{device_id}'")
    parts = CATALOG[device_id]["builder"]()
    return _scaled(parts, scale)


def public_catalog() -> List[Dict[str, Any]]:
    return [
        {
            "id": key,
            "name": value["name"],
            "family": value["family"],
            "regions": value["regions"],
            "description": value["description"],
            "patient_text": value["patient_text"],
            "surgeon_text": value["surgeon_text"],
            "insertion": value["insertion"],
        }
        for key, value in CATALOG.items()
    ]
