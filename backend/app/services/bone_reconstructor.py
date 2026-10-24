"""3D bone reconstruction.

X-ray path
    A smooth signed-distance-field femur (anatomical shape prior) is generated and scaled to
    the measured silhouette. Single-view geometry is fundamentally estimated, so the caller
    must label it as such.

CT/MRI path
    Bone is thresholded (or segmented externally) and meshed with Marching Cubes at the true
    voxel spacing, which yields patient-specific geometry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from ..utils.config import (
    FEMUR_CANONICAL_LENGTH_MM,
    FEMUR_CANONICAL_SHAFT_DIAMETER_MM,
    HU_BONE_MIN,
    HU_METAL_MIN,
)
from .mesh_processor import MeshProcessor


class ReconstructionError(RuntimeError):
    pass


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum for organic SDF blending."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _sphere(px: np.ndarray, py: np.ndarray, pz: np.ndarray, center: Tuple[float, float, float], radius: float) -> np.ndarray:
    return np.sqrt((px - center[0]) ** 2 + (py - center[1]) ** 2 + (pz - center[2]) ** 2) - radius


def _capsule(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    ax, ay, az = a
    bx, by, bz = b
    abx, aby, abz = bx - ax, by - ay, bz - az
    apx, apy, apz = px - ax, py - ay, pz - az
    denom = abx * abx + aby * aby + abz * abz
    if denom <= 1e-9:
        return np.sqrt(apx**2 + apy**2 + apz**2) - radius
    t = np.clip((apx * abx + apy * aby + apz * abz) / denom, 0.0, 1.0)
    dx = apx - abx * t
    dy = apy - aby * t
    dz = apz - abz * t
    return np.sqrt(dx * dx + dy * dy + dz * dz) - radius


class BoneReconstructor:
    def __init__(self, target_faces: int = 22000) -> None:
        self.target_faces = target_faces

    # ---------------------------------------------------------------- X-ray
    def reconstruct_from_xray(
        self,
        silhouette: Dict[str, Any],
        fracture: Dict[str, Any],
        measurements: Dict[str, Any],
    ) -> Dict[str, Any]:
        mesh = self._parametric_femur()

        length_mm = float(measurements.get("femur_length_mm") or FEMUR_CANONICAL_LENGTH_MM)
        length_mm = float(np.clip(length_mm, 260.0, 600.0))
        shaft_mm = float(measurements.get("shaft_diameter_mm") or FEMUR_CANONICAL_SHAFT_DIAMETER_MM)
        shaft_mm = float(np.clip(shaft_mm, 20.0, 45.0))

        scale_z = length_mm / FEMUR_CANONICAL_LENGTH_MM
        scale_xy = shaft_mm / FEMUR_CANONICAL_SHAFT_DIAMETER_MM
        # Depth is unknown from a single projection; reuse the coronal width.
        mesh.apply_scale([scale_xy, scale_xy, scale_z])

        fracture_ratio = self._fracture_ratio(fracture)
        z_split = mesh.bounds[0][2] + (1.0 - fracture_ratio) * (mesh.bounds[1][2] - mesh.bounds[0][2])

        pieces = self._split(mesh, z_split) if fracture.get("detected") else [mesh]
        if len(pieces) == 2:
            meshes = {"femur_proximal": pieces[0], "femur_distal": pieces[1]}
        else:
            meshes = {"femur": mesh}

        frame = {
            "mode": "estimated_xray",
            "canonical_length_mm": FEMUR_CANONICAL_LENGTH_MM,
            "applied_scale": [round(scale_xy, 4), round(scale_xy, 4), round(scale_z, 4)],
            "length_mm": round(length_mm, 1),
            "shaft_diameter_mm": round(shaft_mm, 1),
            "fracture_z": round(float(z_split), 2) if len(pieces) == 2 else None,
            "fracture_ratio": round(float(fracture_ratio), 4),
        }
        return {"meshes": meshes, "frame": frame}

    def _parametric_femur(self, resolution: float = 2.0) -> trimesh.Trimesh:
        xs = np.arange(-62.0, 84.0, resolution)
        ys = np.arange(-48.0, 48.0, resolution)
        zs = np.arange(-12.0, 458.0, resolution)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")

        # Shaft (distal -> proximal), tapering handled by two blended capsules.
        dist = np.minimum(
            _capsule(xx, yy, zz, (0, 0, 30), (0, 0, 210), 16.0),
            _capsule(xx, yy, zz, (0, 0, 210), (0, 0, 392), 13.5),
        )
        # Distal condyles.
        condyles = np.minimum(
            _sphere(xx, yy, zz, (-22, 0, 6), 27.0),
            _sphere(xx, yy, zz, (22, 0, 6), 27.0),
        )
        d = _smin(dist, condyles, 16.0)
        # Femoral neck and head.
        neck = _capsule(xx, yy, zz, (6, 0, 382), (28, 3, 414), 16.0)
        head = _sphere(xx, yy, zz, (37, 4, 428), 24.0)
        d = _smin(d, neck, 14.0)
        d = _smin(d, head, 12.0)
        # Trochanters.
        greater = _sphere(xx, yy, zz, (29, -12, 396), 22.0)
        lesser = _sphere(xx, yy, zz, (-13, 6, 364), 11.0)
        d = _smin(d, greater, 14.0)
        d = _smin(d, lesser, 10.0)

        from skimage import measure

        try:
            verts, faces, normals, _ = measure.marching_cubes(-d, level=0.0, spacing=(resolution,) * 3)
        except Exception as exc:
            raise ReconstructionError(f"Shape-prior Marching Cubes failed: {exc}") from exc

        verts[:, 0] += xs[0]
        verts[:, 1] += ys[0]
        verts[:, 2] += zs[0]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
        if mesh.is_empty:
            raise ReconstructionError("Shape prior produced an empty mesh")

        try:
            if len(mesh.faces) > self.target_faces:
                mesh = mesh.simplify_quadric_decimation(self.target_faces)
        except Exception:
            pass
        return mesh

    # ------------------------------------------------------------- CT / MRI
    def reconstruct_from_volume(
        self,
        array: np.ndarray,
        spacing: Tuple[float, float, float],
        fracture: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        bone = array >= HU_BONE_MIN
        metal = array >= HU_METAL_MIN
        bone = bone & ~metal
        if not np.any(bone):
            raise ReconstructionError("No bone voxels above the HU threshold")

        processor = MeshProcessor(decimation_ratio=0.3, smooth_iterations=4)
        mesh = processor.process(bone, spacing)

        # Normalise to the same frame as the X-ray path: z=0 at the distal end,
        # x/y centred on the shaft, so the implant catalog aligns with the bone.
        bounds = mesh.bounds
        height = float(bounds[1][2] - bounds[0][2])
        scale = height / FEMUR_CANONICAL_LENGTH_MM if height > 1.0 else 1.0
        mesh.apply_translation(
            [
                -(bounds[0][0] + bounds[1][0]) / 2.0,
                -(bounds[0][1] + bounds[1][1]) / 2.0,
                -bounds[0][2],
            ]
        )

        fracture = fracture or {}
        ratio = self._fracture_ratio(fracture)
        pieces = [mesh]
        if fracture.get("detected"):
            low, high = mesh.bounds[0][2], mesh.bounds[1][2]
            z_split = low + (1.0 - ratio) * (high - low)
            pieces = self._split(mesh, z_split)

        if len(pieces) == 2:
            meshes = {"femur_proximal": pieces[0], "femur_distal": pieces[1]}
        else:
            meshes = {"femur": mesh}

        return {
            "meshes": meshes,
            "frame": {
                "mode": "ct_exact",
                "spacing": list(spacing),
                "applied_scale": [round(scale, 5), round(scale, 5), round(scale, 5)],
                "fracture_ratio": round(float(ratio), 4),
            },
        }

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _fracture_ratio(fracture: Dict[str, Any]) -> float:
        bbox = fracture.get("bbox_norm") or [0.15, 0.45, 0.85, 0.55]
        try:
            return float(np.clip((float(bbox[1]) + float(bbox[3])) / 2.0, 0.06, 0.94))
        except (TypeError, ValueError, IndexError):
            return 0.5

    @staticmethod
    def _split(mesh: trimesh.Trimesh, z_split: float) -> List[trimesh.Trimesh]:
        normal = np.array([0.0, 0.0, 1.0])
        origin = np.array([0.0, 0.0, float(z_split)])
        try:
            upper = trimesh.intersections.slice_mesh_plane(mesh, normal, origin, cap=True)
            lower = trimesh.intersections.slice_mesh_plane(mesh, -normal, origin, cap=True)
            if upper.is_empty or lower.is_empty:
                return [mesh]
            return [upper, lower]
        except Exception:
            # Fall back to a vertex split if the plane cut fails.
            centers = mesh.triangles_center[:, 2]
            upper = mesh.submesh([np.where(centers >= z_split)[0]], append=True)
            lower = mesh.submesh([np.where(centers < z_split)[0]], append=True)
            if upper.is_empty or lower.is_empty:
                return [mesh]
            if isinstance(upper, list):
                upper = trimesh.util.concatenate(upper)
            if isinstance(lower, list):
                lower = trimesh.util.concatenate(lower)
            return [upper, lower]
