"""Marching Cubes, smoothing and quadric decimation (region-agnostic)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh


class MeshError(RuntimeError):
    pass


class MeshProcessor:
    def __init__(self, decimation_ratio: float = 0.3, smooth_iterations: int = 5) -> None:
        if not 0.0 < decimation_ratio <= 1.0:
            raise ValueError("decimation_ratio must be in (0, 1]")
        self.decimation_ratio = decimation_ratio
        self.smooth_iterations = smooth_iterations

    def volume_to_mesh(
        self, mask: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> trimesh.Trimesh:
        if mask.ndim != 3:
            raise MeshError("Expected a 3D mask")
        if not np.any(mask):
            raise MeshError("Mask is empty")

        from skimage import measure

        try:
            verts, faces, normals, _ = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, spacing=spacing, allow_degenerate=False
            )
        except Exception as exc:
            raise MeshError(f"Marching Cubes failed: {exc}") from exc

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
        if mesh.is_empty:
            raise MeshError("Marching Cubes produced an empty mesh")
        return mesh

    def smooth(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        try:
            smoothed = mesh.smoothed(self.smooth_iterations)
        except Exception:
            return mesh
        return smoothed if not smoothed.is_empty else mesh

    def decimate(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        target = int(len(mesh.faces) * self.decimation_ratio)
        if target < 500:
            return mesh
        try:
            decimated = mesh.simplify_quadric_decimation(target)
        except Exception:
            return mesh
        return decimated if not decimated.is_empty else mesh

    def process(
        self, mask: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> trimesh.Trimesh:
        mesh = self.volume_to_mesh(mask, spacing)
        mesh = self.smooth(mesh)
        mesh = self.decimate(mesh)
        mesh.apply_translation(-mesh.centroid)
        return mesh
