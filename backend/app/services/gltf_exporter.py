"""Export one hierarchical GLB with semantic node names and PBR materials."""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import trimesh
from trimesh.visual.material import PBRMaterial

BONE_COLOR = [0.90, 0.84, 0.72, 1.0]
METAL_COLOR = [0.78, 0.80, 0.84, 1.0]


def _material_for(name: str) -> PBRMaterial:
    lowered = name.lower()
    if any(token in lowered for token in ("implant", "plate", "screw", "nail", "hardware")):
        return PBRMaterial(
            baseColorFactor=METAL_COLOR,
            metallicFactor=1.0,
            roughnessFactor=0.25,
        )
    return PBRMaterial(
        baseColorFactor=BONE_COLOR,
        metallicFactor=0.05,
        roughnessFactor=0.8,
    )


class GLTFExporter:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_scene(
        self,
        meshes: Dict[str, trimesh.Trimesh],
        filename: str = "scene.glb",
        root_name: str = "femur_fixation",
    ) -> str:
        if not meshes:
            raise ValueError("Nothing to export: mesh dictionary is empty")

        scene = trimesh.Scene()
        for name, mesh in meshes.items():
            if mesh is None or mesh.is_empty:
                continue
            mesh = mesh.copy()
            try:
                mesh.visual = trimesh.visual.TextureVisuals(material=_material_for(name))
            except Exception:
                pass
            scene.add_geometry(mesh, node_name=name, geom_name=name)

        if not scene.geometry:
            raise ValueError("All meshes were empty; nothing exported")

        scene.graph.name = root_name
        out_path = os.path.join(self.output_dir, filename)
        tmp_path = out_path + ".tmp"
        try:
            scene.export(tmp_path, file_type="glb")
            os.replace(tmp_path, out_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return out_path

    def export_single(self, mesh: trimesh.Trimesh, name: str, filename: str) -> str:
        return self.export_scene({name: mesh}, filename=filename)
