"""Assemble fracture fragments and a selected implant into one animated scene."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import trimesh

from .implant_catalog import CATALOG, build_device


class AssemblyError(RuntimeError):
    pass


class Assembler:
    def assemble(
        self,
        bone_meshes: Dict[str, trimesh.Trimesh],
        frame: Dict[str, Any],
        device_id: str,
    ) -> Tuple[Dict[str, trimesh.Trimesh], Dict[str, Any]]:
        if device_id not in CATALOG:
            raise AssemblyError(f"Unknown implant '{device_id}'")
        if not bone_meshes:
            raise AssemblyError("No bone geometry available; run analysis first")

        scale = tuple(float(v) for v in (frame.get("applied_scale") or (1.0, 1.0, 1.0)))
        if len(scale) != 3 or any(v <= 0 for v in scale):
            raise AssemblyError("Invalid reconstruction scale")

        meshes: Dict[str, trimesh.Trimesh] = {}
        for name, mesh in bone_meshes.items():
            if mesh is None or mesh.is_empty:
                continue
            meshes[name] = mesh.copy()

        try:
            parts = build_device(device_id, scale)  # type: ignore[arg-type]
        except Exception as exc:
            raise AssemblyError(f"Failed to build '{device_id}': {exc}") from exc

        for part_name, part in parts.items():
            meshes[f"implant_{device_id}_{part_name}"] = part

        combined = trimesh.util.concatenate(list(meshes.values()))
        bounds = combined.bounds.tolist()
        metadata = {
            "device_id": device_id,
            "device_name": CATALOG[device_id]["name"],
            "insertion": CATALOG[device_id]["insertion"],
            "nodes": list(meshes.keys()),
            "bone_nodes": [n for n in meshes if not n.startswith("implant_")],
            "implant_nodes": [n for n in meshes if n.startswith("implant_")],
            "scale": [round(float(v), 5) for v in scale],
            "bounds": [[round(float(v), 3) for v in row] for row in bounds],
            "fracture_z": frame.get("fracture_z"),
            "estimated": frame.get("mode") == "estimated_xray",
            "units": "mm",
        }
        return meshes, metadata
