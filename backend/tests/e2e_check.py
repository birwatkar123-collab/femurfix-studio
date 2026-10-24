#!/usr/bin/env python3
"""End-to-end check without the HTTP server: synthetic X-ray -> analysis -> GLB assembly."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.assembly import Assembler  # noqa: E402
from app.services.bone_reconstructor import BoneReconstructor  # noqa: E402
from app.services.gltf_exporter import GLTFExporter  # noqa: E402
from app.services.image_loader import ImageLoader  # noqa: E402
from app.services.recommender import Recommender  # noqa: E402
from app.services.xray_analyzer import XRayAnalyzer  # noqa: E402
from app.utils.config import OUTPUT_DIR  # noqa: E402
from make_synthetic_xray import build_xray  # noqa: E402


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="femurfix_"))
    image_path = work / "synth.png"
    Image.fromarray(build_xray(fracture=True).astype(np.uint8)).save(image_path)

    loader = ImageLoader()
    gray, meta = loader.load_xray(image_path)
    print(f"image: {gray.shape} {meta}")

    analysis = XRayAnalyzer().analyze(gray, meta, image_path=str(image_path))
    print(f"engine: {analysis['engine']}")
    print(f"fracture: {analysis['fracture']}")
    print(f"measurements: {analysis['measurements']}")

    reconstruction = BoneReconstructor().reconstruct_from_xray(
        analysis["silhouette"], analysis["fracture"], analysis["measurements"]
    )
    bone = reconstruction["meshes"]
    frame = reconstruction["frame"]
    print(f"reconstruction: {frame['mode']} nodes={list(bone)} tris={[len(m.faces) for m in bone.values()]}")

    recommendations = Recommender().recommend(analysis["fracture"], analysis["measurements"])
    print(f"AO: {recommendations['ao_family']} options={[o['id'] for o in recommendations['options']]}")

    exporter = GLTFExporter(str(OUTPUT_DIR))
    preview = exporter.export_scene(bone, filename="e2e_preview.glb")
    print(f"preview: {preview} ({Path(preview).stat().st_size / 1024:.0f} KB)")

    chosen = recommendations["options"][0]["id"]
    meshes, layout = Assembler().assemble(bone, frame, chosen)
    assembly = exporter.export_scene(meshes, filename="e2e_assembly.glb")
    print(f"assembly: {assembly} ({Path(assembly).stat().st_size / 1024:.0f} KB)")
    print(f"device: {layout['device_name']} nodes={len(layout['nodes'])}")
    print("E2E OK")


if __name__ == "__main__":
    main()
