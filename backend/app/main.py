"""FemurFix Studio API.

Flow: upload -> analyze (fracture + estimated 3D) -> recommend -> assemble (animated GLB).
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .services.assembly import Assembler, AssemblyError
from .services.bone_reconstructor import BoneReconstructor, ReconstructionError
from .services.gltf_exporter import GLTFExporter
from .services.image_loader import ImageLoader, ImageLoadError
from .services.implant_catalog import public_catalog
from .services.recommender import Recommender
from .services.xray_analyzer import AnalysisError, XRayAnalyzer
from .utils.config import (
    CASE_DIR,
    DISCLAIMER,
    FRACTURE_REGIONS,
    MAX_UPLOAD_BYTES,
    OUTPUT_DIR,
    UPLOAD_DIR,
)

CASE_ID_RE = re.compile(r"^[a-f0-9]{12}$")

app = FastAPI(title="FemurFix Studio API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

loader = ImageLoader()
exporter = GLTFExporter(str(OUTPUT_DIR))


class AssembleRequest(BaseModel):
    implant_id: str = Field(..., min_length=1, max_length=64)
    audience: str = Field("surgeon", pattern="^(surgeon|patient)$")


class AnalyzeRequest(BaseModel):
    region_hint: Optional[str] = Field(None, pattern="^(proximal|shaft|distal)$")
    patient_age: Optional[int] = Field(None, ge=0, le=120)


def _case_dir(case_id: str) -> Path:
    if not CASE_ID_RE.match(case_id):
        raise HTTPException(status_code=400, detail="Invalid case id")
    directory = CASE_DIR / case_id
    if not directory.exists():
        raise HTTPException(status_code=404, detail="Case not found")
    return directory


def _read_state(case_id: str) -> Dict[str, Any]:
    path = _case_dir(case_id) / "case.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Case has not been analysed yet")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt case state: {exc}") from exc


def _write_state(directory: Path, state: Dict[str, Any]) -> None:
    (directory / "case.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _save_meshes(directory: Path, meshes: Dict[str, trimesh.Trimesh]) -> List[Dict[str, Any]]:
    target = directory / "meshes"
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.stl"):
        old.unlink(missing_ok=True)
    summary = []
    for name, mesh in meshes.items():
        if mesh is None or mesh.is_empty:
            continue
        mesh.export(target / f"{name}.stl")
        summary.append({"name": name, "triangles": int(len(mesh.faces))})
    return summary


def _load_meshes(directory: Path) -> Dict[str, trimesh.Trimesh]:
    meshes: Dict[str, trimesh.Trimesh] = {}
    for path in sorted((directory / "meshes").glob("*.stl")):
        try:
            meshes[path.stem] = trimesh.load(path, force="mesh")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load stored mesh: {exc}") from exc
    return meshes


def _public_analysis(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "bone_mask"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "femurfix-studio"}


@app.get("/api/meta")
def meta() -> Dict[str, Any]:
    return {
        "disclaimer": DISCLAIMER,
        "catalog": public_catalog(),
        "regions": {key: f"AO/OTA {value}" for key, value in FRACTURE_REGIONS.items()},
        "limits": {
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "estimated_geometry": "X-ray reconstructions are estimated anatomical priors.",
        },
    }


@app.post("/api/cases", status_code=201)
async def create_case(file: UploadFile = File(...)) -> Dict[str, Any]:
    case_id = uuid.uuid4().hex[:12]
    directory = CASE_DIR / case_id
    directory.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload").suffix.lower() or ".bin"
    source = directory / f"source{suffix}"
    try:
        with source.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    except Exception as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Upload failed: {exc}") from exc

    size = source.stat().st_size
    if size == 0:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if size > MAX_UPLOAD_BYTES:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    try:
        kind = loader.detect_kind(source)
    except Exception:
        kind = "xray"

    state = {
        "case_id": case_id,
        "kind": kind,
        "source": {"filename": file.filename, "size": size, "path": source.name},
        "created": datetime.now(timezone.utc).isoformat(),
        "analysis": None,
        "recommendations": None,
        "preview_glb": None,
        "assemblies": {},
    }
    _write_state(directory, state)
    return {
        "case_id": case_id,
        "kind": kind,
        "filename": file.filename,
        "size": size,
    }


@app.post("/api/cases/{case_id}/analyze")
def analyze_case(case_id: str, payload: Optional[AnalyzeRequest] = None) -> Dict[str, Any]:
    directory = _case_dir(case_id)
    state = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    payload = payload or AnalyzeRequest()

    source = directory / state["source"]["path"]
    if not source.exists():
        raise HTTPException(status_code=404, detail="Stored source image is missing")
    kind = state.get("kind", "xray")

    analyzer = XRayAnalyzer()
    reconstructor = BoneReconstructor()
    recommender = Recommender()

    try:
        if kind == "volume":
            image, array, spacing, imaging_meta = loader.load_volume(source)
            region = payload.region_hint or "shaft"
            fracture = {
                "detected": False,
                "region": region,
                "ao_family": FRACTURE_REGIONS.get(region),
                "pattern": "volumetric study — fracture classification pending",
                "confidence": 0.0,
                "bbox_norm": [0.15, 0.4, 0.85, 0.6],
                "notes": [
                    "Fracture detection is optimised for radiographs in this release.",
                    "Volumetric geometry is patient-specific and derived from the CT/MRI voxels.",
                ],
            }
            reconstruction = reconstructor.reconstruct_from_volume(array, spacing, fracture)
            analysis_payload: Dict[str, Any] = {
                "engine": "ct_threshold_segmentation",
                "imaging": imaging_meta,
            }
        else:
            image, imaging_meta = loader.load_xray(source)
            analysis_payload = analyzer.analyze(image, imaging_meta, image_path=str(source))
            fracture = analysis_payload["fracture"]
            if payload.region_hint and payload.region_hint != fracture.get("region"):
                fracture = dict(fracture)
                fracture["region"] = payload.region_hint
                fracture["ao_family"] = FRACTURE_REGIONS.get(payload.region_hint)
                fracture.setdefault("notes", []).append(
                    f"Region overridden by the user to '{payload.region_hint}'."
                )
                analysis_payload["fracture"] = fracture
            reconstruction = reconstructor.reconstruct_from_xray(
                analysis_payload["silhouette"],
                fracture,
                analysis_payload["measurements"],
            )
            analysis_payload["imaging"] = imaging_meta

        bone_meshes = reconstruction["meshes"]
        preview_meshes = reconstruction.get("preview_meshes", bone_meshes)
        frame = reconstruction["frame"]

        preview_name = f"preview_{case_id}.glb"
        exporter.export_scene(preview_meshes, filename=preview_name, root_name=f"case_{case_id}")

        structures = _save_meshes(directory, bone_meshes)
        fracture_detected = bool(fracture.get("detected"))
        if fracture_detected or kind == "volume":
            recommendations = recommender.recommend(
                fracture,
                analysis_payload.get("measurements", {}),
                patient={"age": payload.patient_age} if payload.patient_age is not None else {},
            )
        else:
            recommendations = {
                "ao_family": fracture.get("ao_family"),
                "fracture_region": fracture.get("region"),
                "options": [],
                "withheld": True,
                "note": (
                    "Fixation recommendations are withheld because the local model did not detect a "
                    "fracture. A negative prediction is not evidence of an intact femur."
                ),
            }

        state.update(
            {
                "analysis": _public_analysis(analysis_payload),
                "frame": frame,
                "structures": structures,
                "recommendations": recommendations,
                "preview_glb": f"/outputs/{preview_name}",
                "analysed": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_state(directory, state)

        return {
            "case_id": case_id,
            "kind": kind,
            "analysis": state["analysis"],
            "reconstruction": {
                "mode": frame.get("mode"),
                "estimated": frame.get("mode") == "estimated_xray",
                "frame": frame,
                "structures": structures,
            },
            "preview_glb_url": state["preview_glb"],
            "recommendations": recommendations,
            "disclaimer": DISCLAIMER,
        }
    except (ImageLoadError, AnalysisError, ReconstructionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> Dict[str, Any]:
    state = _read_state(case_id)
    return {
        "case_id": case_id,
        "kind": state.get("kind"),
        "source": state.get("source"),
        "analysis": state.get("analysis"),
        "recommendations": state.get("recommendations"),
        "preview_glb_url": state.get("preview_glb"),
        "assemblies": state.get("assemblies", {}),
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/cases/{case_id}/assemble")
def assemble_case(case_id: str, request: AssembleRequest) -> Dict[str, Any]:
    directory = _case_dir(case_id)
    state = _read_state(case_id)
    if not state.get("analysis"):
        raise HTTPException(status_code=409, detail="Analyse the case before assembling an implant")
    fracture = (state.get("analysis") or {}).get("fracture") or {}
    if state.get("kind") != "volume" and not fracture.get("detected", False):
        raise HTTPException(
            status_code=409,
            detail="No fracture detected; fixation assembly is unavailable for this case.",
        )

    bone_meshes = _load_meshes(directory)
    if not bone_meshes:
        raise HTTPException(status_code=409, detail="No reconstruction meshes stored for this case")

    assembler = Assembler()
    try:
        meshes, metadata = assembler.assemble(bone_meshes, state.get("frame", {}), request.implant_id)
    except AssemblyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    filename = f"assembly_{case_id}_{request.implant_id}.glb"
    try:
        glb_path = exporter.export_scene(meshes, filename=filename, root_name=f"case_{case_id}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GLB export failed: {exc}") from exc

    url = f"/outputs/{Path(glb_path).name}"
    state.setdefault("assemblies", {})[request.implant_id] = url
    _write_state(directory, state)

    selected = next(
        (item for item in (state.get("recommendations") or {}).get("options", []) if item["id"] == request.implant_id),
        None,
    )
    return {
        "case_id": case_id,
        "glb_url": url,
        "device": {
            "id": request.implant_id,
            "name": metadata["device_name"],
            "insertion": metadata["insertion"],
        },
        "animation": {
            "bone_nodes": metadata["bone_nodes"],
            "implant_nodes": metadata["implant_nodes"],
            "fracture_z": metadata.get("fracture_z"),
            "separation_axis": [0, 0, 1],
            "insertion_axis": "lateral" if metadata["insertion"] == "lateral" else "axial",
        },
        "narrative": {
            "audience": request.audience,
            "text": (selected or {}).get("patient_text" if request.audience == "patient" else "surgeon_text"),
        },
        "aspirational_note": "Generic educational implant geometry, not a commercial device.",
        "disclaimer": DISCLAIMER,
    }


@app.delete("/api/cases/{case_id}")
def delete_case(case_id: str) -> Dict[str, Any]:
    directory = _case_dir(case_id)
    shutil.rmtree(directory, ignore_errors=True)
    return {"deleted": True, "case_id": case_id}
