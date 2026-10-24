# FemurFix Studio — Product Spec (v0.1)

> Extends `master_prompt_for_opencode.md`. Same repo skeleton, new product flow.

## Mission

An educational, simulation-first web app that takes a **femur radiograph (X-ray)** and walks a
surgeon or patient through:

1. **Upload** an X-ray (PNG/JPG/DICOM) of a femur — CT/MRI series also accepted.
2. **Fracture analysis** — an ML model detects the fracture, localises it and classifies the
   region (proximal / shaft / distal), with confidence.
3. **Estimated 3D reconstruction** — the femur is rebuilt as an interactive animated WebGL model.
4. **Fixation options** — a rule engine + ML ranking recommends suitable generic fixation
   constructs (DHS, cannulated screws, cephalomedullary nail, locking plates, IM nail).
5. **Selection + assembly** — the user picks an option and the implant is animated onto the
   fractured femur; fragments separate, screws insert, before/after toggle.
6. **Disclaimer** — explicit, blocking: *educational simulation only, not medical advice.*

## Decisions (confirmed)

| Area | Decision |
|---|---|
| Primary input | X-ray (2D). CT/MRI supported as a higher-fidelity path. |
| Implant catalog | Built in-house, **parametric** generic devices (no proprietary CAD). |
| ML hosting | **Hugging Face ZeroGPU** Space, called from the backend via `gradio_client`. |
| ML data | Public datasets (FracAtlas et al.) + pretrained vision backbones, fine-tuned. |
| Audience | Surgeons **and** patients (two language registers). |
| Repo root | `D:\New folder (3)` |

## Non-negotiable honesty rules

- A single 2D radiograph cannot recover the patient's true 3D anatomy. The 3D model is an
  **estimated, anatomical-prior reconstruction**, labelled as such in the UI and API.
- Without a calibration marker, pixel measurements are not physical. All measurements carry
  `estimated: true` unless DICOM pixel spacing is available.
- The recommender is **decision support for discussion**, never "the recommended treatment".
- No PHI retention: uploads are stored per-case temporarily and can be purged; the API never
  returns the source image.
- The app is **not** a medical device. Clinical use requires regulatory clearance (SaMD).

## Architecture

```
Browser (React + R3F wizard)
  │
  ├── POST /api/cases                 upload X-ray / DICOM zip / NIfTI
  ├── POST /api/cases/{id}/analyze    fracture detection + reconstruction
  ├── GET  /api/cases/{id}            case state + recommendations
  ├── POST /api/cases/{id}/assemble   selected implant -> animated GLB
  └── GET  /outputs/...               generated GLB assets
  │
FastAPI backend
  ├── image_loader        X-ray (png/jpg/dcm) + CT/MRI volume (dicom/nii)
  ├── xray_analyzer       HF ZeroGPU call, deterministic fallback, confidence
  ├── bone_reconstructor  X-ray: SDF shape prior; CT/MRI: segmentation + marching cubes
  ├── mesh_processor      Marching Cubes, Laplacian smoothing, quadric decimation
  ├── implant_catalog     parametric generic fixation devices
  ├── recommender         fracture class + measurements -> ranked options + rationale
  ├── assembly            bone fragments + implant -> hierarchical animated GLB
  └── gltf_exporter       semantic node names, PBR materials
  │
HF Space (ml/space)  @spaces.GPU
  ├── analyze_xray        fracture bbox + region + confidence (fine-tuned detector)
  └── segment_femur       bone mask (SAM/MedSAM or U-Net)
```

## Fracture regions & AO/OTA mapping (v1)

- **Proximal** — neck / intertrochanteric / subtrochanteric (AO 31-A/B/C)
- **Shaft** — AO 32-A/B/C (simple wedge, comminuted)
- **Distal** — AO 33-A/B/C (extra-articular, partial/full articular)

## Recommendation rules (v1, explainable)

| Region / pattern | Ranked options |
|---|---|
| Undisplaced neck | Cannulated screws → DHS |
| Displaced neck | Cephalomedullary nail → DHS |
| Intertrochanteric | DHS → Cephalomedullary nail → Proximal locking plate |
| Subtrochanteric | Cephalomedullary nail → Proximal locking plate |
| Shaft, simple | Intramedullary nail → Locking plate |
| Shaft, comminuted | Locking plate → Intramedullary nail |
| Distal | Distal femoral locking plate → Retrograde nail |

Each returned option carries `score`, `rationale[]`, `confidence`, and `caveats[]`.

## Animation model

The assembled GLB contains semantic nodes:

- `femur_proximal`, `femur_distal` (split at the fracture plane; single `femur` if none)
- `implant_<device>_<part>` (plate, lag_screw, locking_screw_1..n, nail)

The frontend animates two parameters:

- `separation` (0–100): fragments separate along the fracture axis.
- `insertion` (0–100): implant components travel from an exploded pose to the fixed pose.

Both are lerped each frame from stored base transforms — no accumulating drift.

## Verification plan

1. Synthetic femur X-ray generator (`backend/tests/make_synthetic_xray.py`).
2. End-to-end: upload → analyze → recommendations → assemble → GLB served over HTTP.
3. `npm run build` passes.
4. HF Space runs standalone (`python ml/space/app.py`) and returns a schema-valid response.
