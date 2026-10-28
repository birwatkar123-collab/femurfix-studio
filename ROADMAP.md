# FemurFix Studio — Future Roadmap

**Created:** 2026-10-28
**Sources:** NVIDIA MedTech Days 2026 notes in `C:\Users\mansi\Pictures\Screenshots\*.md`
(Open-H stack, NV-Generate/MAISI, NV-Reason-CT, NV-Segment-CTMR, MONAI-Physio, Isaac/Newton,
AutoMedBench, agents + digital twins, Holoscan). Cross-referenced against `PROGRESS.md`.

> Current baseline: v3 detector deployed (test image F1 0.715, 5/9 user samples, hip recall weak),
> isotonic calibration capped at 0.95, keyframed GLB surgical animation, guarded no-fracture UX.

## Phase 1 execution status (2026-10-28)

- [x] **A6 grounded UI** — `GET /api/cases/{id}/source-image` serves the upload; SimView shows a
  **Detection Evidence** panel (radiograph + amber detector-box overlay + calibrated probability
  caption) and a **Region review** override row (proximal/shaft/distal re-analyze, previously
  wired only in the app shell). Frontend build passes; endpoint smoke-tested (200/404).
- [x] **B3 physics-baked animation** — `animation_baker.py` rewritten: measured-vector approach
  with overshoot past the seat (-8%), damped settle, non-planar yaw/pitch/roll path, staggered
  implant entries with a ~5% impaction recoil, per-part keyframe grids. Verified on `samples/2.jpg`
  (6 channels) and e2e (`backend/tests/inspect_bake.py`).
- [-] **A1 region dataset** — ✅ **cleared 2026-10-28**: Roboflow v27 (756 imgs) downloaded with
  `ml/scripts/download_roboflow.py` (key kept at `~/.roboflow_key.txt`, outside the repo; delete or
  rotate it when done), converted to boxes, validated: `ml/data/region/` = 607/68/81 splits,
  371 positives (femoral_neck 289 / intertrochanteric 467 / subtrochanteric 136) + 385 negatives.
  Remaining: clinical review + GPU training (`region_yolo_finetune_gpu.ipynb`). Downloader fix:
  export URL is `/{ws}/{proj}/{version}/{format}` (not `/dataset/...`).

---

## Goals

1. **Classification accuracy & honest confidence** — target ~90% *trustworthy* image-level
   confidence via better data + two-reader agreement (never display uncapped 100%).
2. **Visualization quality** — physically plausible reduction animation and 3D renders readable
   by patients and surgeons.

## Reality check

- Every event talk converged on: **the data gap is the central bottleneck**; the app's own
  bottleneck is 51 fractured hip training images in FracAtlas. No architecture swap fixes this.
- Attainable: image-level 80–90% with calibrated probabilities; the 0.95 cap stays (a raw 0.62
  scored negative in the tail audit).
- Progress confidence UX should follow the "grounded CoT" principle: box overlay + calibrated
  probability + reader agreement, no bare "detected" claims.

---

## Track A — Detection / classification / confidence

| # | Action | Source tech | Integrate? | Effort |
| --- | --- | --- | --- | --- |
| A1 | Region-aware v4 detector on Roboflow `proximal-femur-fracture` (756 imgs, boxes, CC BY 4.0) | Existing plan (§ nums in PROGRESS.md) | Replace geometric region labeling | Low |
| A2 | Synthetic hip X-ray engine: MAISI/NV-Generate hip CT volumes + paired masks → DRR projection → fracture-line injection → YOLO augmentation for hip class | NV-Generate/MAISI (HF) | Augment training data only; real val/test untouched | Med-High |
| A3 | Learned region classes: `femoral_neck / intertrochanteric / subtrochanteric / shaft / distal` per ML_TRAINING_PLAN.md | Own plan | Replace silhouette-ratio classification | Medium |
| A4 | Two-reader confidence gate: local YOLO + open reasoning VLM second read; display agreement | NV-Reason-CT pattern; MedGemma/Merlin baselines | Add optional cloud mode; local ONNX stays primary | High |
| A5 | SSL pretraining on unlabeled X-rays (SegResNetDS + cuboid masked-volume reconstruction) | NSW telestroke talk | Improves recipe | Medium |
| A6 | Grounded explanation UI: box overlay + calibrated probability + reader agreement | groundable-CoT principle | Modify SimView | Low |

Expected: hip recall 0–50% → 60–75%; image F1 0.715 → ~0.80; displayed confidence becomes
"two independent reads agree, calibrated 0.86"-style statements.

## Track B — 3D reconstruction / animation

| # | Action | Source tech | Integrate? | Effort |
| --- | --- | --- | --- | --- |
| B1 | CT-path: segment uploads with NV-Segment-CTMR (345+ classes) instead of HU thresholding | NV-Segment-CTMR (⚠️ Non-Commercial license) | Replace `>200 HU` thresholding | Medium |
| B2 | X-ray-path patient prior: MAISI synthetic hip CT from silhouette measurements → mesh from volume | MAISI | Upgrade SDF-estimate path | High |
| B3 | Physics-based reduction animation: rigid fragments + fracture-plane contact (XPBD), rollout baked into existing glTF clip | Newton/Warp or Blender physics | Replace keyframed `animation_baker.py` path | Medium |
| B4 | Biomechanics overlay: Von Mises stress baked into GLB vertex colors | MONAI-Physio / PhysicsNeMo (Apache 2.0) | Add to implant comparison | High |
| B5 | GPU-accelerated re-bake with Warp differentiable physics (event claims 72×–1000× sim speedups) | Warp | Replace CPU baking | Medium |
| B6 | AR export: USDZ alongside GLB for phone-based patient viewing | OpenUSD ecosystem | Add export path; Three.js stays primary | Low-Med |

## Track C — Workflow

- C1: AutoMedBench-style rubric (Plan/Setup/Validate/Inference/Submit) on every retrain so box
  precision regressions (v1 0.535 → v3 0.417) cannot slip through again.
- C2: MONAI agent skills + Brev credits for GPU runs (MAISI, physics) without owning GPUs.
- C3: Agent + tools backend architecture: detector / segmenter / reconstructor / animator each
  inspectable and versioned (ORDIRS/TwinOR pattern, arXiv 2505.03798).

## Do NOT adopt

- GROOT-H / Cosmos3-H-Embodied-Nano / Cosmos-H-Dreams (surgical robotics VLAs — no robot),
  Versius-500 (video+telemetry), NV-Reason-PathWSI / DeepLIIF / ImPartial (pathology),
  Holoscan 5 / IGX Thor / FPGA / Jetson (certified device deployment), Open-H-RF (ultrasound),
  BioNeMo (drug discovery), Foundation Pose + SAM2 robotic automation (OR context).

## Sequencing

```
Phase 1 (wks 1–3):  A1 region v4 + A6 grounded UI + B3 physics-baked animation
Phase 2 (wks 3–8):  A2 MAISI→DRR synthetic hips → v5 region training (A3) + A5 SSL
Phase 3 (wks 8+):   B1 NV-Segment-CTMR CT path + B6 USDZ AR export
Phase 4 (later):    A4 two-reader VLM, B4 stress maps, C3 agentic backend
```

## Dependencies / risks

- **Roboflow API key** — Phase 1 blocker (already known; scripts + notebook are ready).
- **One accessible GPU** for MAISI + physics (Kaggle T4 sufficed for v1–v3; Brev is the backup).
- **Licenses:** NV-Segment-CTMR is Non-Commercial — fine for the educational app, blocks
  commercialization. NV-Generate/MAISI license must be verified before training use.
- **Synthetic data validation:** every synthetic-data step must be gated against the real
  held-out splits (synthetic anatomy accuracy is an open problem even in the source talks).
- Keep the standing rule: `LOCAL_ONNX_CONF=0.15`, calibration tail capped 0.95; re-test the 9
  user samples after every model change (v3 baseline: 5/9).

## Key links

- NV-Generate / MAISI: Hugging Face `nvidia/` (see models-and-functions.md), arXiv WACV 2025
- NV-Reason-CT: HF `nvidia/NV-Reason-CT`, arXiv 2609.27511
- NV-Segment-CTMR: HF `nvidia/` (OneWay Non-Commercial license)
- MONAI-Physio: github.com/Project-MONAI/monai-physio (Apache 2.0)
- Newton / Isaac for Healthcare: docs.nvidia.com i4h agentic workflows course
- AutoMedBench: github.com/AutoMedBench/AutoMedBench
- Open-H: www.open-h.org
- Digital-twin representations: arXiv 2505.03798
