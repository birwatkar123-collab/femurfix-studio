# Progress & Handoff

**Last updated:** 2026-10-29 (two-model zone classifier shipped and committed — see §8d)
**Branch:** `main`. History is **unpushed** past `bc5d66c` (the v3 deploy); the newest commits carry
the two-model design and its documentation — `git log --oneline bc5d66c..HEAD` for the exact list.
**Purpose:** durable state for whoever picks this up next. Working context that only existed in a
chat session is recorded here.

---

## 1. Where the project stands

FemurFix Studio runs end to end: X-ray upload → fracture detection → region classification →
estimated 3D reconstruction (GLB) → ranked fixation constructs → assembly + reduction animation.
The local detector is a **research prototype** and its sensitivity is the main open problem.

| Area | State |
| --- | --- |
| v3 detector (YOLO11s @960, hip-weighted) | **Deployed 2026-10-26** with hash-locked isotonic calibration (section 3) |
| v1 detector (YOLO11n @640) | Replaced; recoverable from git history / `.kaggle_staging/v1-output/` |
| v2 detector (YOLO11s @960) | Trained and evaluated; not installed (v3 supersedes it) |
| Detector score calibration | Isotonic on FracAtlas val, hash-locked to the ONNX; test Brier 0.101 (section 3) |
| No-fracture safety UX | Implemented, verified, pushed (`a88adc0`) |
| Region-labeled model (v4) | **Not a drop-in replacement** (see §8c). Ships as the secondary **zone classifier** at `backend/models/femur_region.onnx`; v3 remains the fracture detector |

### Deployed model facts (v3 since 2026-10-26)

- Installed at `backend/models/femur_fracture.onnx`, tracked in git: 38,138,284 bytes, SHA-256
  `6fad2537fdc21c80cd4c6a37fd96a38d2070250d3158be579870a61b39fe26e0`.
- ONNX I/O: input `[1,3,960,960]`, output `[1,5,18900]`, one class `fracture`.
- YOLO11s fine-tuned from v2 with **train-only** hip oversampling (6×) and leg oversampling (2×);
  best epoch 12 of 24 (patience 12), val mAP50 0.404 / mAP50-95 0.172.
- Test @0.15 (eval script): image P 0.812 / R 0.639; box P 0.417 / R 0.373.
- CPU latency ~0.89 s/image (v1 ~0.25 s, v2 ~1.37 s).
- Calibration `backend/models/femur_fracture.calibration.json` (16 isotonic points, hash-locked to
  the ONNX above).
- Revert to v1: `git checkout -- backend/models/femur_fracture.onnx`, then copy
  `.kaggle_staging/v1-output/femur_fracture.calibration.json` over the model JSON and restart
  uvicorn.

### Historical v1 model facts

- Was installed at `backend/models/femur_fracture.onnx` from 2026-10-25 to 2026-10-26:
  size 10,663,365 bytes, SHA-256
  `7f8c3b72f02640fbfa69a7a46dcd5e059f1107ea3d72417e8d63ec0fe956f24f`.
- ONNX I/O: input `[1,3,640,640]`, output `[1,5,8400]`, one class `fracture`.
- Held-out test box metrics: P 0.766 / R 0.294 / mAP50 0.338 / mAP50-95 0.119.
- App threshold `LOCAL_ONNX_CONF=0.15`: test **image-level P 0.842 / R 0.525** (33 of 61 positive
  images detected); validation image-level R 0.622.

### Why detection feels poor (diagnosis, 2026-10-25)

- Misses are under-confidence, not wrong localization: missed positives cluster at conf 0.01–0.05,
  and precision stays high while recall collapses.
- Per-region image-level recall on the test positives: **hip 0/2, leg 44%, hand 71%**.
- Root causes:
  1. FracAtlas is mixed anatomy with few femur/hip positives (717 fractured images total; hip flag
     covers 63 fractured images; 574 positive train images).
  2. Small targets: median fracture box is 41 px at 640 input; 31% of boxes are <32 px.
  3. YOLO11n capacity + 50 epochs @640 is marginal for subtle cortical lines.
  4. Domain shift: users upload web/stock films quite different from FracAtlas.
- Fixes tried and **rejected**:
  - Lowering threshold to 0.05: image-level recall rises, but only 1 of 12 extra detections landed
    on a fracture.
  - TTA / tiling: +4 to +7 of 61 test positives recalled, but 2–3× false positives.
- This motivated v2: larger model, higher input resolution, flip/rotation augmentation.

### User-sample behavior (v3, 2026-10-26)

v3 detects **5/9** samples at gate 0.15 — `2.jpg` 0.206, `6d91…` 0.286, `images (2).jpg` 0.179, and
both `xray-image-show-close-fracture…webp` 0.341 — versus 1/9 for v1 and 2/9 for v2. `1.jpg` lands
at 0.148, just under the gate, and now renders the weak-signal banner with a ≈50% calibrated
probability. The two proximal films (`images.jpg`, `images (1).jpg`) and the forearm film `5.jpg`
(human label may be forearm; the app is femur-only) are still missed. Calibrated probabilities for
the five detections are 63–67%, not 90%.

---

## 2. No-fracture safeguards (shipped)

Commit `a88adc0` — `fix: withhold fixation options and block assembly when no fracture is detected`.- `backend/app/main.py`
  - Recommendations are only generated when `fracture.detected` is true (or the case is a volume).
    Otherwise the API returns `options: []`, `withheld: true`, and an explanatory note.
  - `POST /api/cases/{id}/assemble` returns **409** for x-ray cases with no detected fracture.
- `frontend/src/components/SimView.jsx`
  - Header says "no fracture detected"; confidence shows `—`; angle/displacement/angulation rows
    are hidden; amber warning says a negative prediction is not evidence of an intact femur.
  - Implant cards are replaced with a withheld notice; the Surgical Simulation section is hidden.
- Deliberately uses `fractureInfo.detected !== false` so older case payloads without the field
  behave as before.
- Verified with the live app (FastAPI + Vite) via API calls and headless Chrome:
  `1.jpg` → detected, 2 implants, assembly 200; `2.jpg` → withheld, 0 implants, assembly 409.

---

## 2b. 3D animation upgrade (2026-10-26, committed in `2199d89`)

- `backend/app/services/animation_baker.py` (new): injects a `surgical_sequence` glTF clip into
  assembled GLBs with pygltflib — femur_distal reduces (same numbers as the SimView slider), then
  the nail inserts axially and lateral parts stagger in along -x. Called from `assemble_case`
  (`main.py`) and `e2e_check.py`; failures fail soft. Verified via `e2e_check.py` and a live
  API assemble (6 channels baked).
- `frontend/src/components/SimView.jsx`: plays the baked clip via `AnimationMixer` with the
  reduction slider scrubbing clip time; for GLBs without a baked clip it stages implants itself
  (retracted -> placed, nail leads). Scene polish: `RoomEnvironment` PBR reflections, PCFSoft
  shadow maps fitted to the model bounds, and the autoplay loop now uses a GSAP eased ping-pong
  (gsap was already a dependency). Camera is set to an AP-style view (bone long axis +z is
  screen-up, viewed perpendicular) — the default end-on view made the reconstruction unreadable.
  Materials are clamped in the loader traverse (metal<=0.6, roughness>=0.3; bone fully matte) so
  implants stay readable even when the env map is weak. `npm run build` passes. Verified in
  headless Chrome (playwright, channel="chrome"): screenshots show displaced/retracted -> nail
  insertion -> reduced/instrumented states.
- `backend/requirements.txt`: added `pygltflib>=1.16` (already present in env).

## 2c. Weak-signal (borderline) UI (2026-10-26, shipped with `bc5d66c`/`2199d89`)

Motivation: a user upload scored 0.102 raw against the 0.15 gate and saw a flat "no fracture
detected". The detector now returns `raw_top` / `raw_scores` (pre-gate scores, `ml_detector.py`);
`xray_analyzer._no_fracture_from_detection(detection)` attaches `weak_signal = {max_confidence,
candidates, gate_conf}` when `raw_top >= LOCAL_ONNX_WEAK_CONF` (new config, default 0.08,
override with env `LOCAL_ONNX_WEAK_CONF`). `SimView.jsx` shows a distinct amber "Weak fracture
signal" notice (counts, strength, threshold, and the not-a-negative caveat), the header reads
"weak fracture signal below threshold", and detector confidence renders as "10% (weak)".
Recommendations stay withheld — semantics unchanged, only the messaging is more truthful.
Verified on the user's case (source.jpg: max 0.102, 1 candidate) via the live UI in headless
Chrome and `onnx_detector_check.py` / `sample_baseline.py` (baseline still 1/9). Two tutorial
bugs hit on the way: passing the `session.run` list (not array) into `_postprocess` produced a
mysterious `(1,1,5,8400)` shape, and a stale `__pycache__` extended the debugging hunt.

## 3. v2/v3 evaluation and the deployed calibration

### v2 artifacts

- Kaggle kernel completed: `birwatkar/femurfix-fracatlas-yolo11s-960`, 80 epochs in 1.39 h on T4.
- Downloaded to `.kaggle_staging/v2-output/` (gitignored): `best_v2.pt` 19,223,962 bytes,
  `femur_fracture_v2.onnx` 38,138,282 bytes, `run_summary_v2.json`, training curves.
- Verified genuine: `nc=1`, `names={0: 'fracture'}`, static 960 input, weights differ from the
  pretrained `yolo11s.pt`.
- Recipe: YOLO11s @960, batch 8, AdamW(auto) lr 0.002, 80 epochs, fliplr 0.5, degrees 10,
  mosaic 1.0 with close_mosaic 12. It **underfit relative to v1** (final val box/cls loss
  1.317/2.120 vs v1 1.253/2.356; best val mAP50 0.357 vs v1 0.449).

### Held-out benchmark (ONNX, `eval_onnx_test_split.py`, gate 0.15)

| Metric | v1 | v2 | **v3 (deployed)** |
| --- | --- | --- | --- |
| FracAtlas test image P / R | 0.842 / 0.525 | 0.829 / 0.475 | **0.812 / 0.639** |
| FracAtlas test box P / R / F1 | 0.535 / 0.343 / 0.418 | 0.561 / 0.343 / 0.426 | 0.417 / 0.373 / 0.394 |
| FracAtlas val image P / R | 0.893 / 0.610 | 0.951 / 0.476 | — |
| FracAtlas val box P / R / F1 | 0.603 / 0.484 / 0.537 | 0.597 / 0.407 / 0.484 | — (training val mAP50 0.404) |
| Test hip / leg / hand recall | 0/2, 11/26, 22/33 | 0/2, 10/26, 18/33 | **1/2, 14/26, 24/33** |
| Negatives flagged, production path | 6/183 (3.3%) | — | 12/183 (6.6%) |
| User samples ≥0.15 | 1/9 | 2/9 | **5/9** |
| CPU latency / image | ~0.25 s | ~1.37 s | ~0.89 s |

v3 is an image-level win on the labeled test set: recall +11.4 pts over v1 and image F1 0.715 vs
v1's 0.647, while box F1 slips slightly (0.394 vs 0.418). It finds more true boxes (box recall
0.343 → 0.373) but emits more false ones (box precision 0.535 → 0.417), which is the same
sensitivity/specificity shift seen on external films. Because the app's region and fracture-plane
placement come from the top box, box precision still matters; a false positive therefore produces
both a spurious detection and a mislabeled region.

On the production path, v1 detected only `1.jpg` (0.170). v3 detects `2.jpg` (0.206), `6d91…`
(0.286), `images (2).jpg` (0.179) and the two webp films (0.341), while `1.jpg` falls to 0.148 —
just under the 0.15 gate, so it now shows the weak-signal banner instead of a detection. Both
proximal hip samples still score 0.000.

### External femur benchmark — OrthoFrac-XR (CC BY 4.0, 1,493 images)

OrthoFrac-XR has **image-level labels only, no bounding boxes** (`Bone_metadata.csv` has bone type,
gap size, visibility). The femur subset is 262 images: 52 proximal fractures, 12 distal, 55
post-fracture, 143 non-fracture. Detection rate at 0.15:

| Femur subset | v1 | v2 | **v3** |
| --- | --- | --- | --- |
| Proximal fracture (n=52) | 21% | 38% | **50%** |
| Distal fracture (n=12) | 50% | **75%** | 67% |
| Post-fracture (n=55) | 38% | 44% | **51%** |
| Non-fracture false positives (n=143) | 13% | **10%** | 19% |

v3 leads every fracture subset except distal, where 67% vs 75% is a single image out of 12. The cost
is specificity: non-fracture false positives rise 10% → 19% at 0.15. At gate 0.20 v3 gives
proximal 40% / distal 50% / post 40% / non-fracture 13%, but FracAtlas test image F1 drops to 0.604
(below v1's 0.647), so 0.15 stays. The hip weighting bought proximal recall with false positives;
that trade is the main caveat of the deployed model.

v2 is the reason v3 fine-tunes **from `best_v2.pt`** rather than discarding it. The earlier
v1+v2 max-ensemble added recall but doubled latency and raised non-fracture false positives to 18%;
on FracAtlas test it did not beat v1 (image F1 0.655 vs 0.660). Not integrated.

### Rejected: ROI/bone-crop second pass

`backend/tests/eval_roi_crop.py` on the full test split (margin 0.15, segmentation usable on
243/244 images, mean crop 73% of frame): at gate 0.15 image recall 0.525 → 0.410 and box recall
0.343 → 0.030. Box mapping was verified on a sample image; the crop removes context and hurts.
Likewise, ad-hoc hip-region crops on `samples/images.jpg` and `images (1).jpg` return 0.000 for
both models. Do not add a crop pass.

### v3 run (complete, deployed 2026-10-26)

- Kaggle kernel: `birwatkar/femurfix-fracatlas-yolo11s-960-hip-weighted-v3` (the live slug is
  derived from the title; the metadata id is `...-hipweighted`). Private, T4.
- Source: `birwatkar/femurfix-fracatlas-yolo` (images), `birwatkar/fracatlas-metadata` (new,
  `dataset.csv`), and kernel source `birwatkar/femurfix-fracatlas-yolo11s-960` for `best_v2.pt`.
- Data: training split only — fractured hip images duplicated 5 extra times (6× total) and
  fractured leg images 1 extra time (2×), using the FracAtlas `hip`/`leg` flags. Val/test unchanged.
- Recipe: base `best_v2.pt`, 45 epochs @960, batch 8, AdamW lr 0.0007 with cosine decay,
  fliplr 0.5, **no rotation**, mosaic 1.0 / close_mosaic 8, patience 12.
- Outcome: early-stopped at epoch 24; best epoch 12. Val mAP50 0.404 / mAP50-95 0.172 (v2: 0.357 /
  0.138). The user recovered only the checkpoint (`best_v3.pt`, 19,216,922 bytes) — the "zip" was a
  raw PyTorch archive — so the deployed ONNX was exported locally at 960:
  `.kaggle_staging/v3-output/femur_fracture_v3.onnx`, 38,138,284 bytes.
- A 640 export was also tested: FracAtlas test image F1 0.641 (vs 0.715 at 960), so 960 was kept.
  640 did score `1.jpg` 0.335, confirming the model is strongly inference-size sensitive.
- Adoption: cleared the primary gate (beat v1 on FracAtlas test, +11.4 pts image recall, every
  anatomical group improved, samples 1/9 → 5/9). It did **not** meet the strict secondary gate
  (v2's distal 75% and 10% non-fracture FP), so that is recorded as an accepted trade-off, not a
  clean pass. Gate stays `LOCAL_ONNX_CONF=0.15`.

```bash
python backend/tests/eval_onnx_test_split.py --model .kaggle_staging/v3-output/femur_fracture_v3.onnx --split test
python backend/tests/sample_baseline.py --model .kaggle_staging/v3-output/femur_fracture_v3.onnx
```

### Score calibration (deployed)

- Detector scores are not probabilities. Isotonic regression maps the best raw score to
  P(image contains fracture), fitted on FracAtlas val (328 images, 82 positives) with production
  preprocessing. `ml_detector._load_calibration` refuses a JSON whose `model_sha256` does not match.
- v1 (22 points): test Brier 0.158 → 0.119.
- v3 (16 points, deployed): val Brier 0.139 → 0.093; test 0.142 → 0.101; val logloss 0.772 → 0.313;
  test logloss 0.730 → 0.335. Test reliability top bin: 0.793 predicted / 0.792 observed.
- **Tail audit (2026-10-26):** uncapped isotonic mapped raw ≥0.402 to 1.0 (val 30/30 positives), but
  a test negative scores 0.623, so 100% was unjustified. The deployed tail is capped at **0.95**
  (test tail ≥0.402: 20/21 positives; ≥0.50: 17/18). The cap left test Brier unchanged (0.101) and
  improved test logloss 0.375 → 0.335.
- Production mapping (step function): 0.15 → 0.50, 0.20 → 0.63, 0.30–0.34 → 0.67, ≥0.40 → 0.95.
- The backend exposes `confidence` (calibrated), `detector_confidence` (raw), `confidence_basis`;
  weak signals include `calibrated_probability`.

### Dataset assessment (2026-10-26)

| Source | Size / labels | License | Usable for training? |
| --- | --- | --- | --- |
| FracAtlas | 2,868 prepared images, 67 test boxes | CC BY 4.0 | Yes — current training set |
| OrthoFrac-XR (figshare) | 1,493 images, image-level only, femur: 64 fractures + 143 negatives | CC BY 4.0 | No boxes; good external benchmark |
| Hips & pelvis (`mohamedahmed12232`) | 12,218 YOLO images, 2,552 unique train, noisy hip/pelvis boxes | Private (data.yaml) | No — license + label noise |
| HBFMID (`orvile`) | 10 fracture-type classes | Private (data.yaml) | No — license, not anatomy-specific |
| FracAtlas-4K | 335 MB copy | CC BY 4.0 | No — byte-identical images, no resolution gain |
| KAUH-HipFx | 768 hip films, good subtype labels | CC BY-NC-ND | No — 404 download, no derivatives |
| Roboflow `thesisyolo-v8/proximal-femur-fracture` | 756 images, segmentation, 7 hip classes | CC BY 4.0 | **Best next source — needs a Roboflow API key** |
| BoneFract (Mendeley) | 47,931 films, anatomy classes | CC BY 4.0 | Image-level only; pretraining, not boxes |

The blocker is not model architecture but **licensed proximal/distal femur films with bounding
boxes**. FracAtlas has only 51 fractured hip training images, which is why oversampling is a thin
lever.

### The 90% confidence target

The recommender shows `min(0.55 + confidence * 0.4, 0.95)` with the calibrated probability, so a
displayed 90% requires calibrated ≈0.875 — reached only from raw ≈0.40 upward, where the calibrated
value is 0.95 and the test tail is 20/21 positives. Typical hip films and 4 of the 5 detected user
samples sit at raw 0.18–0.34 → 50–67% → recommender 75–82%. The FracAtlas test top bin (0.79
predicted / 0.79 observed) is the realistic ceiling.

So 90% is now technically reachable only for the strongest detector hits, and even there it is
indicative, not a guarantee. The lever that would move the typical case is not more isotonic fitting
but licensed proximal/distal femur **boxes**: FracAtlas has 51 fractured hip training images, which
is why the hip-weighting run buys recall at the cost of specificity.

---

## 4. Repo / local state

Committed (pushed history ends at `bc5d66c`; `2199d89`/`4fe7ef7`/`0a66631`/`2c8d338` are local-only):

- `backend/models/femur_fracture.onnx` is **v3** and intentionally tracked: 38,138,284 bytes,
  `6fad2537…fe26e0`, with `femur_fracture.calibration.json` alongside it. Verified byte-identical to
  HEAD after the §8d revert.
- `backend/models/femur_region.onnx` is **v4**, tracked as the **second** model (38,144,548 bytes) —
  allowed through `.gitignore:79` so a fresh clone runs the two-model path with no setup.
- `2c8d338` carries the §8d two-model change (`ml_detector.py`, `xray_analyzer.py`, `config.py`,
  `main.py`, `SimView.jsx`, `backend/tests/region_classifier_check.py` (new), `.gitignore`,
  `ml/scripts/train_region_kaggle.py`); the documentation commits after it touch only these two
  markdown files.

Still **not** committed: `backend/outputs/e2e_assembly.glb` (regenerated by `e2e_check.py`, test
artefact churn) and `ml/notebooks/fracatlas_yolo11s_960_gpu.ipynb` (editor re-save only — escaped
unicode and a `kernelspec` block, no content change). Both are deliberately left out; either can be
`git checkout`'d.

Other local, **not** committed: `.qoder/`, `assembly_d69d8ec686ea_intramedullary_nail.glb`,
`count.js`, `list_raw.ps1`, `data.yaml` and `femurfix-5-class-region-training.log` in the repo root
(stray Kaggle pull-downs, safe to delete), extra `samples/` images including `samples/best_v3.zip`
(a raw PyTorch checkpoint, not a zip), `yolo11n.pt`, `runs/`, `weights/`, the session `.mp4`.
Ignored directories: `backend/cases/`, `backend/uploads/`, `backend/outputs/*.glb`,
`ml/data/images|labels`, `ml/data/region/`, `ml/runs/`, `.kaggle_staging/`.

v4 artifacts are preserved in `.kaggle_staging/v4-output/` (gitignored): `femur_region.onnx`,
`best_region.pt`, `region_summary.json` (see the §8c metrics warning), `data.yaml`.

Servers used during this session (Windows, Git Bash):

```bash
# backend (no --reload; restart after Python edits)
cd backend
"C:/Users/mansi/AppData/Local/Programs/Python/Python311/python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# frontend (Vite proxies /api to 8000)
cd frontend
npm run dev
```

---

## 5. Verification recipe

API smoke test (expect weak signal + withheld options on `1.jpg`; detected on `2.jpg`):

```bash
CID=$(curl -s -X POST http://127.0.0.1:8000/api/cases -F "file=@samples/1.jpg" | python -c "import sys,json;print(json.load(sys.stdin)['case_id'])")
curl -s -X POST "http://127.0.0.1:8000/api/cases/$CID/analyze" -H "Content-Type: application/json" -d '{}'
# assemble should return 200 for a detected fracture, 409 when none is detected
curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://127.0.0.1:8000/api/cases/$CID/assemble" \
  -H "Content-Type: application/json" -d '{"implant_id":"intramedullary_nail","audience":"surgeon"}'
```

Browser: open `http://localhost:5173` (Vite binds IPv6 only on this machine — `127.0.0.1:5173` gets
no response), accept the disclaimer, upload `samples/1.jpg`, wait for
Fracture Morphology. Expected: amber weak-signal banner (raw 15%, ≈50% calibrated), no implant
cards, no Surgical Simulation section, and a *view context* line naming the zone the anatomical-zone
model reads the film as. Then repeat with `samples/2.jpg`: Fracture Probability 63% (score 21%),
implant cards, a region row whose sub-label says which model assigned it, and assemble produces the
reduction animation.

Model sanity checks:

```bash
python backend/tests/onnx_detector_check.py --model backend/models/femur_fracture.onnx   # 1 class, [1,5,18900]
python backend/tests/onnx_detector_check.py --model backend/models/femur_region.onnx     # 5 classes, [1,9,18900]
python backend/tests/eval_onnx_test_split.py --split test   # v3 benchmark: image P 0.812 / R 0.639
python backend/tests/sample_baseline.py   # the 9 user uploads; deployed v3 baseline 5/9, cal= non-null
python backend/tests/region_classifier_check.py   # per-sample zone + region_source from the real analyzer
python backend/tests/e2e_check.py   # full pipeline smoke test
```

`region_classifier_check.py` is the guard for the two-model design: it prints the zone the classifier
picked and where the region label came from (`zone_model` / `geometric` / `user_override`). Expect
`2.jpg` detected again (it was 0.000 under the v4-as-replacement swap) and its zone not forced to
`proximal`.

The previous session's scratch scripts and screenshots lived in
`C:\Users\mansi\AppData\Local\Temp\femurfix-e2e` and `C:\tmp\femurfix-e2e\shots`; they are
temporary and may be gone. The reproducible evaluator is `backend/tests/eval_onnx_test_split.py`.

---

## 6. Known issues / caveats

- v3 sensitivity is better than v1 (test image recall 0.525 → 0.639, all anatomical groups up) but
  hip/proximal films remain weak, and external non-fracture false positives rose to 19% at 0.15.
  Box precision also fell (0.535 → 0.417), so a false positive can mislabel the region. The UI
  presents a calibrated probability and the weak-signal band instead of inventing certainty.
- The calibration is fitted on the val split that early stopping also used; test reliability
  (Brier 0.101, top-bin 0.79 predicted vs 0.79 observed) is the honest number. The high tail is
  capped at 0.95 after the audit (uncapped 100% was refuted by a test negative scoring 0.623), so
  probabilities above ~0.8 are indicative, not definitive.
- No-fracture cases still show the estimated 3D preview volume. That is intentional (it is a generic
  reconstruction, not a diagnosis).
- Region labels are **not learned for the fracture itself**. The zone classifier only names where a
  already-detected fracture sits; when it has no confident box, the region falls back to geometry
  (box position against the silhouette), so a hip case can still be labeled `shaft`. The UI's
  region sub-label and `region_source` field say which path was used — trust those, not the label.
- The zone classifier (`femur_region.onnx`) has **no calibration of its own**, and must never be
  presented as a fracture probability. Its scores only gate which zone wins; the v3 detector plus
  `femur_fracture.calibration.json` owns every probability in the UI.
- Zone coverage is hip-only (`neck` / `intertrochanteric` / `subtrochanteric` — the Roboflow source
  has no `femoral_shaft` or `distal_femur` classes), so shaft and distal films resolve geometrically
  by design. Roadmap A2 (MAISI→DRR synthetic hips) is the route to real coverage.
- Two ONNX models are now loaded once per process. Restart uvicorn after changing either model,
  `config.py`, or the detector/analyzer services.
- Old case results do not retroactively gain the safeguards; re-upload/re-analyze.

---

## 7. Credentials / security

- Kaggle API token is stored **only** at `~/.kaggle/access_token` on this machine. It is not in the
  repo; `.kaggle_staging/` is gitignored.
- The token was exposed in chat during setup. **Rotate it** at Kaggle → Settings → API if that has
  not been done, then update `~/.kaggle/access_token`.
- Before any future push, run `git status` and confirm no `kaggle.json`, token, `.env`, or case data
  is staged. No secrets are currently tracked.
- HF Space credentials belong in env vars (`HF_SPACE_ID`, `HF_TOKEN`), never in the repo.

---

## 8b. Phase 1 execution (2026-10-28, MedTech Days roadmap — see `ROADMAP.md`)

- **Grounded UI:** new `GET /api/cases/{id}/source-image` (FileResponse) and a Detection Evidence
  panel in `SimView.jsx` (radiograph + detector-box overlay + calibrated-probability caption).
  The region override (proximal/shaft/distal re-analyze) is now wired into the SimView header card;
  earlier only the app shell exposed it. TestClient verified 200/404; `npm run build` passes.
- **Physics-style animation bake:** `animation_baker.py` rewritten — reduction approaches along the
  measured displacement vector with ease, overshoots the seat by -8%, rebounds, settles; non-planar
  yaw/pitch/roll fragment rotation; implants (per-part keyframe grids, staggered) seat with a ~5%
  impaction recoil. Verified numerically on `samples/2.jpg` via `backend/tests/inspect_bake.py`
  (6 channels; nail recoil 203 mm → 9.06 mm → 0) and e2e. Times are per-part, so clip duration is
  the longest track (~3.4 s).
- **Region dataset A1 cleared:** Roboflow `thesisyolo-v8/proximal-femur-fracture` **v27**
  (756 images, CC BY 4.0, no-aug 1024) downloaded via `ml/scripts/download_roboflow.py`
  (`RF_API_KEY` env var; key read from `~/.roboflow_key.txt`, never committed; export URL shape is
  `/{ws}/{proj}/{version}/{format}`). The export is segmentation polygons;
  `prepare_region_dataset.py` now converts polygons → boxes. Output `ml/data/region/`
  (gitignored): 607/68/81 splits, 371 positives (neck 289, intertrochanteric 467, subtrochanteric
  136 boxes) + 385 explicit negatives; all splits pass `validate_labels.py`. No
  `femoral_shaft`/`distal_femur` in this proximal-only source. Clinical review of
  `review_manifest.csv` is the next manual step before GPU training on
  `region_yolo_finetune_gpu.ipynb`.

---

## 8c. v4 region training on Kaggle (2026-10-29)

**Datasets (private, user `birwatkar`):**

| Dataset | Contents | Mount on kernel |
| --- | --- | --- |
| `birwatkar/femurfix-region-dataset` | 756 images + labels + `data.yaml` + `mapping.json` | `/kaggle/input/datasets/birwatkar/femurfix-region-dataset/region_dataset/` |
| `birwatkar/femurfix-v3-checkpoint` | `best_v3.pt` (18.3 MB) | `/kaggle/input/datasets/birwatkar/femurfix-v3-checkpoint/` |

Kernel: **`birwatkar/femurfix-5-class-region-training`** (script kernel, T4 GPU, internet on).
Entry script is `ml/scripts/train_region_kaggle.py`; the push folder is `D:\k_pull\`
(`kernel-metadata.json` + renamed `femurfix-5-class-region-training.py`).

**CLI push works** — `kaggle kernels push -p D:\k_pull` succeeds when the folder's
`kernel-metadata.json` came from `kaggle kernels init` and has `is_private`/`enable_gpu`/... as
strings. The earlier `409 Conflict` came from hand-written metadata that didn't match the existing
kernel record. `kaggle kernels run` does **not** exist in CLI 2.2.4; use `push` + `status` + `output`.

**Six failure modes already fixed (do not regress):**

1. `ModuleNotFoundError: ultralytics` — script bootstraps `pip install ultralytics>=8.3,<9`
   (`enable_internet: "true"` required).
2. Kaggle mounts under `/kaggle/input/datasets/<user>/<slug>/` (three levels) — resolve paths with
   `rglob`, never hardcode.
3. Uploaded `data.yaml` contains the authoring machine's absolute path
   (`D:/New folder (3)/...`); the kernel rewrites it to `/kaggle/working/data.yaml` with
   `path` = mounted dataset root.
4. The dataset folder contains a *stray duplicate* `data.yaml` at the mount root; pick the copy
   whose sibling `images/val` exists.
5. `data.yaml` also gets uploaded **inside** the auto-extracted `region_dataset/` subfolder — that
   is the copy that has the images.
6. `metrics.box.maps` is a **numpy array** in Ultralytics 8.4, not a dict (`AttributeError` killed the
   first completed run *after* training). Use `np.ravel(np.asarray(metrics.box.maps))` + `metrics.names`.

**Result of run v6 (training + test eval completed before the crash):** YOLO11s @960, 60 epochs,
early-stopped ~22 min total, test split (81 images / 102 boxes):

| Class | Imgs | Boxes | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| all | 81 | 102 | 0.437 | 0.575 | **0.480** | **0.248** |
| femoral_neck | 12 | 24 | 0.473 | 0.667 | 0.636 | 0.387 |
| intertrochanteric | 22 | 55 | 0.332 | 0.655 | 0.424 | 0.183 |
| subtrochanteric | 9 | 23 | 0.507 | 0.403 | 0.379 | 0.174 |

This beats v3 (test mAP50 0.404) and is the first model that names the fracture region directly.

**Run v7 (`2026-10-29 ~03:30` local) is the clean re-run** with the metrics fix + ONNX export +
`best_region.pt` copy. **Completed Successfully** on GPU T4×2; artifacts pulled to `D:\region_out\`
and preserved in `.kaggle_staging/v4-output/` (gitignored):

- `femur_region.onnx` — 38,144,548 bytes, SHA-256
  `acbb73c031e28b7147df1a511ebb0478dbe6d16de4bbc32cc38126fc7cf6f9a8`, input `[1,3,960,960]`,
  output `[1,9,18900]`, names `femoral_neck / intertrochanteric / subtrochanteric / femoral_shaft /
  distal_femur`
- `best_region.pt` (19,248,154 bytes), `region_summary.json`, `data.yaml`, run log

**v7 metrics correction.** `region_summary.json` is wrong and must not be quoted: the script read
`metrics.box.maps`, which in Ultralytics 8.4 is the per-class **mAP@50-95** vector, and labelled it
`per_class_ap50`. `ml/scripts/train_region_kaggle.py` now reads `box.maps50` into
`per_class.<cls>.AP50`. True numbers from the run log:

| Split | Class | Imgs | Boxes | P | R | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val (68 img) | all | 68 | 81 | 0.583 | 0.525 | **0.546** | 0.276 |
| val | femoral_neck | 12 | 24 | 0.637 | 0.750 | 0.750 | 0.441 |
| val | intertrochanteric | 18 | 45 | 0.447 | 0.576 | 0.497 | 0.239 |
| val | subtrochanteric | 5 | 12 | 0.666 | 0.250 | 0.391 | 0.147 |
| **test** (81 img) | all | 81 | 102 | 0.437 | 0.575 | **0.480** | 0.248 |
| test | femoral_neck | 12 | 24 | 0.473 | 0.667 | **0.636** | 0.387 |
| test | intertrochanteric | 22 | 55 | 0.332 | 0.655 | **0.424** | 0.183 |
| test | subtrochanteric | 9 | 23 | 0.507 | 0.403 | **0.379** | 0.174 |

`femoral_shaft` / `distal_femur` are **unlabelled in this source** (proximal-only Roboflow export); the
0.248 recorded for them in the JSON is the overall mAP50-95 echoed into empty slots, not a measurement.
Training was deterministic enough that v7 reproduced v6's test table exactly.

---

## 8d. v4 is NOT a drop-in replacement — two-model design (2026-10-29)

**What was tried.** v4 was copied over `backend/models/femur_fracture.onnx`, the v3 calibration was
deleted, and `xray_analyzer.py` was changed to derive `region` from the model class.

**Measured against the deployed benchmark (gate 0.15, `eval_onnx_test_split.py`):**

| Metric | v3 (deployed) | v4 as replacement |
| --- | --- | --- |
| FracAtlas test image P / R | 0.812 / 0.639 | **0.256 / 0.180** |
| FracAtlas test box P / R / F1 | 0.417 / 0.373 / 0.394 | **0.020 / 0.015 / 0.017** (1 of 67 boxes) |
| Best test F1 threshold | 0.15 → image F1 0.715 | 0.30 → F1 **0.025** |
| User samples | 5/9 | 3/9 |

**Why — target convention, not a bad training run.** FracAtlas boxes delimit the **fracture line**
(median 41 px at 640 input). Roboflow `proximal-femur-fracture` boxes delimit **anatomical zones**, so
v4 emits large zone boxes that cannot match fracture-line GT. The consequence is visible on the user
samples: v4 finds the hip films v3 missed (`images.jpg` → `intertrochanteric` 0.300 where v3 scored
0.000) but loses every distal/shaft case (`2.jpg` → 0.000 where v3 scored 0.206). Reverting was the
only correct move; the 0.480 figure is real but describes a different task.

**Shipped design.** v3 stays authoritative for *whether* and *where*; v4 labels the anatomical zone.

- `backend/models/femur_fracture.onnx` — reverted to v3 (`6fad2537…`, calibration live again).
- `backend/models/femur_region.onnx` — v4, tracked in git via a `.gitignore` allowlist entry.
- `config.py`: `REGION_ONNX_MODEL` / `USE_REGION_ONNX` / `REGION_ONNX_CONF` (default **0.10**).
  `LOCAL_ONNX_CONF` stays 0.15 and the 0.95 calibration cap is untouched.
- `ml_detector.py`: `OnnxFractureDetector(bind_calibration=False)` (it already handled N classes —
  `_read_names`, `channels = 4 + len(names)`); `get_region_classifier()` singleton returns `None`
  when the file is missing, refuses a binary model, so the app degrades to the old behaviour;
  `classify_region(image, fracture_box)` picks the zone overlapping the fracture box (containment of
  the box centre, else IoU — zone boxes are far larger than fracture boxes so plain IoU under-selects);
  `classify_view(image)` returns the top zone over the whole frame; `detect()` now also returns `boxes`.
- `xray_analyzer.py`: region precedence is detector class → zone model above gate → geometric ratio.
  `_map_class_to_region` returns `""` for the binary `fracture` class instead of forcing `proximal`.
  The fracture payload gained `region_source` (`model` | `zone_model` | `geometric` | `unresolved`),
  `region_score`, `anchor_ratio`, and `view_hint` on negatives. `main.py` sets `user_override`.
- The zone model never influences `detected`, recommendations, or the assemble 409 guard.

**Verified 2026-10-29** (`backend/tests/region_classifier_check.py`, both models loaded):
| Sample | detected | calibrated | region | via |
| --- | --- | --- | --- | --- |
| `2.jpg` | ✓ | 0.632 | shaft | geometric (zone abstains — distal film) |
| `6d91…big_gallery.jpg` | ✓ | 0.632 | **proximal** | **zone_model** (`subtrochanteric`) |
| `images (2).jpg` | ✓ | 0.632 | shaft | geometric |
| both `xray-image-show-close-fracture…webp` | ✓ | 0.667 | shaft | geometric |
| `1.jpg` | ✗ (weak) | — | — | view_hint **intertrochanteric @0.42** |
| `images.jpg` | ✗ | — | — | view_hint **intertrochanteric @0.28** |

Baseline restored to **5/9 with calibration live**. `e2e_check.py` passes; `npm run build` passes.
UI verified in headless Chrome (playwright, `channel="chrome"`) against the live servers:

- zone case → `Proximal (AO/OTA 31)` + `zone: subtrochanteric · 23% · via anatomical-zone model`
- geometric case → `Shaft (AO/OTA 32)` + `via silhouette geometry` (provenance was previously hidden
  whenever no zone class existed — the sub-label is now gated on `region_source`, not on the class)
- negative case → `Not determined by model` + `film reads as intertrochanteric (42%) — view context
  only`, and the weak-signal banner carries the same sentence **once** (hoisted into one `viewContext`
  fragment; the duplicate `weak_signal.view_hint` string was removed from the payload so `view_hint`
  is single-sourced as `{class_name, score}`)
- **Pattern/region contradiction fixed.** The heuristic writes its own region word into `pattern`, so a
  zone-model override produced "region: proximal" next to "oblique **shaft** fracture". `_fracture_from_detection`
  now rewrites that word to the final region; verified → `Model-detected oblique proximal fracture`.
- `capitalize` on the parent does not override `normal-case` on the sub-label (rendered lowercase "via").

UI: the Region row now shows `zone: <class> · <score>% · via <source>`, the Region-review tooltip
names the actual source, and a `view_hint` appears on negatives as view context only.

**Open.** Four of five detections still resolve geometrically — the zone model abstains on distal/shaft
films because it has no such class. Genuine `femoral_shaft`/`distal_femur` coverage is roadmap **A2**
(MAISI→DRR synthetic hips), not a re-run of this dataset. The zone model has no calibration of its own;
its scores are displayed as raw detector scores and must never be presented as probabilities.

---

## 9. Next steps, in order

1. **~~Get licensed hip/femur boxes~~ DONE (2026-10-28)** — Roboflow
   `proximal-femur-fracture` v27 downloaded and converted to `ml/data/region/`. Remaining:
   clinical review of `ml/data/region/review_manifest.csv`, then GPU fine-tune via
   `ml/notebooks/region_yolo_finetune_gpu.ipynb` (attach the dataset + a checkpoint on Kaggle).
   **Update 2026-10-29:** training ran through the CLI kernel instead (see §8c). Test mAP50 0.480.
   The "collect v7 artifacts, swap the ONNX into the backend, recalibrate, extend `xray_analyzer.py`
   for the 5-class output" plan that used to sit here is **rejected** — v4 as a replacement is a
   severe regression (§8d). v4 ships instead as the secondary zone classifier, and it is not
   recalibrated because it is not a fracture detector. Next real work on this axis is A2 (synthetic
   hip data) so the zone model gains `femoral_shaft` / `distal_femur` coverage.
2. **~~Refine the calibration tail~~ DONE** — v3 isotonic mapped raw ≥0.402 to 1.0 from a sparse val
   block; the audit (`.kaggle_staging/calibration_report.py`) refuted it (a test negative scored
   0.623) and the tail is now capped at 0.95. Platt remains an option if the calibrator is rebuilt.
3. **Re-test the 9 user samples** after any model change; track detects and confidences.
   *v3 baseline 5/9 (calibrated 63–67%); v1 was 1/9, v2 2/9.*
4. **Track external specificity.** v3 flags 19% of OrthoFrac non-fracture femur films at 0.15;
   re-check this when the zone model has broader coverage, and keep the calibrated-probability
   display rather than a bare "detected" claim.
5. **Push the local-only commits.** `2c8d338` and its three predecessors are unpushed; nothing is
   left uncommitted except the two artefacts named in §4. Re-read §7 before pushing.
6. **Finish A3.** The zone classifier names hip zones on detected fractures today; it does not yet
   feed implant selection or the surgical plan, and it has no shaft/distal classes. A2 is the
   prerequisite for the latter.
