# Progress & Handoff

**Last updated:** 2026-10-26
**Branch:** `main` (guards pushed as `a88adc0`; v3 install, calibration, and docs are local/uncommitted)
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
| Region-labeled model | Not started (needs a licensed hip/pelvis box dataset; see `ml/README.md`) |

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

## 2b. 3D animation upgrade (local, 2026-10-26, uncommitted)

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

## 2c. Weak-signal (borderline) UI (local, 2026-10-26, uncommitted)

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

Committed and pushed:

- `main` at `a88adc0` (model v1 + guards).
- `backend/models/femur_fracture.onnx` is intentionally tracked.

Working tree, **not** committed: the v3 ONNX replaces the tracked v1 binary (38.1 MB),
`backend/models/femur_fracture.calibration.json` (v3, untracked), the UI confidence changes, and
these docs. The v1 model and its calibration are backed up in `.kaggle_staging/v1-output/`
(gitignored) for an immediate revert.

Other local, **not** committed: `.qoder/`, `assembly_d69d8ec686ea_intramedullary_nail.glb`,
`count.js`, `list_raw.ps1`, extra `samples/` images, `yolo11n.pt`, `runs/`, the session `.mp4`.
Ignored directories: `backend/cases/`, `backend/uploads/`, `backend/outputs/*.glb`,
`ml/data/images|labels`, `ml/runs/`, `.kaggle_staging/`.

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

Browser: open `http://127.0.0.1:5173`, accept the disclaimer, upload `samples/1.jpg`, wait for
Fracture Morphology. Expected: amber weak-signal banner (raw 15%, ≈50% calibrated), no implant
cards, no Surgical Simulation section. Then repeat with `samples/2.jpg`: Fracture Probability 63%
(score 21%), implant cards, and assemble produces the reduction animation.

Model sanity checks:

```bash
python backend/tests/onnx_detector_check.py --model backend/models/femur_fracture.onnx
python backend/tests/eval_onnx_test_split.py --split test
python backend/tests/sample_baseline.py   # the 9 user uploads; deployed v3 baseline 5/9
python backend/tests/e2e_check.py   # full pipeline smoke test
```

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
- Region labels are geometric (box position against the silhouette), not learned. A hip case can
  therefore be labeled `shaft`.
- Old case results do not retroactively gain the safeguards; re-upload/re-analyze.
- `Local ONNX detector` is loaded once per process; restart uvicorn after changing the model or
  `config.py`.

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

## 9. Next steps, in order

1. **~~Get licensed hip/femur boxes~~ DONE (2026-10-28)** — Roboflow
   `proximal-femur-fracture` v27 downloaded and converted to `ml/data/region/`. Remaining:
   clinical review of `ml/data/region/review_manifest.csv`, then GPU fine-tune via
   `ml/notebooks/region_yolo_finetune_gpu.ipynb` (attach the dataset + a checkpoint on Kaggle).
2. **Refine the calibration tail.** v3 isotonic maps raw ≥0.402 to 1.0 from a sparse val block;
   audit the support (`.kaggle_staging/calibration_report.py`) and, if needed, cap the tail at 0.95
   or switch to Platt before calling the probability final.
3. **Re-test the 9 user samples** after any model change; track detects and confidences.
   *v3 baseline 5/9 (calibrated 63–67%); v1 was 1/9, v2 2/9.*
4. **Track external specificity.** v3 flags 19% of OrthoFrac non-fracture femur films at 0.15;
   re-check this when licensed boxes allow a region-aware model, and keep the calibrated-probability
   display rather than a bare "detected" claim.
