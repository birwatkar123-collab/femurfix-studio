# ML Training for FemurFix Studio

**Status (2026-10-26):** The deployed detector is **v3** — YOLO11s @960 fine-tuned from v2 with
train-only hip/leg oversampling, installed at `backend/models/femur_fracture.onnx` with a
hash-locked isotonic calibration (`backend/models/femur_fracture.calibration.json`). It remains a
research prototype: held-out image recall is 0.639, hip/proximal films are still weak, and the
region label is geometric rather than learned. The original YOLO11n baseline, the v2 run (not
installed), and v3 are documented below; recipes live in `ml/notebooks/` and handoff state in
[`PROGRESS.md`](../PROGRESS.md).

## Trained baseline (Kaggle)

| Split | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| Val (best epoch) | 0.715 | 0.451 | 0.461 | 0.194 |
| Test (best.pt / ONNX) | 0.766 | 0.294 | 0.338 | 0.119 |

- Training: 50 epochs, `imgsz=640`, `batch=16`, YOLO11n, ~15 min on a T4.
- The test ONNX export reproduces the PyTorch box metrics (P 0.769, R 0.299 at `conf=0.25`).
- The backend uses `conf=0.15` (best image-level F1 on validation). At that threshold the test
  split gives image-level P 0.842 / R 0.525 and box-level P 0.535 / R 0.343.
- Most errors are missed detections, not wrong locations: at `conf=0.15` precision stays high
  while recall falls away, especially on low-contrast and small fracture lines.
- Artifacts: Kaggle kernel `birwatkar/femurfix-fracatlas-yolo11n-training`; local outputs in
  `ml/runs/fracatlas_yolo11n/` and `ml/runs/fracatlas_yolo11n_test/` (`run_summary.json`,
  `results.csv`, plots, `weights/best.pt`, `weights/best.onnx`).

```bash
# score the installed ONNX model on a split (box + image metrics, threshold sweep)
python backend/tests/eval_onnx_test_split.py --split test

# validate the installed model's I/O and letterbox mapping
python backend/tests/onnx_detector_check.py --model backend/models/femur_fracture.onnx
```

## Model evaluation report (v1, 2026-10-26)

Full threshold sweep of the installed `backend/models/femur_fracture.onnx` (YOLO11n @640) from
`eval_onnx_test_split.py`. Box metrics are at IoU 0.5; "image" metrics count an image positive if
any detection clears the threshold.

**Test split** — 244 images, 61 positives, 67 boxes:

| conf | box P | box R | box F1 | image P | image R | TP / FP / FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.257 | 0.388 | 0.310 | 0.631 | 0.672 | 26 / 75 / 41 |
| 0.10 | 0.407 | 0.358 | 0.381 | 0.776 | 0.623 | 24 / 35 / 43 |
| **0.15 (deployed)** | 0.535 | 0.343 | 0.418 | 0.842 | 0.525 | 23 / 20 / 44 |
| 0.20 | 0.636 | 0.313 | 0.420 | 0.812 | 0.426 | 21 / 12 / 46 |
| 0.25 | 0.769 | 0.299 | 0.430 | 0.923 | 0.393 | 20 / 6 / 47 |
| 0.30 | 0.760 | 0.284 | 0.413 | 0.920 | 0.377 | 19 / 6 / 48 |
| 0.40 | 0.895 | 0.254 | 0.395 | 0.947 | 0.295 | 17 / 2 / 50 |
| 0.50 | 0.938 | 0.224 | 0.361 | 0.938 | 0.246 | 15 / 1 / 52 |
| 0.60 | 0.857 | 0.090 | 0.162 | 0.857 | 0.098 | 6 / 1 / 61 |

**Val split** — 328 images, 82 positives, 91 boxes:

| conf | box P | box R | box F1 | image P | image R | TP / FP / FN |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.324 | 0.527 | 0.402 | 0.644 | 0.683 | 48 / 100 / 43 |
| 0.10 | 0.459 | 0.495 | 0.476 | 0.732 | 0.634 | 45 / 53 / 46 |
| **0.15 (deployed)** | 0.603 | 0.484 | 0.537 | 0.893 | 0.610 | 44 / 29 / 47 |
| 0.20 | 0.712 | 0.462 | 0.560 | 0.917 | 0.537 | 42 / 17 / 49 |
| 0.25 | 0.745 | 0.418 | 0.535 | 0.932 | 0.500 | 38 / 13 / 53 |
| 0.30 | 0.837 | 0.396 | 0.537 | 0.975 | 0.476 | 36 / 7 / 55 |
| 0.40 | 0.967 | 0.319 | 0.479 | 1.000 | 0.341 | 29 / 1 / 62 |
| 0.50 | 1.000 | 0.253 | 0.404 | 1.000 | 0.268 | 23 / 0 / 68 |
| 0.60 | 1.000 | 0.110 | 0.198 | 1.000 | 0.110 | 10 / 0 / 81 |

Best box-F1: test 0.430 @ conf 0.25; val 0.560 @ conf 0.20. The deployed 0.15 trades ~4 points of
box F1 for a meaningful image-level recall gain, which favours safety (fewer undetected fractures)
at higher false-positive cost. Same conclusion as the 2026-10-25 diagnosis: misses are
under-confidence, not wrong localization.

### User-sample baseline (v1, `backend/tests/sample_baseline.py`)

At the deployed conf 0.15, the local detector finds a fracture in **1 of 9** user uploads:

| Sample | Detected | Conf | Detections |
| --- | --- | --- | --- |
| 1.jpg | yes | 0.170 | 1 |
| 2.jpg, 5.jpg, 6d91c93a…jpg, images.jpg, images (1).jpg, images (2).jpg, xray-…178.webp, xray-…178 (1).webp | no | — | 0 |

This is the v1 baseline; v2 scored 2/9 and the deployed v3 scores 5/9. Re-run with
`python backend/tests/sample_baseline.py` (or `--model <path>` to score a candidate).

## v2 evaluation — YOLO11s @960 (2026-10-26)

Run completed in 1.39 h on a T4 (`birwatkar/femurfix-fracatlas-yolo11s-960`). Artifacts live in
`.kaggle_staging/v2-output/` (gitignored): `best_v2.pt` (19.22 MB), `femur_fracture_v2.onnx`
(38.14 MB), `run_summary_v2.json`, curves. Checkpoint verified (one class, 960 input, weights
differ from pretrained).

**Training:** 80 epochs, batch 8, AdamW(auto) lr 0.002, fliplr 0.5, degrees 10, mosaic 1.0,
close_mosaic 12. It underfit vs v1: final val box/cls loss 1.317/2.120 (v1 1.253/2.356), best val
mAP50 0.357 (v1 0.449).

**FracAtlas held-out (ONNX, `eval_onnx_test_split.py`, gate 0.15):**

| Metric | v1 | v2 |
| --- | --- | --- |
| Test image P / R | 0.842 / 0.525 | 0.829 / 0.475 |
| Test box P / R / F1 | 0.535 / 0.343 / 0.418 | 0.561 / 0.343 / 0.426 |
| Val image P / R | 0.893 / 0.610 | 0.951 / 0.476 |
| Val box P / R / F1 | 0.603 / 0.484 / 0.537 | 0.597 / 0.407 / 0.484 |
| Test hip / leg / hand recall | 0/2, 42%, 67% | 0/2, 38%, 55% |

**External femur benchmark — OrthoFrac-XR** (CC BY 4.0; image-level labels only; femur subset
n=262). Detection rate at 0.15:

| Femur subset | v1 | v2 |
| --- | --- | --- |
| Proximal fracture (n=52) | 21% | **38%** |
| Distal fracture (n=12) | 50% | **75%** |
| Post-fracture (n=55) | 38% | 44% |
| Non-fracture false positives (n=143) | 13% | **10%** |

**User samples:** v2 finds 2/9 at 0.15 vs v1's 1/9 (`1.jpg` 0.170 → 0.272; `6d91…` 0.086 →
0.239). Both proximal hip samples remain 0.000. CPU latency 1.37 s vs 0.25 s (5.5×).

**Verdict:** not a drop-in replacement. It loses ~5 points of FracAtlas image recall but gains on
external high-resolution proximal/distal femurs, which is the direction that matters for real
uploads. v2 is retained as the base checkpoint for the hip-weighted v3 fine-tune, not deployed.

**Rejected preprocessing:** `backend/tests/eval_roi_crop.py` (bone-mask crop, margin 0.15) on the
full test split gives image recall 0.525 → 0.410 and box recall 0.343 → 0.030 at gate 0.15. Hip
region crops on the user samples also return 0.000. No crop second pass.

## v3 — hip/leg-weighted fine-tune (deployed 2026-10-26)

`ml/notebooks/fracatlas_yolo11s_960_hipweighted_gpu.ipynb` starts from `best_v2.pt` and duplicates
fractured **hip** images 6× and fractured **leg** images 2× in the training split only, using the
FracAtlas `dataset.csv` flags (uploaded as `birwatkar/fracatlas-metadata`). Recipe: 45 epochs @960,
batch 8, AdamW lr 0.0007 with cosine decay, fliplr 0.5, no rotation, close_mosaic 8, patience 12.
Kernel: `birwatkar/femurfix-fracatlas-yolo11s-960-hip-weighted-v3` (private, T4).

**Training outcome:** early-stopped at epoch 24; best epoch 12. Val mAP50 0.404 / mAP50-95 0.172
(v2: 0.357 / 0.138). Only the checkpoint (`best_v3.pt`, 19.22 MB) was recovered, so the ONNX was
exported locally at 960 (`femur_fracture_v3.onnx`, 38.14 MB). A 640 export scored test image F1
0.641 vs 0.715 at 960, so 960 was kept; the model is strongly inference-size sensitive.

**FracAtlas test @0.15:** image P 0.812 / R 0.639 (F1 0.715; v1 0.647), box P 0.417 / R 0.373 /
F1 0.394. Production-path per-anatomy recall: hip 1/2, leg 14/26, hand 24/33; negatives flagged
12/183. Box precision fell while box recall rose — the model finds more fractures but also emits
more false boxes.

**External femur benchmark (OrthoFrac-XR, n=262), detection at 0.15:**

| Femur subset | v1 | v2 | v3 |
| --- | --- | --- | --- |
| Proximal fracture (n=52) | 21% | 38% | **50%** |
| Distal fracture (n=12) | 50% | **75%** | 67% |
| Post-fracture (n=55) | 38% | 44% | **51%** |
| Non-fracture false positives (n=143) | 13% | **10%** | 19% |

At gate 0.20 v3 gives proximal 40 / distal 50 / post 40 / non-fracture 13, but FracAtlas test image
F1 falls to 0.604, so the app keeps 0.15.

**User samples:** 5/9 detected (v1 1/9, v2 2/9) — `2.jpg` 0.206, `6d91…` 0.286, `images (2).jpg`
0.179, both webp films 0.341; `1.jpg` 0.148 (weak-signal band). CPU latency 0.89 s.

**Calibration:** isotonic on FracAtlas val (328 images / 82 positives), hash-locked to the ONNX.
Test Brier 0.142 → 0.101, logloss 0.730 → 0.335 (capped), top bin 0.793 predicted / 0.792 observed.
Raw 0.15 → 50%, 0.20 → 63%, 0.30 → 67%, ≥0.40 → 95%. The tail is capped at 95%: uncapped isotonic
said 100% from a 30/30 val block, but a test negative scores 0.623; the test tail is 20/21 positive.

**Verdict:** deployed. It clears the primary gate (beat v1 on FracAtlas test, improves every
anatomical group and the samples) but not the strict secondary gate (v2's distal 75% and 10%
non-fracture FP). The hip weighting buys proximal recall with specificity; that trade, plus the
sparse calibration tail, are the known caveats.

```bash
python backend/tests/eval_onnx_test_split.py --model .kaggle_staging/v3-output/femur_fracture_v3.onnx --split test
python backend/tests/sample_baseline.py --model .kaggle_staging/v3-output/femur_fracture_v3.onnx
```

## Dataset

- Source: FracAtlas (4,083 images; 717 fractured with 922 boxes; 3,366 negatives).
- Prepared: **2,296 train / 328 val / 244 test** (574/82/61 positives + 3x negatives per split).
- Labels: one class, `fracture`; negative images have empty label files.
- All 2,868 images pass Ultralytics's strict image checker (0 corrupt after repair).
- `ml/data.yaml` resolves paths relative to its own directory, so it works from any CWD.

Rebuild and validate:

```bash
python ml/scripts/convert_fracatlas.py
for split in train val test; do
  python ml/scripts/validate_labels.py \
    --images-dir ml/data/images/$split \
    --labels-dir ml/data/labels/$split
done
```

## Install

```bash
python -m pip install -r ml/requirements.txt
```

`albumentations>=1.4.21` is pinned because Kaggle ships 1.3.1, which lacks the `quality_range`
argument that Ultralytics 8.4 passes to `ImageCompression`.

## GPU training (recommended)

Open `ml/notebooks/fracatlas_yolo11n_gpu.ipynb` on Kaggle/Colab. Attach the prepared `ml/data`
images and labels as a Kaggle Dataset (do not upload `ml/data/raw`), or attach raw FracAtlas and
let the notebook run the conversion. Alternatively, run the trainer directly on a GPU machine:

```bash
python ml/train/train_yolo.py --device 0 --epochs 50 --imgsz 640 --batch 16
```

Outputs land in `ml/runs/fracatlas_yolo11n/`:

- `weights/best.pt` and `weights/best.onnx`
- `results.csv`, `results.png`, `confusion_matrix.png`
- `run_summary.json` with held-out test precision/recall/mAP

Remove `--cache` from the notebook if RAM is tight; lower `--batch` to 8 on a CUDA OOM.

## Local CPU smoke test

```bash
python ml/train/train_yolo.py --device cpu --epochs 1 --imgsz 320 --batch 8 \
  --workers 0 --fraction 0.02 --name smoke --no-export-onnx --no-eval-test
```

This only verifies the pipeline; it is not a useful model.

## Scope

FracAtlas is mixed anatomy and has a single fracture class. It is a **pretraining** dataset.
The femoral-neck-vs-shaft classifier the app needs requires hip/pelvis AP images annotated with
`femoral_neck`, `intertrochanteric`, `subtrochanteric`, `femoral_shaft`, and `distal_femur`.

### The 90% confidence target

The recommender reports `min(0.55 + confidence * 0.4, 0.95)`, where `confidence` is now the
calibrated probability. A displayed 90% therefore needs calibrated ≈0.875, which the mapping
reaches only from raw ≈0.40 upward (test tail 20/21 positive), capped at 0.95 — indicative, not a
guarantee. Typical hip films and four of the five detected user samples sit at raw 0.18–0.34 →
calibrated 50–67% → recommender 75–82%. The FracAtlas test top bin (0.79 predicted / 0.79 observed)
is the realistic reliability ceiling. Reaching a defensible 90% needs licensed proximal/distal femur
boxes and a region-focused model; rescaling the score to hit 90% would be misrepresentation, not a
model improvement.

## Integration (complete)

1. `femur_fracture.onnx` (v3, YOLO11s @960) is installed at `backend/models/femur_fracture.onnx`.
2. `backend/models/femur_fracture.calibration.json` holds the isotonic score→probability mapping,
   hash-locked to that ONNX.
3. `backend/app/services/ml_detector.py` runs it with `onnxruntime` (CPU, letterbox, NMS) and
   exposes raw and calibrated scores.
4. `xray_analyzer.py` prefers `onnx_local`, then the HF Space, then the heuristic.

The ONNX binary is tracked in git through an explicit `.gitignore` exception. On a fresh checkout
it is already present; replace it only when installing a newly trained model.

Environment overrides: `FRACTURE_ONNX_MODEL`, `USE_LOCAL_ONNX=0`, `LOCAL_ONNX_CONF`,
`LOCAL_ONNX_IOU`.

The ONNX export is detector-only (no region classes). Region is assigned from the bbox position
relative to the silhouette, so hip-region cases still rely on geometry. That is the gap the
region-labeled track below addresses.

## Region-labeled dataset track (next)

Target classes: `femoral_neck`, `intertrochanteric`, `subtrochanteric`, `femoral_shaft`,
`distal_femur`. A YOLO dataset with those classes would replace both the binary detector and the
geometric region assignment.

**Prepared tooling (2026-10-26, smoke-tested):**

- `ml/scripts/download_roboflow.py` — pulls the dataset below once an API key exists (env
  `RF_API_KEY`; `--format yolov8/yolov11`, pins or takes the latest version, unzips to
  `ml/data/raw/proximal-femur-fracture/` and prints the conversion command).

- `ml/scripts/prepare_region_dataset.py` — converts any licensed region-labelled YOLO export into
  the 5-class schema: class re-mapping (neck/lesser/greater trochanter per the table below,
  `normal`/`dislocation` dropped or kept as negatives via `--keep-negatives`), patient-grouped
  splits, `data.yaml`, `mapping.json`, and `review_manifest.csv` for clinical review. Unmapped
  source classes are counted and reported instead of silently dropped.
  ```bash
  python ml/scripts/prepare_region_dataset.py \
      --source ml/data/raw/proximal-femur-fracture \
      --classes "neck,intertrochanteric,subtrochanteric,greater_trochanter,lesser_trochanter,dislocation,normal" \
      --dst ml/data/region
  ```
  After inspection, run `ml/scripts/validate_labels.py` on the output and review
  `review_manifest.csv` clinically before training.
- `ml/notebooks/region_yolo_finetune_gpu.ipynb` — Kaggle skeleton (kernelspec metadata included)
  for 960 px fine-tuning, per-class AP on the held-out set, and ONNX export. Attach the prepared
  dataset and a `best.pt` checkpoint to use it.

Candidate sources (checked 2026-10-25):

- **Roboflow Universe `thesisyolo-v8/proximal-femur-fracture`** — 756 images, instance
  segmentation, CC BY 4.0. Classes: neck, intertrochanteric, subtrochanteric, greater trochanter,
  lesser trochanter, dislocation, normal. Covers the proximal subtypes; export needs a Roboflow
  API key. Best immediate source for the neck/intertrochanteric/shaft confusion.
- **`González et al. 2025, Insights into Imaging`** — 1,454 femurs from 986 hip X-rays, three
  region classes and segmentation masks; available on request from the authors.
- **KAUH-HipFx (Kaggle, CC BY-NC-ND 4.0)** — 768 pelvic/hip X-rays: 300 normal, 234
  intertrochanteric, 234 femoral neck, from King Abdullah University Hospital. Labels fit the
  hip classification task exactly, but the license forbids derivatives/commercial use and the
  dataset download endpoint returned 404 on 2026-10-25. Do not ship weights trained on it
  without resolving the license.
- **OrthoFrac-XR (figshare, CC BY 4.0)** — verified 2026-10-26: 1,493 images, **no bounding
  boxes** (only `Bone_metadata.csv` with bone type, gap size, visibility), femur subset 262 images
  (52 proximal fractures, 12 distal, 55 post-fracture, 143 non-fracture). Mixed anatomy including
  elbow/tibia; `fracture_type` is a location class, not a femur region. Used as the external
  benchmark above; cannot train the detector directly.
- **BoneFract (Mendeley, CC BY 4.0)** — 47,931 X-rays with `Pelvis` / `Thigh` / `Lower Leg`
  anatomy classes and fracture status. Good for anatomy pretraining, not for hip subtypes.
- **FracAtlas metadata** — `dataset.csv` has `hip` (338 images, 63 fractured) and `leg` flags.
  Useful as a weak seed for hip vs shaft, but no subtype labels and `leg` includes tibia/fibula.

Checked and rejected (2026-10-26):

- **`mohamedahmed12232/hips-and-pelvis-fraction-detection`** (Kaggle, 733 MB, 12,218 images,
  2,552 unique train) — useful volume but the hip project's `data.yaml` says `Private`, boxes are
  noisy (pelvis/iliac crest), and labels are not anatomy-reviewed. Do not train on it.
- **HBFMID** (`orvile/human-bone-fractures-image-dataset-hbfmid`, 38 MB) — Kaggle metadata says
  CC BY 4.0 but the Roboflow `data.yaml` says `Private`; fracture-type classes, not anatomy.
- **FracAtlas 4K** (`birwatkar/...` copy, 335 MB) — byte-identical images to the original
  (same sizes, dimensions and Laplacian variance); no resolution gain.
- **KAUH-HipFx** — download endpoint 404 and CC BY-NC-ND forbids derivatives.

Suggested pipeline once a source is licensed/accessible:

1. Export to YOLO format; map `neck` -> `femoral_neck`, `intertrochanteric`/trochanteric ->
   `intertrochanteric`, `subtrochanteric` -> `subtrochanteric`; add `femoral_shaft` and
   `distal_femur` from FracAtlas cases after manual review.
2. Build a review manifest and have an orthopaedic surgeon verify every box and class; exclude
   `normal`/`dislocation` images from the detector or keep them as explicit negatives.
3. Split by patient, not image, and hold out a hip-only test set. Report per-class AP and a
   confusion matrix; the binary FracAtlas metrics above do not transfer.
4. Fine-tune from the current checkpoint at `imgsz=960`; add TTA only at evaluation time.

The source data acquisition and clinical review need owner access/credentials; the code path and
schema can be prepared without them.

