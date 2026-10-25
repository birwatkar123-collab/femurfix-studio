# ML Training for FemurFix Studio

**Status (2026-10-25):** A YOLO11n baseline was trained on Kaggle (T4, 50 epochs) and exported to
ONNX. The model is installed at `backend/models/femur_fracture.onnx` and is the backend's primary
detector. It is a research prototype: held-out recall is low, and the region label is still
geometric rather than learned. FracAtlas is converted with the official splits, 61 truncated JPEGs
were repaired, and a 1-epoch CPU smoke test also completed.

A **v2 run (YOLO11s @960, 80 epochs)** is in flight on Kaggle; the notebook is
`ml/notebooks/fracatlas_yolo11s_960_gpu.ipynb`, kernel
`birwatkar/femurfix-fracatlas-yolo11s-960`. Integration criteria and run state are tracked in
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

## Integration (complete)

1. `best.onnx` is installed at `backend/models/femur_fracture.onnx`.
2. `backend/app/services/ml_detector.py` runs it with `onnxruntime` (CPU, letterbox, NMS).
3. `xray_analyzer.py` prefers `onnx_local`, then the HF Space, then the heuristic.

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
- **OrthoFrac-XR (figshare, CC BY 4.0)** — 1,493 X-rays with proximal/distal/post-fracture/
  non-fracture classes and metadata. Coarse regions only, but useful as extra negatives.
- **BoneFract (Mendeley, CC BY 4.0)** — 47,931 X-rays with `Pelvis` / `Thigh` / `Lower Leg`
  anatomy classes and fracture status. Good for anatomy pretraining, not for hip subtypes.
- **FracAtlas metadata** — `dataset.csv` has `hip` (338 images, 63 fractured) and `leg` flags.
  Useful as a weak seed for hip vs shaft, but no subtype labels and `leg` includes tibia/fibula.

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

