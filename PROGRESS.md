# Progress & Handoff

**Last updated:** 2026-10-25
**Branch:** `main` (guard changes pushed as `a88adc0`; this handoff is the next commit)
**Purpose:** durable state for whoever picks this up next. Working context that only existed in a
chat session is recorded here.

---

## 1. Where the project stands

FemurFix Studio runs end to end: X-ray upload → fracture detection → region classification →
estimated 3D reconstruction (GLB) → ranked fixation constructs → assembly + reduction animation.
The local detector is a **research prototype** and its sensitivity is the main open problem.

| Area | State |
| --- | --- |
| v1 detector (YOLO11n @640) | Integrated, tracked in git, working |
| v2 detector (YOLO11s @960) | **Training on Kaggle, still RUNNING** at time of writing |
| No-fracture safety UX | Implemented, verified, pushed (`a88adc0`) |
| Region-labeled model | Not started (needs a hip/pelvis dataset; see `ml/README.md`) |

### v1 model facts

- Installed at `backend/models/femur_fracture.onnx` (tracked in git; `.gitignore` has an explicit
  exception). Size 10,663,365 bytes, SHA-256
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

### v1 user-sample behavior

The current `samples/` set (user-uploaded) is mostly missed at 0.15 — e.g. `samples/1.jpg` detects
at **17% confidence**; `samples/2.jpg` detects nothing. That is a model limitation, not an
integration bug; labels, preprocessing, letterbox mapping, and ONNX parity were validated.

---

## 2. No-fracture safeguards (shipped)

Commit `a88adc0` — `fix: withhold fixation options and block assembly when no fracture is detected`.

- `backend/app/main.py`
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

## 3. v2 training run (in flight)

- **Kaggle kernel:** `birwatkar/femurfix-fracatlas-yolo11s-960`
  (`https://www.kaggle.com/code/birwatkar/femurfix-fracatlas-yolo11s-960`), private, T4, internet on.
- **Dataset source:** `birwatkar/femurfix-fracatlas-yolo` (prepared FracAtlas YOLO export).
- **Recipe:** `yolo11s.pt`, 80 epochs, `imgsz=960`, `batch=8`, `patience=20`, seed 0,
  deterministic, `cache=True`, `fliplr=0.5`, `degrees=10`, mosaic with `close_mosaic=12`,
  no mixup/copy-paste/erasing. Expect roughly 1.5–3 h on a T4.
- **Notebook kept in-repo:** `ml/notebooks/fracatlas_yolo11s_960_gpu.ipynb` (and
  `..._kernel-metadata.json`). The live copy also exists in `.kaggle_staging/femurfix-kernel-v2/`,
  which is gitignored.
- **Outputs on completion:** `/kaggle/working/femur_fracture_v2.onnx`,
  `best_v2.pt`, `run_summary_v2.json`, `runs/fracatlas_yolo11s_960/` (curves, confusion matrix).

### Integrate v2 only if it beats v1

```bash
# 1. Check completion, then download outputs (do not commit these)
python -m kaggle kernels status birwatkar/femurfix-fracatlas-yolo11s-960
python -m kaggle kernels output birwatkar/femurfix-fracatlas-yolo11s-960 -p .kaggle_staging/v2-output

# 2. Score the candidate on the held-out split (works at any static input size)
python backend/tests/eval_onnx_test_split.py --model .kaggle_staging/v2-output/femur_fracture_v2.onnx --split test

# 3. Compare with v1 numbers in section 1; re-tune LOCAL_ONNX_CONF on validation
#    (evaluate at 0.05/0.10/0.15/0.20/0.25; choose best image-level F1)

# 4. Only if better: replace the tracked model, then restart the backend
cp .kaggle_staging/v2-output/femur_fracture_v2.onnx backend/models/femur_fracture.onnx
# edit backend/app/utils/config.py LOCAL_ONNX_CONF if the calibration changes

# 5. Re-run the sample + browser checks (section 5) and update ml/README.md metrics
```

The detector reads its input size from the ONNX graph (`ml_detector.py`), so a 960 model needs no
code change. If the kernel failed, read its log in the Kaggle UI; likely causes are CUDA OOM
(reduce `batch` to 4–6) or the dataset not being mounted (check `dataset_sources`).

---

## 4. Repo / local state

Committed and pushed:

- `main` at `a88adc0` (model v1 + guards).
- `backend/models/femur_fracture.onnx` is intentionally tracked.

Local, **not** committed: `.qoder/`, `assembly_d69d8ec686ea_intramedullary_nail.glb`, `count.js`,
`list_raw.ps1`, extra `samples/` images, `yolo11n.pt`, `runs/`, the session `.mp4`. Ignored
directories: `backend/cases/`, `backend/uploads/`, `backend/outputs/*.glb`, `ml/data/images|labels`,
`ml/runs/`, `.kaggle_staging/`.

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

API smoke test (expect detected on `1.jpg`, no detection + withheld options on `2.jpg`):

```bash
CID=$(curl -s -X POST http://127.0.0.1:8000/api/cases -F "file=@samples/1.jpg" | python -c "import sys,json;print(json.load(sys.stdin)['case_id'])")
curl -s -X POST "http://127.0.0.1:8000/api/cases/$CID/analyze" -H "Content-Type: application/json" -d '{}'
# assemble should return 200 for a detected fracture, 409 when none is detected
curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://127.0.0.1:8000/api/cases/$CID/assemble" \
  -H "Content-Type: application/json" -d '{"implant_id":"intramedullary_nail","audience":"surgeon"}'
```

Browser: open `http://127.0.0.1:5173`, accept the disclaimer, upload `samples/2.jpg`, wait for
Fracture Morphology. Expected: amber warning, no implant cards, no Surgical Simulation section.
Then repeat with `samples/1.jpg` and assemble; the reduction slider should appear.

Model sanity checks:

```bash
python backend/tests/onnx_detector_check.py --model backend/models/femur_fracture.onnx
python backend/tests/eval_onnx_test_split.py --split test
python backend/tests/e2e_check.py   # full pipeline smoke test
```

The previous session's scratch scripts and screenshots lived in
`C:\Users\mansi\AppData\Local\Temp\femurfix-e2e` and `C:\tmp\femurfix-e2e\shots`; they are
temporary and may be gone. The reproducible evaluator is `backend/tests/eval_onnx_test_split.py`.

---

## 6. Known issues / caveats

- v1 sensitivity is low, especially hip/proximal and low-contrast films. The UI now says so instead
  of inventing implant advice.
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

## 8. Next steps, in order

1. **Finish v2:** wait for the kernel, download, evaluate, integrate only if it beats v1; recalibrate
   `LOCAL_ONNX_CONF`; restart and re-verify the UI.
2. **Re-test the 9 user samples** after integration; track how many detect and at what confidence.
3. **Region-labeled detector:** acquire a hip/pelvis AP dataset with region classes and fine-tune
   from the current checkpoint at 960. Candidate sources and licensing notes are in `ml/README.md`.
4. Optional: add a model evaluation report (per-region recall, threshold sweep) to `ml/README.md`.
