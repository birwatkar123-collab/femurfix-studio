# ML Training Plan - FemurFix Studio

> **Status:** Resumed 2026-10-25 — dataset rebuilt with the official splits; GPU-ready trainer and notebook added. Full GPU run and hip-region data remain.  
> **Goal:** Fix femoral neck vs shaft misclassification by adding a learned fracture detector, improve realistic animation.

## 1) Problem Recap
Current heuristic in `backend/app/services/xray_analyzer.py` classifies fracture region by vertical ratio of the gradient peak along the femur silhouette. Neck fractures often sit above the shaft band it inspects and get misclassified as shaft. ML is needed.

## 2) Recommended Approach
- **Task**: Object detection (YOLOv11/RT-DETR) to localize fracture and predict region class.
- **Classes**: `femoral_neck`, `intertrochanteric`, `subtrochanteric`, `femoral_shaft`, `distal_femur`, `no_fracture`
- **Strategy**: Pre-train on general fracture datasets, fine-tune on hip/pelvis AP views. Export to ONNX and run locally in backend as fallback/primary.

## 3) Datasets to Search For
| Dataset | Where | Free | Notes |
| :-- | :-- | :-- | :-- |
| FracAtlas | https://github.com/mahmad00/FracAtlas | Yes (research) | Start here. Check for femur/hip subsets and annotations. |
| GRAZPEDWRI-DX | https://www.kaggle.com/datasets/gianmarco96/grazpedwri-dx | Yes | Pretraining on pediatric wrist; transfers well. Has boxes. |
| MURA | https://stanfordmlgroup.github.io/competitions/mura/ | Yes | Good for normal/abnormal pretraining, weak for boxes/AO. |
| Bone Fracture Detection (Kaggle search) | https://www.kaggle.com/search?q=bone%20fracture%20detection%20computer%20vision%20project%20in%3Adatasets | Varies | Search: `femoral neck fracture`, `hip fracture xray`, `proximal femur fracture`, `pelvis hip fracture`. Filter for AP views. |
| RSNA Cervical Spine Fracture | https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection | Yes | Pipeline ideas only. |
| RibFrac/FracNet | https://ribfrac.grand-challenge.org/ | Yes | Reference only. |
| TotalSegmentator (for DRRs) | https://github.com/wasserth/TotalSegmentator | Yes | If data scarce, generate synthetic AP hip views from femur/pelvis CTs. |

## 4) Annotation Schema & Pipeline
### Classes
```yaml
names:
  0: femoral_neck
  1: intertrochanteric
  2: subtrochanteric
  3: femoral_shaft
  4: distal_femur
  5: no_fracture
```

### Rules
- Only AP hip/pelvis views showing proximal femur clearly
- Draw tight bbox around fractured cortices/fragment (neck fractures box around neck/head junction)
- Patient-level split (never split same patient)
- Keep some normal AP hips as negatives
- Be consistent with box coverage

### Structure
```text
ml/
├── data/images/{train,val,test}/
├── data/labels/{train,val,test}/  # YOLO format
├── data.yaml
├── scripts/ (convert_coco_to_yolo.py, split_dataset.py, validate_labels.py)
├── train/train_yolo.py
└── models/ (best.pt, femur_fracture.onnx)
```

## 5) Training Plan
1. **Setup**: `pip install ultralytics onnxruntime` in separate venv
2. **Pretrain** (optional): Merge GRAZPEDWRI-DX + FracAtlas
3. **Finetune**: Start from `yolo11n.pt`, imgsz=640, focus on neck class
4. **Export**: `yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640`
5. **Validate**: Check confusion matrix, especially neck vs shaft

## 6) Integration into Repo
Files to add/change:
| File | Change |
| :-- | :-- |
| `backend/app/services/ml_detector.py` (new) | Load ONNX via onnxruntime, run inference, return normalized fracture dict |
| `backend/app/services/xray_analyzer.py` | Try local ONNX (`backend/models/femur_fracture.onnx`) first, else HF Space, else heuristic. Trust model region for ML path. |
| `backend/models/` (new, gitignore weights) | Drop ONNX here |
| `backend/requirements.txt` | Add `onnxruntime` |
| `ml/` (scaffolded) | Data/training pipeline |

In `xray_analyzer.analyze()`: if local model exists -> get prediction, call `_normalise_fracture(ml_result["fracture"], silhouette)`, append note "Fracture model: local ONNX", set `self.last_engine="ml_local"`. Map model classes to `proximal/shaft/distal` and set AO family.

## 7) Animation Improvements
| Change | Location | Notes |
| :-- | :-- | :-- |
| GSAP tweens | `frontend/src/components/SimView.jsx` | Smoother easing for reduction |
| Tag fragments via userData | `backend/app/services/gltf_exporter.py` | `{role: "proximal"|"distal", bone: "femur"}` instead of name matching |
| Better reduction path | `SimView.jsx` | Interpolate along shortest path, add settle |
| Materials/AO | `SimView.jsx`/exporter | Subtle roughness/AO in gap |
| Camera choreography | `SimView.jsx` | Tween to fracture view on animate |

**Do not** use image-to-video diffusion for in-simulator animation.

## Immediate Next Steps (Resume)
1. Hunt datasets on Kaggle (FracAtlas first) - download and inspect
2. Create annotation schema + conversion scripts
3. Annotate pilot set (15-25 neck + 15-25 others)
4. Train YOLOv11n pilot, export ONNX
5. Wire into backend and test on misclassified neck case

## Progress (2026-10-25)
- [x] FracAtlas downloaded and verified (4,083 images; 717 fractured, 922 boxes).
- [x] Official splits wired (574/82/61 positives) + 3x negatives per split.
- [x] YOLO dataset rebuilt: 2,296/328/244 images, 922 boxes, single `fracture` class.
- [x] `train_yolo.py` fixed (checkpoint paths, test evaluation, ONNX export).
- [x] GPU notebook added: `ml/notebooks/fracatlas_yolo11n_gpu.ipynb`.
- [x] Local 1-epoch CPU smoke test passed (weights + results saved, 0 corrupt images).
- [ ] Full GPU training (next).
- [ ] ONNX integration into the backend.
- [ ] Hip/pelvis AP region dataset + fine-tune (critical path for neck vs shaft).

> Note: the earlier runs in `runs/detect/ml/runs/` stopped before epoch 1 and have no weights.
> They are kept only as evidence and are gitignored.
