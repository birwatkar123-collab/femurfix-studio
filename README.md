# FemurFix Studio

Educational femur-fracture fixation simulator.

Upload a femur X-ray (or CT/MRI) → fracture analysis → estimated 3D femur → fixation options →
animated 3D assembly with the selected implant.

**Disclaimer: educational simulation only, not medical advice.**

See `SPEC.md` for the product and architecture spec.

## Layout

```
backend/    FastAPI pipeline (analysis, reconstruction, implants, assembly)
frontend/   React + React Three Fiber wizard and 3D viewer
ml/space/   Hugging Face ZeroGPU Space (fracture detection + bone segmentation)
```

## Quick start

### Backend

```powershell
cd backend
C:\Users\mansi\AppData\Local\Programs\Python\Python311\python.exe -m pip install -r requirements.txt
C:\Users\mansi\AppData\Local\Programs\Python\Python311\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### HF Space (optional, for ML inference)

```powershell
cd ml\space
python app.py
```

Set `HF_SPACE_ID=<user>/<space>` in the backend environment to route analysis through ZeroGPU.
Without it the backend uses its deterministic, clearly-labelled fallback analyzer.
