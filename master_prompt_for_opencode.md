# MISSION

Build a full-stack medical 3D rendering and animation web application capable of converting 3D DICOM scans (CT/MRI) and 2D Radiographs into interactive, animated 3D WebGL models.

While the initial testing target includes fractured femur bones fixed with internal hardware (plates, screws, nails), the architecture MUST be modular and scalable to process any anatomical region (full skeleton, spine, skull, chest, organs, and soft tissue) by leveraging dynamic region-of-interest (ROI) parameters and multi-class segmentation libraries.

# TOOL USE & EXECUTION ENVIRONMENT PERMISSIONS

You have full authority to execute terminal commands and orchestrate system tools to complete this project autonomously:

1. **System & File Management:** Create directories, write modular files, edit configuration files, and initialize Git repositories.

2. **Package Management:** Run `pip install` for Python packages and `npm install` / `npx` for Node.js/React dependencies.

3. **Execution & Verification:** Launch backend servers (`uvicorn`), build frontend assets (`npm run build`), run verification test scripts, and fix runtime or syntax errors autonomously.

# ARCHITECTURAL DESIGN & REPOSITORY STRUCTURE

Scaffold the project into the following directory structure:

```
medical-3d-pipeline/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI server routes
│   │   ├── services/
│   │   │   ├── dicom_loader.py         # PyDICOM & SimpleITK ingestion
│   │   │   ├── segmenter.py            # TotalSegmentator & MONAI engine
│   │   │   ├── hardware_extractor.py   # High HU metallic thresholding (>1500 HU)
│   │   │   ├── mesh_processor.py       # VTK Marching Cubes, smoothing, decimation
│   │   │   └── gltf_exporter.py        # Multi-node GLTF/GLB assembly
│   │   └── utils/
│   │       └── config.py               # Anatomical presets & HU thresholds
│   ├── tests/
│   │   └── create_test_volume.py       # Synthetic volume matrix generator
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Viewer3D.jsx            # React Three Fiber Canvas & OrbitControls
│   │   │   ├── AnatomicalTree.jsx      # Toggle visibility/opacity per body part
│   │   │   ├── ExplosionControls.jsx   # Sliders for hardware disassembly animation
│   │   │   └── UploadZone.jsx          # DICOM drag-and-drop ingestion UI
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
└── PROMPT.md

```

# STEP-BY-STEP IMPLEMENTATION INSTRUCTIONS

### Step 1: Backend Setup (`backend/`)

1. Create `backend/requirements.txt` with: `fastapi`, `uvicorn`, `pydicom`, `SimpleITK`, `totalsegmentator`, `trimesh`, `nibabel`, `vtk`, `numpy`, `python-multipart`.

2. Install dependencies using `pip`.

3. Implement `backend/app/services/segmenter.py`:

   * Wrap `totalsegmentator` Python API.

   * Accept a parameter `roi_list` (e.g., `["femur_right", "femur_left"]`, `["spine"]`, `["ribs"]`, `["skull"]`, or `["full_skeleton"]`).

   * Allow dynamic switching between targeted bone segmentation and full-body scans.