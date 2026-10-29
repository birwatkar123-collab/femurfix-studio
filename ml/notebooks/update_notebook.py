import json
path = r"D:\New folder (3)\ml\notebooks\region_yolo_finetune_gpu.ipynb"
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell['source'])
    if 'REGION-DATASET' in src:
        cell['source'] = [
            'from pathlib import Path\n',
            'from ultralytics import YOLO\n',
            '\n',
            '# The dataset was uploaded as birwatkar/femurfix-region-dataset\n',
            'DATA_YAML = Path("/kaggle/input/birwatkar/femurfix-region-dataset/data.yaml")\n',
            'assert DATA_YAML.is_file(), "attach birwatkar/femurfix-region-dataset in the kernel\'s Input panel"\n',
            '\n',
            '# fine-tune from the v3 backbone (best.pt is the 5-class region model target; v3 is the best\n',
            '# fracture detector). Attach best_v3.pt from Kaggle as a second dataset if available.\n',
            'model = YOLO("yolo11s.pt")  # or YOLO("/kaggle/input/.../best_v3.pt") if attached\n',
            'model.train(\n',
            '    data=str(DATA_YAML),\n',
            '    epochs=60,\n',
            '    patience=20,\n',
            '    imgsz=960,\n',
            '    batch=8,\n',
            '    device=0,\n',
            '    seed=0,\n',
            '    deterministic=True,\n',
            '    cache=True,\n',
            '    fliplr=0.5,\n',
            '    degrees=10.0,\n',
            '    close_mosaic=12,\n',
            '    project="/kaggle/working/runs",\n',
            '    name="region_yolo11s_960",\n',
            ')\n'
        ]
        break

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
print('Notebook updated')