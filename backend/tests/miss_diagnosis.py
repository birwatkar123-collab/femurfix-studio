"""Diagnose detector misses instead of guessing at them.

`sample_baseline.py` says *whether* a sample fired. This says *why not*, by running
the same production preprocessing/postprocessing at a probe confidence far below the
app gate and reporting the whole score distribution: a miss with max score 0.13 is a
threshold problem, a miss with max score 0.02 is a data problem, and the two need
completely different fixes.

Writes `ml/reports/miss_analysis.md` plus an overlay PNG per miss showing where the
model's best low-confidence candidates actually landed.

Adjudication lives in `ml/reports/sample_truth.json` (one entry per sample: is the bone a
femur, is a fracture present, and where). It is a **provisional, non-clinical** label set
whose only job is to stop counting a forearm film and a normal hip as detector misses.
Samples with no entry are reported as `unadjudicated` and excluded from the in-scope recall
figure.

Usage (from the repo root):
    python backend/tests/miss_diagnosis.py
    python backend/tests/miss_diagnosis.py --model backend/models/femur_region.onnx --probe-conf 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend"))

import numpy as np
from PIL import Image, ImageDraw

from app.services.image_loader import ImageLoader
from app.services.ml_detector import OnnxFractureDetector
from app.utils import config


def _overlay(image: np.ndarray, boxes: List[Dict[str, Any]], path: Path, count: int = 3) -> None:
    """Draw the top-`count` low-confidence candidates on the original frame."""
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    canvas = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    colours = [(255, 80, 80), (255, 180, 40), (80, 200, 255)]
    for i, box in enumerate(boxes[:count]):
        x0, y0, x1, y1 = [float(v) for v in box["bbox_pixels"]]
        colour = colours[i % len(colours)]
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)
        draw.text((x0 + 3, max(0, y0 - 12)), f"#{i + 1} {box['confidence']:.3f}", fill=colour)
    canvas.save(path)


def _status(fired: bool, max_score: float, truth: Dict[str, Any] | None) -> str:
    if truth is None:
        return "unadjudicated"
    in_scope = truth.get("in_scope", True)
    expected = truth.get("fracture")
    if in_scope is None or expected is None:
        return "indeterminate"
    if not in_scope:
        return "out-of-scope"
    if expected:
        if fired:
            return "hit"
        return "miss-below-gate" if max_score >= 0.05 else "miss-not-seen"
    return "false-positive" if fired else "correct-abstain"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=config.LOCAL_ONNX_MODEL)
    parser.add_argument("--probe-conf", type=float, default=0.01,
                        help="confidence used to keep candidates below the app gate")
    parser.add_argument("--gate", type=float, default=config.LOCAL_ONNX_CONF,
                        help="app decision threshold (default LOCAL_ONNX_CONF)")
    parser.add_argument("--samples", type=Path, default=REPO / "samples")
    parser.add_argument("--truth", type=Path, default=None,
                        help="adjudication JSON (default ml/reports/sample_truth.json)")
    parser.add_argument("--outdir", type=Path, default=REPO / "ml" / "reports")
    args = parser.parse_args()

    truth_path = args.truth or (args.outdir / "sample_truth.json")
    truth_all: Dict[str, Any] = {}
    if truth_path.exists():
        truth_all = json.loads(truth_path.read_text(encoding="utf-8"))

    det = OnnxFractureDetector(model_path=args.model, conf=args.probe_conf)
    files = sorted(
        p for p in args.samples.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    overlay_dir = args.outdir / "miss_boxes"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    loader = ImageLoader()
    for path in files:
        image, _meta = loader.load_xray(path)
        height, width = image.shape[:2]
        tensor, scale, pad = det._preprocess(image)
        pred = np.asarray(det.session.run(None, {det.input_name: tensor})[0], dtype=np.float32)
        boxes = det._postprocess(pred, image.shape[:2], scale, pad)
        scores = sorted((float(b["confidence"]) for b in boxes), reverse=True)
        top = scores[0] if scores else 0.0
        best = boxes[0] if boxes else None
        centre = None
        if best:
            x0, y0, x1, y1 = [float(v) for v in best["bbox_norm"]]
            centre = (round((x0 + x1) / 2, 3), round((y0 + y1) / 2, 3))
        truth = truth_all.get(path.name)
        rows.append({
            "file": path.name,
            "size": [width, height],
            "scale": round(float(scale), 3),
            "fired": top >= args.gate,
            "max_score": round(top, 4),
            "max_calibrated": (
                round(det._calibrate(top), 4) if det.calibration and top > 0 else None
            ),
            "scores": [round(s, 4) for s in scores],
            "top_scores": [round(s, 4) for s in scores[:5]],
            "best_centre_norm": centre,
            "best_area_frac": (
                round(float((best["bbox_norm"][2] - best["bbox_norm"][0])
                            * (best["bbox_norm"][3] - best["bbox_norm"][1])), 4)
                if best else None
            ),
            "truth": truth,
            "status": _status(top >= args.gate, top, truth),
        })
        if rows[-1]["status"] != "hit":
            _overlay(image, boxes, overlay_dir / f"{path.stem}.png")

    def c(r: Dict[str, Any], threshold: float) -> int:
        return sum(1 for s in r["scores"] if s >= threshold)

    adjudicated = [r for r in rows if r["truth"]]
    unadjudicated = [r for r in rows if not r["truth"]]
    by = lambda *statuses: [r for r in rows if r["status"] in statuses]
    positives = by("hit", "miss-below-gate", "miss-not-seen")
    negatives = by("false-positive", "correct-abstain")
    out_of_scope = by("out-of-scope")
    indeterminate = by("indeterminate")
    hits = by("hit")
    below_gate = by("miss-below-gate")
    not_seen = by("miss-not-seen")
    false_positives = by("false-positive")

    fired_all = sum(1 for r in rows if r["fired"])
    lines: List[str] = [
        "# Miss diagnosis",
        "",
        f"Model `{det.path.name}` (imgsz {det.imgsz}), probe conf {args.probe_conf}, app gate {args.gate}.",
        f"Generated by `backend/tests/miss_diagnosis.py`. Overlays in `ml/reports/miss_boxes/`.",
        f"Adjudication from `{truth_path.name}`: {len(adjudicated)}/{len(rows)} samples labelled, "
        f"{len(unadjudicated)} unadjudicated.",
        "",
        f"- raw fired: {fired_all}/{len(rows)} — **this is the number `sample_baseline.py` prints, and "
        "it is the wrong denominator** (see below)",
        f"- in-scope positives: {len(positives)} -> hit {len(hits)}, "
        f"**below-gate {len(below_gate)}**, **not-seen {len(not_seen)}**",
        f"- in-scope negatives: {len(negatives)} -> correct abstain "
        f"{len(negatives) - len(false_positives)}, false positive {len(false_positives)}",
        f"- out of scope (not a femur): {len(out_of_scope)}",
        f"- indeterminate (needs a clinical read): {len(indeterminate)}",
        "",
        "Below-gate means the model put a candidate on the pathology and the gate rejected it — a "
        "threshold problem. Not-seen means nothing in the frame scored above 0.05 — a data problem. "
        "They need different fixes, and counting them together hides that.",
        "",
        "| sample | size | scale | fired | max | cal | #>0.05 | best centre (x,y) | best area | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for r in rows:
        centre = r["best_centre_norm"]
        cal = r["max_calibrated"]
        lines.append(
            f"| {r['file']} | {r['size'][0]}x{r['size'][1]} | {r['scale']} | "
            f"{'yes' if r['fired'] else 'NO'} | {r['max_score']:.3f} | "
            f"{f'{cal:.3f}' if cal is not None else '-'} | {c(r, 0.05)} | "
            f"{f'{centre[0]},{centre[1]}' if centre else '-'} | "
            f"{r['best_area_frac'] if r['best_area_frac'] is not None else '-'} | {r['status']} |"
        )
    lines += [
        "",
        "`scale` is `min(imgsz/w, imgsz/h)`; above 1.0 the frame is **upsampled** into the network, so "
        "lost resolution cannot explain that miss. `cal` is the isotonic-calibrated probability of the "
        "best raw score — the gate at raw "
        f"{args.gate} sits at calibrated {det._calibrate(args.gate) if det.calibration else 'n/a'}.",
        "",
    ]
    if below_gate:
        lines.append("## Below-gate (a candidate exists, the gate rejected it)")
        for r in below_gate:
            lines.append(f"- `{r['file']}` max {r['max_score']:.3f} "
                         f"(cal {r['max_calibrated']}), {c(r, 0.05)} candidate(s) >= 0.05, "
                         f"best centred at {r['best_centre_norm']} — "
                         f"{(r['truth'] or {}).get('site', '')}")
        lines.append("")
    if not_seen:
        lines.append("## Not-seen (nothing above 0.05 anywhere in the frame)")
        for r in not_seen:
            lines.append(f"- `{r['file']}` max {r['max_score']:.3f} at scale {r['scale']}, "
                         f"best candidate at {r['best_centre_norm']} — "
                         f"{(r['truth'] or {}).get('site', '')}")
        lines.append("")
    if out_of_scope:
        lines.append("## Out of scope (excluded from recall)")
        for r in out_of_scope:
            lines.append(f"- `{r['file']}` {(r['truth'] or {}).get('note', '')}")
        lines.append("")
    if indeterminate:
        lines.append("## Indeterminate (excluded until a clinician reads them)")
        for r in indeterminate:
            lines.append(f"- `{r['file']}` max {r['max_score']:.3f} — "
                         f"{(r['truth'] or {}).get('note', '')}")
        lines.append("")
    if unadjudicated:
        lines.append("## Unadjudicated (add an entry to the truth JSON to classify these)")
        lines.append(", ".join(f"`{r['file']}`" for r in unadjudicated))
        lines.append("")

    (args.outdir / "miss_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    (args.outdir / "miss_analysis.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for r in rows:
        flag = "+" if r["fired"] else "-"
        print(f"  [{flag}] {r['file']:44s} max={r['max_score']:.3f} "
              f"cal={r['max_calibrated']} {r['status']} scale={r['scale']}")
    print(f"\nfired {fired_all}/{len(rows)} (raw) | in-scope positives {len(positives)}: "
          f"hit {len(hits)}, below-gate {len(below_gate)}, not-seen {len(not_seen)} | "
          f"false positives {len(false_positives)} | unadjudicated {len(unadjudicated)}")
    print(f"report -> {args.outdir / 'miss_analysis.md'}")


if __name__ == "__main__":
    main()
