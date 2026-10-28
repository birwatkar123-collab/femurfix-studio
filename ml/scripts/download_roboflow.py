#!/usr/bin/env python3
"""Download the licensed proximal-femur dataset from Roboflow Universe.

Target: ``thesisyolo-v8/proximal-femur-fracture`` (756 images, instance
segmentation, CC BY 4.0) — the best licensed source for the region track
(see ``ml/README.md`` and the prepared ``prepare_region_dataset.py``).

Usage::

    set RF_API_KEY=...                     # Windows (or --api-key / stdin)
    python ml/scripts/download_roboflow.py \
        --workspace thesisyolo-v8 \
        --project proximal-femur-fracture \
        --format yolov8 \
        --dst ml/data/raw/proximal-femur-fracture

Then convert::

    python ml/scripts/prepare_region_dataset.py \\
        --source ml/data/raw/proximal-femur-fracture/<version dir> \\
        --dst ml/data/region --keep-negatives

The Roboflow key is read from the RF_API_KEY environment variable (never pass
it on the command line, never commit it).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _api_request(url: str, params: dict[str, str]) -> dict:
    from urllib.parse import urlencode

    full = f"{url}?{urlencode(params)}"
    request = urllib.request.Request(full, headers={"User-Agent": "femurfix-ml/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        # Some export endpoints redirect to the zip itself.
        raise SystemExit(f"Unexpected non-JSON response from {url} (HTTP body {len(body)} bytes). "
                         "Check the workspace/project names and your API key.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default="thesisyolo-v8")
    parser.add_argument("--project", default="proximal-femur-fracture")
    parser.add_argument("--version", type=int, default=None, help="Pin a version; default = latest with an export")
    parser.add_argument("--format", default="yolov8",
                        help="Export format: yolov8 / yolov11 / yolov9 / coco; YOLO variants give bbox labels")
    parser.add_argument("--dst", type=Path, default=ROOT / "data" / "raw" / "proximal-femur-fracture")
    parser.add_argument("--api-key", default=None, help="Prefer the RF_API_KEY env var or interactive stdin")
    parser.add_argument("--download-only", action="store_true", help="Do not unzip; keep the archive")
    args = parser.parse_args()

    key = args.api_key or os.environ.get("RF_API_KEY")
    if not key:
        key = getpass.getpass("Roboflow API key (input hidden, stored nowhere): ")
    if not key:
        raise SystemExit("No API key provided. Create one at https://app.roboflow.com/settings/api")

    slug = f"{args.workspace}/{args.project}"
    info = _api_request(f"https://api.roboflow.com/{slug}", {"api_key": key})
    versions = info.get("versions") or []
    if not versions:
        raise SystemExit(f"No versions published for {slug}. Check the project name.")

    def version_number(entry: dict) -> str:
        # Roboflow shapes vary: entry["version"] when present, else the id suffix.
        num = entry.get("version")
        if num is None and "/" in (entry.get("id") or ""):
            num = entry["id"].rsplit("/", 1)[-1]
        if num is None:
            raise SystemExit(f"Cannot determine version for entry {entry}")
        return str(num)

    def export_formats(entry: dict) -> set:
        raw = entry.get("exports")
        if isinstance(raw, list):
            return {str(x) for x in raw}
        if isinstance(raw, str):
            return set(raw.split())
        export = entry.get("export") or {}
        return {"ready"} if export.get("link") else set()

    chosen = None
    for entry in sorted(versions, key=lambda v: version_number(v), reverse=True):
        num = version_number(entry)
        if args.version is None and export_formats(entry):
            chosen = entry
            break
        if args.version is not None and num == str(args.version):
            chosen = entry
            break
    if chosen is None:
        raise SystemExit(
            f"Version {args.version or '(any)'} with a ready export not found. "
            f"Published: {[version_number(v) + ':' + v.get('name', '') for v in versions][:10]}"
        )
    available = export_formats(chosen)
    if available and args.format not in available:
        print(f"note: version {version_number(chosen)} exports: {sorted(available)}; "
              f"trying '{args.format}' anyway")

    version_num = version_number(chosen)
    print(f"project={slug} version={version_num} name={chosen.get('name')} images={chosen.get('images')}")

    # Request the export (Roboflow generates it on first request; this call
    # returns an `export.link` json or a queued response we would retry).
    export_url = (
        f"https://api.roboflow.com/{args.workspace}/{args.project}/"
        f"{version_num}/{args.format}"
    )
    export = _api_request(export_url, {"api_key": key, "version": str(version_num), "overwrite": "false"})
    payload = export.get("export") or {}
    link = payload.get("link")
    status = payload.get("status") or export.get("status")
    if not link:
        raise SystemExit(f"Export not ready (status={status}). Retry in a minute or pick another --format.")

    args.dst.mkdir(parents=True, exist_ok=True)
    zip_path = args.dst / f"proximal-femur-fracture-v{version_num}-{args.format}.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print(f"downloading -> {zip_path}")
        urllib.request.urlretrieve(link, zip_path)
    else:
        print(f"cached archive: {zip_path}")

    if args.download_only:
        return
    out_dir = args.dst / f"v{version_num}"
    if not (out_dir / "train").exists() and not (out_dir / "images").exists():
        print(f"unzipping -> {out_dir}")
        with zipfile.ZipFile(zip_path) as handle:
            handle.extractall(out_dir)
    print(
        "\nNext step:\n"
        f"  python ml/scripts/prepare_region_dataset.py \\\n"
        f"      --source {out_dir} \\\n"
        "      --dst ml/data/region --keep-negatives\n"
        "Then validate labels and review ml/data/region/review_manifest.csv."
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - network path
        print(f"download failed: {exc}", file=sys.stderr)
        sys.exit(1)
