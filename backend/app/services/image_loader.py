"""Image ingestion for radiographs (PNG/JPG/DICOM) and volumetric CT/MRI series."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..utils.config import ALLOWED_IMAGE_SUFFIXES


class ImageLoadError(RuntimeError):
    """Raised when an upload cannot be interpreted as a supported medical image."""


def _dicom_pixel_array(ds: Any) -> np.ndarray:
    if not hasattr(ds, "PixelData"):
        raise ImageLoadError("DICOM file has no pixel data")
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return arr * slope + intercept


def _normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32))
    low, high = np.percentile(arr, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(arr.min()), float(arr.max() or 1.0)
    if high <= low:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - low) / (high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


class ImageLoader:
    """Loads a radiograph or a CT/MRI volume from a file, directory, or ZIP."""

    def load_xray(self, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            raise ImageLoadError(f"File not found: {path}")

        meta: Dict[str, Any] = {
            "source": path.name,
            "modality": "XRAY",
            "pixel_spacing_mm": None,
            "calibrated": False,
        }

        if path.suffix.lower() in {".dcm", ".dicom"}:
            gray = self._load_dicom_radiograph(path, meta)
        else:
            try:
                with Image.open(path) as img:
                    gray = np.array(img.convert("L"), dtype=np.float32)
            except Exception as exc:
                raise ImageLoadError(f"Unreadable image '{path.name}': {exc}") from exc
            if gray.ndim != 2:
                raise ImageLoadError("Expected a single-frame grayscale radiograph")

        # Very large photographs slow the pipeline down; keep the long edge sane.
        gray = self._downscale(gray, max_edge=2048)
        meta["width"], meta["height"] = int(gray.shape[1]), int(gray.shape[0])
        return _normalise_to_uint8(gray), meta

    def _load_dicom_radiograph(self, path: Path, meta: Dict[str, Any]) -> np.ndarray:
        try:
            import pydicom
        except Exception as exc:  # pragma: no cover - dependency is installed
            raise ImageLoadError(f"pydicom unavailable: {exc}") from exc

        try:
            ds = pydicom.dcmread(str(path), force=False)
            arr = _dicom_pixel_array(ds)
        except Exception as exc:
            raise ImageLoadError(f"Corrupt DICOM file '{path.name}': {exc}") from exc

        spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
        if spacing and len(spacing) == 2:
            try:
                meta["pixel_spacing_mm"] = [float(spacing[0]), float(spacing[1])]
                meta["calibrated"] = True
            except (TypeError, ValueError):
                meta["pixel_spacing_mm"] = None
        meta["modality"] = str(getattr(ds, "Modality", "XRAY"))
        return arr

    @staticmethod
    def _downscale(arr: np.ndarray, max_edge: int) -> np.ndarray:
        height, width = arr.shape[:2]
        longest = max(height, width)
        if longest <= max_edge:
            return arr
        factor = max_edge / float(longest)
        new_size = (max(1, int(width * factor)), max(1, int(height * factor)))
        img = Image.fromarray(_normalise_to_uint8(arr)).resize(new_size, Image.BILINEAR)
        return np.array(img, dtype=np.float32)

    # ------------------------------------------------------------------ volumes
    def load_volume(self, path: Path) -> Tuple[Any, np.ndarray, Tuple[float, float, float], Dict[str, Any]]:
        """Load a CT/MRI volume. Returns (sitk_image, array, spacing, meta)."""
        try:
            import SimpleITK as sitk
        except Exception as exc:  # pragma: no cover
            raise ImageLoadError(f"SimpleITK unavailable: {exc}") from exc

        path = Path(path)
        try:
            if path.is_dir():
                image = self._read_series(path)
            elif path.suffix.lower() == ".zip":
                image = self._read_zip(path)
            elif path.suffix.lower() in {".nii", ".gz", ".mha", ".mhd"}:
                image = sitk.ReadImage(str(path))
            else:
                image = sitk.ReadImage(str(path))
        except Exception as exc:
            raise ImageLoadError(f"Failed to read volume: {exc}") from exc

        array = sitk.GetArrayFromImage(image).astype(np.float32)
        if array.size == 0:
            raise ImageLoadError("Volume contains no voxels")

        spacing = tuple(float(v) for v in image.GetSpacing()[::-1])  # z, y, x
        meta = {
            "modality": "CT/MRI",
            "source": path.name,
            "shape": list(array.shape),
            "spacing": list(spacing),
        }
        return image, array, spacing, meta

    def _read_series(self, directory: Path) -> Any:
        import SimpleITK as sitk

        files = self._dicom_files(directory)
        if not files:
            raise ImageLoadError("No DICOM slices found in the uploaded series")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        return reader.Execute()

    def _read_zip(self, zip_path: Path) -> Any:
        import SimpleITK as sitk

        extract_dir = zip_path.parent / f"_{zip_path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            raise ImageLoadError(f"Corrupt ZIP archive: {exc}") from exc

        files = self._dicom_files(extract_dir)
        if not files:
            raise ImageLoadError("ZIP contains no DICOM slices")
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(files)
        return reader.Execute()

    @staticmethod
    def _dicom_files(directory: Path) -> List[str]:
        import pydicom

        found: List[str] = []
        for candidate in sorted(directory.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES | {"", ".ima"}:
                continue
            try:
                pydicom.dcmread(str(candidate), stop_before_pixels=True, force=True)
                found.append(str(candidate))
            except Exception:
                continue
        return found

    def detect_kind(self, path: Path) -> str:
        """Classify an upload as 'xray' or 'volume' without fully loading it."""
        path = Path(path)
        if path.suffix.lower() == ".zip" or path.is_dir():
            return "volume"
        if path.suffix.lower() in {".nii", ".mha", ".mhd"} or path.suffix.lower().endswith(".nii.gz"):
            return "volume"
        if path.suffix.lower() in {".dcm", ".dicom"}:
            try:
                import pydicom

                ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
                if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1 or str(getattr(ds, "Modality", "")).upper() in {"CT", "MR"}:
                    return "volume"
            except Exception:
                pass
        return "xray"
