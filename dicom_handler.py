"""Read DICOM (.dcm / .dicom) chest X-rays into a normal 8-bit RGB image."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

DICOM_EXTENSIONS = (".dcm", ".dicom")

# (tag name, label shown to the user). Patient name / ID are deliberately left out.
_FIELDS = [
    ("Modality", "Modality"),
    ("BodyPartExamined", "Body part"),
    ("ViewPosition", "View position"),
    ("StudyDate", "Study date"),
    ("StudyDescription", "Study description"),
    ("Manufacturer", "Manufacturer"),
    ("Rows", "Rows"),
    ("Columns", "Columns"),
    ("PixelSpacing", "Pixel spacing"),
    ("PhotometricInterpretation", "Photometric interpretation"),
]


def is_dicom_name(name: str) -> bool:
    return str(name).lower().endswith(DICOM_EXTENSIONS)


def _first(value):
    """DICOM values can be single numbers or lists; take the first number."""
    try:
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            value = list(value)[0]
        return float(value)
    except Exception:
        return None


def read_dicom(data: bytes) -> tuple[Image.Image, dict]:
    """Return (RGB PIL image, metadata dict) for the bytes of a DICOM file."""
    import pydicom  # imported here so the app still starts if it is missing

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    if "PixelData" not in ds:
        raise ValueError("this DICOM file contains no image data")

    try:
        arr = np.asarray(ds.pixel_array)
    except Exception as exc:
        raise ValueError(
            f"could not decode the pixel data ({exc}). Compressed DICOM files may need extra packages."
        )

    # Multi-frame: use the first frame.
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    if samples == 1 and arr.ndim == 3:
        arr = arr[0]
    elif samples > 1 and arr.ndim == 4:
        arr = arr[0]

    if samples >= 3 and arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float32)
        lo, hi = float(rgb.min()), float(rgb.max())
        rgb = (rgb - lo) / max(hi - lo, 1e-6) * 255.0
        img = Image.fromarray(rgb.astype(np.uint8), "RGB")
    else:
        gray = arr.astype(np.float32)
        slope = _first(getattr(ds, "RescaleSlope", None)) or 1.0
        intercept = _first(getattr(ds, "RescaleIntercept", None)) or 0.0
        gray = gray * slope + intercept

        center = _first(getattr(ds, "WindowCenter", None))
        width = _first(getattr(ds, "WindowWidth", None))
        if center is not None and width and width > 1:
            lo, hi = center - width / 2.0, center + width / 2.0
        else:
            lo, hi = np.percentile(gray, [0.5, 99.5])
        if hi <= lo:
            lo, hi = float(gray.min()), float(gray.max())
        gray = np.clip((gray - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0

        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            gray = 255.0 - gray  # MONOCHROME1 stores inverted values
        img = Image.fromarray(gray.astype(np.uint8), "L").convert("RGB")

    meta = {}
    for tag, label in _FIELDS:
        value = getattr(ds, tag, None)
        if value not in (None, ""):
            try:
                if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    value = ", ".join(str(v) for v in value)
                meta[label] = str(value)
            except Exception:
                pass
    return img, meta
