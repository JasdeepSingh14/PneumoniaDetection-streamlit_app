"""Severity estimate derived from the Grad-CAM heatmap.

This is a simple, transparent heuristic: it measures how much of the image the
model focused on (heatmap >= 50% of its peak) and splits that area into
Mild / Moderate / Severe bands. It is NOT a clinical severity grade.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

THRESHOLD = 0.5          # heatmap value that counts as "highlighted"
MILD_BELOW = 20.0        # % of image area highlighted
MODERATE_BELOW = 40.0

_NOTES = {
    "Mild": "The model's attention covers a small part of the image.",
    "Moderate": "The model's attention covers a moderate part of the image.",
    "Severe": "The model's attention covers a large part of the image.",
}


def estimate_severity(cam: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """Return severity level, affected area (%) and a regional breakdown.

    `cam` is the normalised (0-1) Grad-CAM map. Region names refer to the image
    as displayed (left/right of the picture), not the patient's anatomical side.
    """
    cam = np.clip(np.asarray(cam, dtype=float), 0.0, 1.0)
    big = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    mask = (np.asarray(big) / 255.0) >= threshold

    affected = float(mask.mean() * 100.0)
    regions = {
        "Upper left": float(mask[:112, :112].mean() * 100.0),
        "Upper right": float(mask[:112, 112:].mean() * 100.0),
        "Lower left": float(mask[112:, :112].mean() * 100.0),
        "Lower right": float(mask[112:, 112:].mean() * 100.0),
        "Central": float(mask[56:168, 56:168].mean() * 100.0),
    }

    if affected < MILD_BELOW:
        level = "Mild"
    elif affected < MODERATE_BELOW:
        level = "Moderate"
    else:
        level = "Severe"

    return {
        "severity_level": level,
        "affected_percentage": affected,
        "affected_regions": regions,
        "clinical_note": _NOTES[level]
        + " This is an estimate from the heatmap, not a clinical grade; a clinician must review the X-ray.",
    }
