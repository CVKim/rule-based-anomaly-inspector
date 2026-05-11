"""Rule-based shape classification for anomaly blobs.

Once the dynamic-threshold + morphology pipeline emits a binary anomaly mask,
the connected components are inspected one by one and tagged as one of:

* ``"scratch"``   — long, thin, low solidity.
* ``"spot"``      — small and round.
* ``"dent"``      — round-ish dark blob (mean darker than master).
* ``"smudge"``    — large, low-contrast, irregular.
* ``"unknown"``   — does not match any of the above.

Polarity (``"dark"`` / ``"bright"``) is derived from the sign of the mean
``(target - master)`` over the blob, which is more informative than the
absolute difference used for thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


Category = Literal["scratch", "spot", "dent", "smudge", "unknown"]
Polarity = Literal["dark", "bright"]


@dataclass(frozen=True)
class ShapeFeatures:
    area: int
    perimeter: float
    aspect_ratio: float       # long/short side of min-area rect (>= 1.0)
    circularity: float        # 4πA / P^2 (1.0 = perfect circle)
    solidity: float           # area / convex-hull area (1.0 = convex)
    extent: float             # area / bbox area (1.0 = fills its bbox)
    polarity: Polarity
    signed_mean_diff: float   # mean(target - master) over the blob


def shape_features(label_mask: np.ndarray,
                   signed_diff: np.ndarray) -> ShapeFeatures:
    """Compute geometric + photometric descriptors for one binary blob.

    ``label_mask`` is a uint8 mask of just this blob (255 inside, 0 outside).
    ``signed_diff`` is the full-image ``target - master`` float32 map; we use
    it to derive the polarity sign.
    """
    if label_mask.dtype != np.uint8:
        label_mask = label_mask.astype(np.uint8)

    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        # Degenerate — shouldn't happen, but stay safe.
        return ShapeFeatures(area=0, perimeter=0.0, aspect_ratio=1.0,
                             circularity=0.0, solidity=0.0, extent=0.0,
                             polarity="dark", signed_mean_diff=0.0)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))

    if len(contour) >= 5:
        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), _ = rect
        long_side = max(rw, rh)
        short_side = max(min(rw, rh), 1e-6)
        aspect_ratio = float(long_side / short_side)
    else:
        x, y, w, h = cv2.boundingRect(contour)
        long_side = float(max(w, h))
        short_side = max(float(min(w, h)), 1e-6)
        aspect_ratio = long_side / short_side

    circularity = float(4.0 * np.pi * area / (perimeter ** 2 + 1e-6))
    circularity = min(circularity, 1.0)

    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1e-6)
    solidity = float(area / hull_area)

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = max(float(w * h), 1e-6)
    extent = float(area / bbox_area)

    inside = label_mask > 0
    if inside.any():
        signed_mean = float(signed_diff[inside].mean())
    else:
        signed_mean = 0.0
    polarity: Polarity = "bright" if signed_mean >= 0 else "dark"

    return ShapeFeatures(
        area=int(area), perimeter=perimeter, aspect_ratio=aspect_ratio,
        circularity=circularity, solidity=solidity, extent=extent,
        polarity=polarity, signed_mean_diff=signed_mean,
    )


def classify(features: ShapeFeatures,
             scratch_aspect: float = 3.0,
             spot_max_area: int = 60,
             spot_circularity: float = 0.6,
             smudge_min_area: int = 200,
             smudge_solidity: float = 0.7) -> Category:
    """Map shape descriptors to a single human-meaningful category.

    The thresholds are deliberate, conservative defaults; tune them per line.
    The order of checks is from most-specific to least-specific.
    """
    # Scratches are long and thin — minAreaRect aspect is the discriminator.
    # Solidity intentionally not checked: a clean razor scratch is a near-
    # perfect rectangle (solidity≈1) and we still want to call it a scratch.
    if features.aspect_ratio >= scratch_aspect:
        return "scratch"

    # Tiny round blobs.  Bright = "spot" (e.g. specular glint), dark = "dent".
    if features.area <= spot_max_area and features.circularity >= spot_circularity:
        return "dent" if features.polarity == "dark" else "spot"

    # Larger, irregular regions = smudge / contamination.
    if (features.area >= smudge_min_area
            and features.solidity <= smudge_solidity):
        return "smudge"

    # Round-ish big blob — call it dent if dark, spot otherwise.
    if features.circularity >= 0.5:
        return "dent" if features.polarity == "dark" else "spot"

    return "unknown"


def auto_unreliable_mask(tolerance_map: np.ndarray,
                         percentile: float = 99.0,
                         dilate_px: int = 1) -> np.ndarray:
    """Mark pixels with very high natural variability as untrustworthy.

    Anywhere the per-pixel dispersion exceeds the given percentile of the
    tolerance map is set to 255 in the returned mask, then the mask is
    dilated by ``dilate_px`` to cover the immediate neighbourhood (high-std
    pixels are nearly always near a real edge whose sub-pixel jitter spills
    into adjacent pixels).

    Pass the result as ``ignore_mask`` to ``DynamicToleranceInspector.inspect``
    to suppress the "I always over-detect along the part outline" failure
    mode without needing a hand-drawn ROI.
    """
    if not (0.0 < percentile < 100.0):
        raise ValueError("percentile must be in (0, 100)")
    threshold = float(np.percentile(tolerance_map, percentile))
    # Strict greater-than so a degenerate "almost-everywhere zero" tolerance
    # map (with a small high-variance pocket) doesn't flag the entire image.
    mask = (tolerance_map > threshold).astype(np.uint8) * 255
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask
