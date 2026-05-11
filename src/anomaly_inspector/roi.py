"""Automatic part-region (ROI) extraction.

Backlit / dark-field side-view rigs typically capture a small bright part
against a much larger dark background — often <20% of the frame is the part
itself. The dynamic-tolerance pipeline does not care about pixels outside
the part, but residual modes that don't degenerate to zero on a dark
background (NCC, gradient) will produce nuisance detections out there.

This module computes a binary "where the part lives" mask once at reference
build time and persists it in the .npz, so the inspector can suppress every
out-of-part response in a single ``bitwise_and`` regardless of which
residual mode is in play.

Algorithm (``method="otsu_close"``, the default):

1. Otsu threshold the master to peel the bright surfaces off the dark
   background.
2. Morphological **closing** with a large square kernel — fills the
   recesses, slots and shadows that live *inside* the part but happen to
   read darker than the Otsu threshold.
3. Keep only the **largest connected component**, killing isolated bright
   specks (particles, sensor hot pixels, marker glints).
4. Optionally **erode** by a few pixels so the residual halo right on the
   part outline is excluded — sub-pixel alignment never lines up the rim
   exactly, and that band is the single biggest source of false positives.
5. Optionally **convex hull** for parts whose true silhouette is
   guaranteed to be convex; this also patches up any concave bays the
   close kernel didn't reach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


Method = Literal["none", "otsu_close", "fixed_threshold"]


@dataclass(frozen=True)
class RoiConfig:
    """How the part-region mask is derived from the master image.

    ``method="none"`` skips ROI extraction entirely (full image is in scope).
    """

    method: Method = "none"
    # otsu_close
    close_ksize: int = 41         # >> typical recess width; odd
    erode_px: int = 6             # exclude the boundary halo
    convex_hull: bool = False     # tighten the silhouette to convex
    # fixed_threshold (rare — when Otsu is unreliable)
    fixed_value: float = 32.0
    # shared
    min_area_fraction: float = 0.005   # a "part" must occupy at least 0.5%
    # Reject masks that cover almost the entire frame — that means Otsu
    # found no real bimodal split (e.g. blank or saturated image) and a
    # blanket mask would defeat the point of having an ROI at all.
    max_area_fraction: float = 0.95

    def __post_init__(self) -> None:
        if self.method not in ("none", "otsu_close", "fixed_threshold"):
            raise ValueError(f"unknown method '{self.method}'")
        if self.close_ksize < 3 or self.close_ksize % 2 == 0:
            raise ValueError("close_ksize must be an odd integer >= 3")
        if self.erode_px < 0:
            raise ValueError("erode_px must be >= 0")
        if self.fixed_value < 0:
            raise ValueError("fixed_value must be >= 0")
        if not (0.0 <= self.min_area_fraction < 1.0):
            raise ValueError("min_area_fraction must be in [0, 1)")
        if not (0.0 < self.max_area_fraction <= 1.0):
            raise ValueError("max_area_fraction must be in (0, 1]")
        if self.max_area_fraction <= self.min_area_fraction:
            raise ValueError("max_area_fraction must exceed min_area_fraction")

    def to_meta(self) -> dict:
        return {
            "method": self.method,
            "close_ksize": int(self.close_ksize),
            "erode_px": int(self.erode_px),
            "convex_hull": bool(self.convex_hull),
            "fixed_value": float(self.fixed_value),
            "min_area_fraction": float(self.min_area_fraction),
            "max_area_fraction": float(self.max_area_fraction),
        }

    @classmethod
    def from_meta(cls, meta: dict | None) -> "RoiConfig":
        if not meta:
            return cls()
        return cls(
            method=meta.get("method", "none"),
            close_ksize=int(meta.get("close_ksize", 41)),
            erode_px=int(meta.get("erode_px", 6)),
            convex_hull=bool(meta.get("convex_hull", False)),
            fixed_value=float(meta.get("fixed_value", 32.0)),
            min_area_fraction=float(meta.get("min_area_fraction", 0.005)),
            max_area_fraction=float(meta.get("max_area_fraction", 0.95)),
        )


def auto_part_roi(master: np.ndarray, config: RoiConfig) -> np.ndarray | None:
    """Return a uint8 mask (255 inside the part) or None if disabled.

    ``master`` may be uint8 or float32. The returned mask has the same
    HxW shape as the master.
    """
    if config.method == "none":
        return None

    src = master if master.dtype == np.uint8 else _to_uint8(master)

    # 1) seed mask
    if config.method == "otsu_close":
        # OTSU returns a float threshold value via cv2.threshold's first
        # output; we ignore it and just keep the binary mask.
        _, seed = cv2.threshold(src, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif config.method == "fixed_threshold":
        _, seed = cv2.threshold(src, float(config.fixed_value), 255,
                                cv2.THRESH_BINARY)
    else:
        raise AssertionError(f"unhandled method '{config.method}'")  # pragma: no cover

    # 2) close to fill dark recesses *inside* the part
    if config.close_ksize > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.close_ksize, config.close_ksize)
        )
        closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k)
    else:
        closed = seed

    # 3) keep only the largest CC that meets the min-area floor
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    largest_area = float(areas[largest_idx - 1])
    if largest_area < config.min_area_fraction * src.size:
        return None
    if largest_area > config.max_area_fraction * src.size:
        # Otsu found no real bimodal split (e.g. blank or saturated frame);
        # a near-blanket mask is worse than no mask, so bail out.
        return None
    part = (labels == largest_idx).astype(np.uint8) * 255

    # 4) optional convex hull (nice for parts whose silhouette is convex
    #    by design — this skips the bay-filling that closing didn't reach)
    if config.convex_hull:
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            hull = cv2.convexHull(np.vstack(contours))
            part = np.zeros_like(part)
            cv2.drawContours(part, [hull], -1, 255, thickness=cv2.FILLED)

    # 5) erode boundary halo
    if config.erode_px > 0:
        ek = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * config.erode_px + 1, 2 * config.erode_px + 1)
        )
        part = cv2.erode(part, ek)

    return part


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 255.0).astype(np.uint8)
