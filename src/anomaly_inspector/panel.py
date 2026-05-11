"""Six-panel debug visualisation for anomaly inspection results.

Layout (left -> right):

    image | heatmap | mask pred | pred conf fg | pred conf bg | overlay

* ``image``         — the aligned target as displayed (grayscale -> BGR).
* ``heatmap``       — JET-coloured residual magnitude over the image.
* ``mask pred``     — image with the binary anomaly mask tinted in cyan.
* ``pred conf fg``  — viridis-coloured "anomaly confidence" map (residual /
  per-pixel threshold), masked to the predicted anomaly region.
* ``pred conf bg``  — the complementary "background confidence" (1 - fg) over
  the *non*-anomaly region, in cool tones.
* ``overlay``       — image with the per-defect bounding boxes from
  ``draw_defects`` (color-coded by category).

All cells are resized to the same height; the function caps total width to
keep PNG files manageable for very large source images (12 MP+).
"""

from __future__ import annotations

import cv2
import numpy as np

from .inspector import InspectionResult
from .visualization import draw_defects


def make_panel(image: np.ndarray, result: InspectionResult,
               threshold_map: np.ndarray | None = None,
               max_cell_width: int = 600,
               title: str | None = None,
               roi_mask: np.ndarray | None = None) -> np.ndarray:
    """Compose a horizontal six-cell panel as a single BGR uint8 image.

    Parameters
    ----------
    image:
        Grayscale or BGR original (will be converted to BGR if needed).
    result:
        The ``InspectionResult`` returned by ``DynamicToleranceInspector``.
    threshold_map:
        Optional override; defaults to ``result.threshold_map`` for the
        confidence ratio.
    max_cell_width:
        Each cell is resized so its width is at most this value while
        preserving aspect ratio. The whole panel ends up ~6 * this value
        wide. 600 keeps a 12 MP source under ~4 MB on disk.
    title:
        Optional caption stamped above the panel.
    """
    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    # Outline the ROI on the image cell so the operator can see exactly
    # what region was actually inspected.
    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_bgr, contours, -1, (0, 255, 0), 2)

    aligned = result.aligned
    if aligned.ndim == 2:
        aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    else:
        aligned_bgr = aligned.copy()

    diff = result.diff
    mask = result.anomaly_mask
    thresh = threshold_map if threshold_map is not None else result.threshold_map

    heatmap_bgr = _heatmap_overlay(aligned_bgr, diff)
    mask_bgr = _mask_overlay(aligned_bgr, mask, color=(255, 200, 0))
    conf_fg = _confidence_fg(diff, thresh, mask)
    conf_bg = _confidence_bg(diff, thresh, mask)
    overlay_bgr = draw_defects(aligned_bgr, result.defects)

    cells = [
        ("image", image_bgr),
        ("heatmap", heatmap_bgr),
        ("mask pred", mask_bgr),
        ("pred conf fg", conf_fg),
        ("pred conf bg", conf_bg),
        ("overlay", overlay_bgr),
    ]

    resized = [(_resize_to_width(img, max_cell_width), label) for label, img in cells]
    target_h = max(img.shape[0] for img, _ in resized)
    target_w = max(img.shape[1] for img, _ in resized)
    framed = [_pad_label(_pad_to(img, target_h, target_w), label) for img, label in resized]

    panel = np.hstack(framed)

    if title:
        bar = np.full((36, panel.shape[1], 3), 28, dtype=np.uint8)
        cv2.putText(bar, title, (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        panel = np.vstack([bar, panel])

    return panel


# ---------- internals --------------------------------------------------------


def _heatmap_overlay(image_bgr: np.ndarray, diff: np.ndarray) -> np.ndarray:
    d = np.clip(diff, 0, None).astype(np.float32)
    if d.max() > 1e-3:
        d = (d / d.max() * 255.0).astype(np.uint8)
    else:
        d = np.zeros_like(d, dtype=np.uint8)
    heat = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.55, heat, 0.45, 0)


def _mask_overlay(image_bgr: np.ndarray, mask: np.ndarray,
                  color: tuple[int, int, int] = (255, 200, 0)) -> np.ndarray:
    overlay = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr)
    color_layer[:] = color
    blended = cv2.addWeighted(overlay, 0.7, color_layer, 0.3, 0)
    out = np.where(mask[..., None] > 0, blended, overlay)
    # Outline the mask boundary so faint regions stay visible.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def _confidence_fg(diff: np.ndarray, threshold: np.ndarray,
                   mask: np.ndarray) -> np.ndarray:
    """Viridis-colored residual / threshold ratio, restricted to the mask."""
    ratio = diff / (threshold + 1e-3)
    ratio = np.clip(ratio, 0.0, 4.0) / 4.0          # [0, 1]
    g = (ratio * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(g, cv2.COLORMAP_VIRIDIS)
    blacked = np.zeros_like(colored)
    return np.where(mask[..., None] > 0, colored, blacked)


def _confidence_bg(diff: np.ndarray, threshold: np.ndarray,
                   mask: np.ndarray) -> np.ndarray:
    """Cool-toned 'safe-margin' map (1 - normalised residual) outside the mask."""
    ratio = diff / (threshold + 1e-3)
    safe = 1.0 - np.clip(ratio, 0.0, 1.0)
    g = (safe * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(g, cv2.COLORMAP_OCEAN)
    blacked = np.zeros_like(colored)
    return np.where(mask[..., None] > 0, blacked, colored)


def _resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def _pad_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    pad = np.full((target_h, target_w, 3), 18, dtype=np.uint8)
    pad[:h, :w] = img
    return pad


def _pad_label(img: np.ndarray, label: str) -> np.ndarray:
    bar = np.full((28, img.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(bar, label, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, img])
