"""Six-cell debug visualisation for anomaly inspection results.

Layout (2 rows x 3 cols):

    +----------+----------+----------+
    |  origin  | heatmap  | mask pred|
    +----------+----------+----------+
    |    gt    | pred conf|  overlay |
    +----------+----------+----------+

The 2x3 grid keeps the aspect ratio readable for source images that are
themselves wider than tall (FOOSUNG side-views are 4096x2851; a flat
1x6 strip would be 6 x wider than each cell, hard to inspect at any
display size).

* ``origin``      — the aligned target (grayscale -> BGR), with the part-ROI
                    contour drawn in green when an ROI mask is supplied.
* ``heatmap``     — JET-coloured residual magnitude blended over the image.
* ``mask pred``   — image with the binary anomaly mask tinted in cyan
                    plus the mask outline.
* ``gt``          — image with GT defect polygons (green outlines) and any
                    rectangle-only GT drawn as bboxes; rendered side-by-side
                    with ``mask pred`` so a human eye can match TPs/FNs at
                    a glance.
* ``pred conf``   — combined "anomaly confidence" map. Inside the predicted
                    mask the pixels are coloured by ``residual / threshold``
                    (warm: orange/yellow = strong); outside the mask the
                    pixels are coloured by ``1 - ratio`` (cool: blue/cyan
                    = high confidence the pixel is normal).
* ``overlay``     — image with GT polygons (green) AND predicted mask
                    contours (magenta) AND per-defect bounding boxes from
                    ``draw_defects``. The "is the prediction overlapping
                    the truth?" cell.

The function accepts optional ``gt_polygons`` / ``gt_bboxes`` lists in the
panel-display coordinate frame so callers that work in inspect-space
(e.g. ``scripts/evaluate.py``) can pass them through after their own
``pred_scale`` adjustment. Both lists may be empty for OK images.

Cell heights are equalised; total width is roughly ``6 * max_cell_width``.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .inspector import InspectionResult
from .visualization import draw_defects


def make_panel(image: np.ndarray, result: InspectionResult,
               threshold_map: np.ndarray | None = None,
               max_cell_width: int = 600,
               title: str | None = None,
               roi_mask: np.ndarray | None = None,
               gt_polygons: Sequence[Sequence[tuple[float, float]]] | None = None,
               gt_bboxes: Sequence[tuple[float, float, float, float]] | None = None,
               crop_to_roi: bool = True,
               crop_pad: int = 50,
               ) -> np.ndarray:
    """Compose the six-cell debug panel as a single BGR uint8 image.

    Parameters
    ----------
    image:
        Grayscale or BGR image to use as the visual backdrop. The ``aligned``
        field of ``result`` is what was actually inspected; pass the same
        image you fed to ``inspect()`` so the panel shows the source the
        operator captured.
    result:
        ``InspectionResult`` from ``DynamicToleranceInspector``.
    threshold_map:
        Per-pixel threshold map for the confidence cell. Defaults to
        ``result.threshold_map``.
    max_cell_width:
        Each cell is shrunk to this width while preserving aspect ratio.
        The total panel ends up ~6 * this value wide. 600 keeps a 12 MP
        source under ~4 MB on disk.
    title:
        Optional caption stamped above the panel (filename, mode, metrics).
    roi_mask:
        Optional uint8 mask of the part region in inspect-space. Drawn as
        a green outline on the ``origin`` cell so the operator sees what
        was actually evaluated.
    gt_polygons:
        Optional list of polygon-vertex lists (in panel-display coordinates).
        Drawn as green outlines on the ``gt`` and ``overlay`` cells.
    gt_bboxes:
        Optional list of axis-aligned bboxes ``(x, y, w, h)`` (panel-display
        coordinates). Used as a fallback green outline when no polygon is
        available for a given GT.
    """
    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()

    aligned = result.aligned
    if aligned.ndim == 2:
        aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    else:
        aligned_bgr = aligned.copy()

    diff = result.diff
    mask = result.anomaly_mask
    thresh = threshold_map if threshold_map is not None else result.threshold_map

    # Optionally crop everything to the ROI bbox so the panel doesn't
    # waste 70-80% of pixels on dark background. Polygons / bboxes get
    # translated into the cropped frame. If the cropped region is
    # markedly taller than wide (FOOSUNG side-views are ~4.5x tall),
    # rotate 90° CCW so each cell becomes wide-and-short — the 2x3
    # grid then has a sensible overall aspect ratio.
    if crop_to_roi and roi_mask is not None:
        ys, xs = np.where(roi_mask > 0)
        if ys.size > 0:
            y0 = max(0, int(ys.min()) - crop_pad)
            y1 = min(roi_mask.shape[0], int(ys.max()) + crop_pad + 1)
            x0 = max(0, int(xs.min()) - crop_pad)
            x1 = min(roi_mask.shape[1], int(xs.max()) + crop_pad + 1)
            image_bgr = image_bgr[y0:y1, x0:x1]
            aligned_bgr = aligned_bgr[y0:y1, x0:x1]
            diff = diff[y0:y1, x0:x1]
            mask = mask[y0:y1, x0:x1]
            thresh = thresh[y0:y1, x0:x1]
            roi_mask = roi_mask[y0:y1, x0:x1]
            if gt_polygons:
                gt_polygons = [
                    [(x - x0, y - y0) for (x, y) in poly]
                    for poly in gt_polygons
                ]
            if gt_bboxes:
                gt_bboxes = [
                    (bx - x0, by - y0, bw, bh)
                    for (bx, by, bw, bh) in gt_bboxes
                ]

            # Rotate 90° CCW for tall parts so each cell ends up wider
            # than tall. After rotation: new_x = old_y, new_y = (W - old_x).
            crop_h, crop_w = image_bgr.shape[:2]
            if crop_h > crop_w * 1.5:
                image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                aligned_bgr = cv2.rotate(aligned_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                diff = cv2.rotate(diff, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                thresh = cv2.rotate(thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)
                roi_mask = cv2.rotate(roi_mask,
                                      cv2.ROTATE_90_COUNTERCLOCKWISE)
                W = crop_w
                if gt_polygons:
                    gt_polygons = [
                        [(y, W - x) for (x, y) in poly]
                        for poly in gt_polygons
                    ]
                if gt_bboxes:
                    gt_bboxes = [
                        (by, W - bx - bw, bh, bw)
                        for (bx, by, bw, bh) in gt_bboxes
                    ]

    # Build each cell ----------------------------------------------------

    origin_bgr = aligned_bgr.copy()
    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(origin_bgr, contours, -1, (0, 220, 0), 2)

    heatmap_bgr = _heatmap_overlay(aligned_bgr, diff)
    mask_bgr = _mask_overlay(aligned_bgr, mask, color=(255, 200, 0))   # cyan-ish
    gt_bgr = _gt_overlay(aligned_bgr, gt_polygons, gt_bboxes)
    conf_bgr = _confidence_combined(aligned_bgr, diff, thresh, mask)
    overlay_bgr = _composite_overlay(aligned_bgr, mask, result, gt_polygons,
                                     gt_bboxes)

    cells = [
        ("origin", origin_bgr),
        ("heatmap", heatmap_bgr),
        ("mask pred", mask_bgr),
        ("gt", gt_bgr),
        ("pred conf", conf_bgr),
        ("overlay", overlay_bgr),
    ]

    resized = [(_resize_to_width(img, max_cell_width), label)
               for label, img in cells]
    target_h = max(img.shape[0] for img, _ in resized)
    target_w = max(img.shape[1] for img, _ in resized)
    framed = [_pad_label(_pad_to(img, target_h, target_w), label)
              for img, label in resized]

    # 2 rows x 3 cols. Much more readable for tall source images
    # than a flat 1x6 strip — each row is at most ``3 * max_cell_width``
    # wide vs ``6 *`` for the flat layout.
    row1 = np.hstack(framed[:3])
    row2 = np.hstack(framed[3:])
    panel = np.vstack([row1, row2])

    if title:
        bar = np.full((36, panel.shape[1], 3), 28, dtype=np.uint8)
        cv2.putText(bar, title, (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2,
                    cv2.LINE_AA)
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
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def _gt_overlay(image_bgr: np.ndarray,
                gt_polygons: Sequence[Sequence[tuple[float, float]]] | None,
                gt_bboxes: Sequence[tuple[float, float, float, float]] | None,
                ) -> np.ndarray:
    """Draw GT polygons (or bboxes) as green outlines."""
    out = image_bgr.copy()
    drew_any = False
    if gt_polygons:
        for poly in gt_polygons:
            if poly and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], isClosed=True,
                              color=(0, 255, 0), thickness=2)
                drew_any = True
    if gt_bboxes:
        for i, bb in enumerate(gt_bboxes):
            poly = (gt_polygons[i] if gt_polygons and i < len(gt_polygons)
                    else None)
            if poly and len(poly) >= 3:
                continue   # already drawn as polygon
            x, y, w, h = bb
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            drew_any = True
    if not drew_any:
        # OK image — just stamp "no GT" so the cell isn't visually empty.
        cv2.putText(out, "no GT", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2,
                    cv2.LINE_AA)
    return out


def _confidence_combined(image_bgr: np.ndarray, diff: np.ndarray,
                         threshold: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """One cell that paints fg-confidence (warm INFERNO) inside the mask
    and bg-confidence (cool OCEAN) outside, blended over the source so the
    anatomy stays visible.

    fg ratio = clip(diff / threshold, 0..4) / 4   (warm; brighter = more confident defect)
    bg ratio = 1 - clip(diff / threshold, 0..1)   (cool; brighter = more confident OK)
    """
    ratio = diff / (threshold + 1e-3)
    fg_norm = np.clip(ratio, 0.0, 4.0) / 4.0
    bg_norm = 1.0 - np.clip(ratio, 0.0, 1.0)

    fg_color = cv2.applyColorMap((fg_norm * 255.0).astype(np.uint8),
                                 cv2.COLORMAP_INFERNO)
    bg_color = cv2.applyColorMap((bg_norm * 255.0).astype(np.uint8),
                                 cv2.COLORMAP_OCEAN)

    inside = mask[..., None] > 0
    coloured = np.where(inside, fg_color, bg_color)
    return cv2.addWeighted(image_bgr, 0.45, coloured, 0.55, 0)


def _composite_overlay(image_bgr: np.ndarray, mask: np.ndarray,
                       result: InspectionResult,
                       gt_polygons: Sequence[Sequence[tuple[float, float]]] | None,
                       gt_bboxes: Sequence[tuple[float, float, float, float]] | None,
                       ) -> np.ndarray:
    """Origin + GT polygons (green) + predicted mask contour (magenta) +
    per-defect bbox (category-coloured). The single 'is the prediction
    on the truth?' cell."""
    out = image_bgr.copy()

    # Predicted mask boundary in magenta.
    pred_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, pred_contours, -1, (255, 0, 255), 2)

    # GT polygons in green.
    if gt_polygons:
        for poly in gt_polygons:
            if poly and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], isClosed=True,
                              color=(0, 255, 0), thickness=2)
    if gt_bboxes:
        for i, bb in enumerate(gt_bboxes):
            poly = (gt_polygons[i] if gt_polygons and i < len(gt_polygons)
                    else None)
            if poly and len(poly) >= 3:
                continue
            x, y, w, h = bb
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Per-defect bbox via the existing draw_defects.
    out = draw_defects(out, result.defects)
    return out


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
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])
