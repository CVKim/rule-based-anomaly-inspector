"""Defect overlay rendering."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .inspector import DefectInfo, InspectionResult


def draw_defects(image: np.ndarray, defects: Iterable[DefectInfo],
                 color: tuple[int, int, int] = (0, 0, 255),
                 thickness: int = 2,
                 label: bool = True) -> np.ndarray:
    """Return a BGR copy of `image` with bounding boxes drawn for each defect."""
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for i, d in enumerate(defects, start=1):
        x, y, w, h = d.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        if label:
            txt = f"#{i} A={d.area} max={d.max_diff:.0f}"
            cv2.putText(vis, txt, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def heatmap(diff: np.ndarray) -> np.ndarray:
    """Return a JET-colored heatmap of the difference image."""
    d = diff.copy()
    d = np.clip(d, 0, None)
    if d.max() > 0:
        d = (d / d.max() * 255.0).astype(np.uint8)
    else:
        d = d.astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_JET)


def side_by_side(result: InspectionResult, master: np.ndarray) -> np.ndarray:
    """Compose a four-panel debug image: master | aligned | diff heatmap | defects."""
    master_bgr = cv2.cvtColor(master.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    aligned_bgr = cv2.cvtColor(result.aligned, cv2.COLOR_GRAY2BGR)
    diff_bgr = heatmap(result.diff)
    defect_bgr = draw_defects(result.aligned, result.defects)

    def _label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    panels = [
        _label(master_bgr, "master"),
        _label(aligned_bgr, "aligned"),
        _label(diff_bgr, "diff"),
        _label(defect_bgr, f"defects: {len(result.defects)}"),
    ]
    return np.hstack(panels)
