"""GT-driven evaluation harness for anomaly predictions.

Consumes labelme-style ground-truth JSON (rectangles only, multi-label OK)
and the ``DefectInfo`` list produced by ``DynamicToleranceInspector`` and
emits per-image and aggregate metrics.

Critical convention — **GT is treated as incomplete**:

* TP, FN are computed against the labelled GT bboxes.
* FPs are split into two buckets:
    - ``hard FP``: prediction is outside the part ROI **or** is far from any
      GT bbox. These are the only FPs that count toward precision.
    - ``soft FP``: prediction is inside the ROI and reasonably close to a
      labelled GT but did not pass the IoU threshold. These are surfaced
      in the report so the operator can decide whether they're unlabelled
      true positives, but they're excluded from the precision denominator.

This avoids penalising the inspector for finding defects the human labeller
did not annotate, which the user explicitly cautioned about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------- GT loading ------------------------------------------------------


@dataclass(frozen=True)
class GtBox:
    """A single labelled defect in image-pixel coordinates.

    Always carries an axis-aligned ``bbox`` for cheap IoU matching. If the
    underlying labelme shape was a polygon, the original vertex list is
    preserved in ``polygon`` so callers that want pixel-accurate IoU /
    dice / coverage can rasterise it on demand.
    """
    label: str
    bbox: tuple[float, float, float, float]   # x, y, w, h (axis-aligned)
    polygon: tuple[tuple[float, float], ...] = ()   # empty -> rectangle source

    @property
    def cx(self) -> float:
        return self.bbox[0] + self.bbox[2] / 2.0

    @property
    def cy(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2.0

    @property
    def is_polygon(self) -> bool:
        return len(self.polygon) >= 3

    def rasterise(self, height: int, width: int) -> "np.ndarray":
        """Return a uint8 mask of this defect at (height, width).

        Both polygon and rectangle paths use OpenCV draw functions
        (``fillPoly`` / ``rectangle`` with ``thickness=-1``) so the
        boundary-pixel convention matches the predicted-bbox rasterisation
        in ``pixel_metrics`` — otherwise an off-by-one rim shows up at
        every defect edge and inflates the union.
        """
        import cv2
        mask = np.zeros((int(height), int(width)), dtype=np.uint8)
        if self.is_polygon:
            pts = np.array(self.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        else:
            x, y, w, h = self.bbox
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        return mask


@dataclass(frozen=True)
class GtImage:
    """One image's worth of ground truth."""
    filename: str
    image_width: int
    image_height: int
    boxes: tuple[GtBox, ...]

    @property
    def is_ng(self) -> bool:
        return len(self.boxes) > 0


def load_labelme_json(path: str | Path) -> GtImage:
    """Parse a single labelme JSON. Tolerates files with no shapes (all-OK).

    Supports both ``rectangle`` (two corner points) and ``polygon`` (>=3
    vertex) shape types. Polygons get an axis-aligned bbox derived from
    their min/max extents AND the original vertex list preserved in
    ``GtBox.polygon`` for downstream pixel-accurate IoU.
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes: list[GtBox] = []
    for s in data.get("shapes", []):
        label = s.get("label", "") or ""
        if not label:
            continue
        kind = s.get("shape_type")
        pts_raw = s.get("points") or []
        if kind == "rectangle" and len(pts_raw) == 2:
            (x1, y1), (x2, y2) = pts_raw
            x = float(min(x1, x2))
            y = float(min(y1, y2))
            w = float(abs(x2 - x1))
            h = float(abs(y2 - y1))
            boxes.append(GtBox(label=label, bbox=(x, y, w, h)))
        elif kind == "polygon" and len(pts_raw) >= 3:
            arr = np.array(pts_raw, dtype=np.float32)
            x_min = float(arr[:, 0].min())
            y_min = float(arr[:, 1].min())
            x_max = float(arr[:, 0].max())
            y_max = float(arr[:, 1].max())
            polygon = tuple((float(px), float(py)) for px, py in pts_raw)
            boxes.append(GtBox(
                label=label,
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                polygon=polygon,
            ))
        # Other shape_types (line, circle, point) are not currently used
        # by this dataset; ignore quietly.

    return GtImage(
        filename=p.stem + ".bmp",            # JSON stem == BMP stem in our data
        image_width=int(data.get("imageWidth", 0)),
        image_height=int(data.get("imageHeight", 0)),
        boxes=tuple(boxes),
    )


def load_gt_folder(folder: str | Path) -> dict[str, GtImage]:
    """Load every JSON in ``folder`` keyed by filename (e.g. ``"#1.bmp"``)."""
    out: dict[str, GtImage] = {}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() == ".json":
            gt = load_labelme_json(p)
            out[gt.filename] = gt
    return out


# ---------- IoU + matching --------------------------------------------------


def bbox_iou(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    """Standard axis-aligned IoU on (x, y, w, h) rectangles."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def bbox_centre_distance(a: tuple[float, float, float, float],
                         b: tuple[float, float, float, float]) -> float:
    acx = a[0] + a[2] / 2.0
    acy = a[1] + a[3] / 2.0
    bcx = b[0] + b[2] / 2.0
    bcy = b[1] + b[3] / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


# ---------- per-image evaluation -------------------------------------------


@dataclass
class PerImageEval:
    filename: str
    n_gt: int
    tp: int
    fn: int
    hard_fp: int
    soft_fp: int
    matched_pairs: list[tuple[int, int, float]]   # (pred_idx, gt_idx, iou)
    fn_indices: list[int]                          # GT indices not matched
    fp_pred_indices: list[int]                     # pred indices not matched (any FP)
    soft_fp_pred_indices: list[int]                # subset of fp that are "soft"

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt > 0 else float("nan")

    @property
    def precision(self) -> float:
        denom = self.tp + self.hard_fp
        return self.tp / denom if denom > 0 else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (p == p) or not (r == r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)


def evaluate_image(pred_boxes: list[tuple[float, float, float, float]],
                   gt: GtImage,
                   iou_threshold: float = 0.1,
                   soft_fp_max_centre_distance: float = 100.0,
                   roi_mask: np.ndarray | None = None,
                   pred_scale: float = 1.0,
                   use_polygon_iou: bool = False) -> PerImageEval:
    """Match predictions to GT; bucket leftovers as hard / soft FPs.

    Parameters
    ----------
    pred_boxes:
        Prediction bboxes in *prediction-space* coordinates (e.g. the
        downsampled inspection resolution).
    gt:
        Loaded GT for this image, in original full-res coordinates.
    iou_threshold:
        Minimum IoU for a TP match. Cracks are small; 0.1 is intentionally
        permissive (a 50x30 GT bbox + 60x40 pred bbox at the right place
        easily clears IoU=0.1).
    soft_fp_max_centre_distance:
        FP whose centroid is within this distance (full-res pixels) of any
        labelled GT bbox is bucketed as ``soft`` (likely unlabelled defect).
    roi_mask:
        Optional uint8 mask of the part region in *prediction-space*. FPs
        outside this mask go straight to ``hard`` regardless of distance.
    pred_scale:
        Multiplier to convert prediction-space pixels to full-res pixels
        (e.g. 4096/1600 = 2.56 when inspecting at 1600 wide).
    """
    # Lift predictions to full-res for fair matching against GT.
    pred_full = [(x * pred_scale, y * pred_scale,
                  w * pred_scale, h * pred_scale)
                 for (x, y, w, h) in pred_boxes]

    # Greedy IoU matching: for each GT, take the highest-IoU unmatched pred.
    n_pred = len(pred_full)
    n_gt = len(gt.boxes)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_pairs: list[tuple[int, int, float]] = []

    if n_pred and n_gt:
        ious = np.zeros((n_gt, n_pred), dtype=np.float32)
        for gi, g in enumerate(gt.boxes):
            for pi, p in enumerate(pred_full):
                if use_polygon_iou and g.is_polygon:
                    ious[gi, pi] = polygon_iou(g, p,
                                                gt.image_height, gt.image_width)
                else:
                    ious[gi, pi] = bbox_iou(g.bbox, p)
        # Iterate in descending IoU order
        for _ in range(min(n_gt, n_pred)):
            gi, pi = np.unravel_index(int(ious.argmax()), ious.shape)
            best = float(ious[gi, pi])
            if best < iou_threshold:
                break
            if gi in matched_gt or pi in matched_pred:
                ious[gi, pi] = -1
                continue
            matched_gt.add(int(gi))
            matched_pred.add(int(pi))
            matched_pairs.append((int(pi), int(gi), best))
            ious[gi, :] = -1
            ious[:, pi] = -1

    tp = len(matched_pairs)
    fn_indices = sorted(set(range(n_gt)) - matched_gt)
    fp_pred_indices = sorted(set(range(n_pred)) - matched_pred)

    soft_fp_pred_indices: list[int] = []
    hard_fp = 0
    soft_fp = 0
    for pi in fp_pred_indices:
        is_soft = False
        # ROI test (in pred-space): if ROI exists and the pred centroid is
        # outside it, force hard.
        in_roi = True
        if roi_mask is not None:
            px, py, pw, ph = pred_boxes[pi]
            cx = int(round(px + pw / 2.0))
            cy = int(round(py + ph / 2.0))
            cy = max(0, min(roi_mask.shape[0] - 1, cy))
            cx = max(0, min(roi_mask.shape[1] - 1, cx))
            in_roi = bool(roi_mask[cy, cx] > 0)

        if in_roi and gt.boxes:
            d_min = min(
                bbox_centre_distance(pred_full[pi], g.bbox)
                for g in gt.boxes
            )
            if d_min <= soft_fp_max_centre_distance:
                is_soft = True

        if is_soft:
            soft_fp += 1
            soft_fp_pred_indices.append(pi)
        else:
            hard_fp += 1

    return PerImageEval(
        filename=gt.filename,
        n_gt=n_gt, tp=tp, fn=len(fn_indices),
        hard_fp=hard_fp, soft_fp=soft_fp,
        matched_pairs=matched_pairs,
        fn_indices=fn_indices,
        fp_pred_indices=fp_pred_indices,
        soft_fp_pred_indices=soft_fp_pred_indices,
    )


# ---------- aggregate report ------------------------------------------------


@dataclass
class ModeReport:
    mode: str
    per_image: list[PerImageEval] = field(default_factory=list)
    iou_threshold: float = 0.1

    # ----- bbox-level (counted only on hard FPs) -----

    @property
    def total_tp(self) -> int:    return sum(e.tp for e in self.per_image)
    @property
    def total_fn(self) -> int:    return sum(e.fn for e in self.per_image)
    @property
    def total_hard_fp(self) -> int: return sum(e.hard_fp for e in self.per_image)
    @property
    def total_soft_fp(self) -> int: return sum(e.soft_fp for e in self.per_image)
    @property
    def total_gt(self) -> int:    return sum(e.n_gt for e in self.per_image)

    @property
    def precision(self) -> float:
        denom = self.total_tp + self.total_hard_fp
        return self.total_tp / denom if denom > 0 else float("nan")

    @property
    def recall(self) -> float:
        return self.total_tp / self.total_gt if self.total_gt > 0 else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (p == p) or not (r == r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def hard_fp_per_image(self) -> float:
        n = len(self.per_image) or 1
        return self.total_hard_fp / n

    # ----- image-level (NG / OK classification) -----

    @property
    def image_classification(self) -> dict[str, int]:
        """Per-image confusion: NG predicted iff at least one TP or hard FP."""
        tp_img = fp_img = tn_img = fn_img = 0
        for e in self.per_image:
            gt_ng = e.n_gt > 0
            pred_ng = (e.tp + e.hard_fp) > 0
            if gt_ng and pred_ng:    tp_img += 1
            elif gt_ng and not pred_ng: fn_img += 1
            elif not gt_ng and pred_ng: fp_img += 1
            else:                       tn_img += 1
        return {"TP": tp_img, "FP": fp_img, "TN": tn_img, "FN": fn_img}

    def to_dict(self) -> dict:
        ic = self.image_classification
        return {
            "mode": self.mode,
            "iou_threshold": self.iou_threshold,
            "n_images": len(self.per_image),
            "bbox": {
                "total_gt": self.total_gt,
                "tp": self.total_tp,
                "fn": self.total_fn,
                "hard_fp": self.total_hard_fp,
                "soft_fp": self.total_soft_fp,
                "precision": _safe(self.precision),
                "recall": _safe(self.recall),
                "f1": _safe(self.f1),
                "hard_fp_per_image": self.hard_fp_per_image,
            },
            "image": ic,
            "per_image": [
                {
                    "filename": e.filename,
                    "n_gt": e.n_gt,
                    "tp": e.tp,
                    "fn": e.fn,
                    "hard_fp": e.hard_fp,
                    "soft_fp": e.soft_fp,
                    "recall": _safe(e.recall),
                    "precision": _safe(e.precision),
                    "f1": _safe(e.f1),
                }
                for e in self.per_image
            ],
        }


def _safe(x: float) -> float | None:
    """Convert NaN -> None for clean JSON output."""
    return None if x != x else float(x)


# ---------- mode complementarity -------------------------------------------


def mode_complementarity(reports: dict[str, ModeReport]) -> dict[str, dict[str, int]]:
    """How many GT defects each mode catches that another misses.

    Returns a nested dict ``out[mode_a][mode_b] = number of GT defects mode_a
    found that mode_b missed``. Diagonal = total caught by that mode.
    """
    # Map (image, gt_idx) -> set of modes that caught it
    caught: dict[tuple[str, int], set[str]] = {}
    for mode, rep in reports.items():
        for e in rep.per_image:
            matched = {gi for (_, gi, _) in e.matched_pairs}
            for gi in matched:
                caught.setdefault((e.filename, gi), set()).add(mode)
    modes = list(reports)
    matrix: dict[str, dict[str, int]] = {a: {b: 0 for b in modes} for a in modes}
    for who in caught.values():
        for a in modes:
            for b in modes:
                if a in who and (a == b or b not in who):
                    matrix[a][b] += 1
    return matrix


def boxes_from_defects(defects: Iterable) -> list[tuple[float, float, float, float]]:
    """Convenience: extract ``bbox`` tuples from a sequence of ``DefectInfo``."""
    out: list[tuple[float, float, float, float]] = []
    for d in defects:
        x, y, w, h = d.bbox
        out.append((float(x), float(y), float(w), float(h)))
    return out


# ---------- pixel-level (polygon-aware) metrics ----------------------------


@dataclass
class PixelMetrics:
    """Pixel-level summary across all GT defect masks in one image.

    GT masks come from rasterised polygons (or rectangles when only
    ``rectangle`` GT exists). Predictions are rasterised from their bbox.
    Computed on the union of all defects per side, so a single image
    yields one (intersection, GT-area, pred-area) triple.
    """
    image_w: int
    image_h: int
    intersection: int
    gt_area: int
    pred_area: int

    @property
    def iou(self) -> float:
        union = self.gt_area + self.pred_area - self.intersection
        return self.intersection / union if union > 0 else float("nan")

    @property
    def dice(self) -> float:
        denom = self.gt_area + self.pred_area
        return 2.0 * self.intersection / denom if denom > 0 else float("nan")

    @property
    def pixel_recall(self) -> float:
        return self.intersection / self.gt_area if self.gt_area > 0 else float("nan")

    @property
    def pixel_precision(self) -> float:
        return self.intersection / self.pred_area if self.pred_area > 0 else float("nan")


def pixel_metrics(pred_boxes_full: list[tuple[float, float, float, float]],
                  gt: GtImage) -> PixelMetrics:
    """Rasterise GT (polygon when available, rectangle otherwise) and
    predicted bboxes at full resolution, then compute pixel agreement.

    ``pred_boxes_full`` must already be in full-resolution coordinates —
    the caller is responsible for the ``pred_scale`` lift.
    """
    import cv2
    h = max(int(gt.image_height), 1)
    w = max(int(gt.image_width), 1)
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for g in gt.boxes:
        gt_mask = np.maximum(gt_mask, g.rasterise(h, w))

    pred_mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, bw, bh) in pred_boxes_full:
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(w - 1, int(round(x + bw)))
        y2 = min(h - 1, int(round(y + bh)))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(pred_mask, (x1, y1), (x2, y2), 255, thickness=-1)

    inter = int(((gt_mask > 0) & (pred_mask > 0)).sum())
    return PixelMetrics(
        image_w=w, image_h=h,
        intersection=inter,
        gt_area=int((gt_mask > 0).sum()),
        pred_area=int((pred_mask > 0).sum()),
    )


def polygon_iou(gt_box: GtBox, pred_bbox: tuple[float, float, float, float],
                image_h: int, image_w: int) -> float:
    """True polygon IoU between one GT (polygon or rectangle) and a
    predicted axis-aligned bbox. Uses pixel rasterisation, so it's
    O(image area) — call sparingly (per-(GT, pred) pair).
    """
    import cv2
    h = max(int(image_h), 1)
    w = max(int(image_w), 1)
    gt_mask = gt_box.rasterise(h, w)
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = pred_bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(w - 1, int(round(x + bw)))
    y2 = min(h - 1, int(round(y + bh)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    cv2.rectangle(pred_mask, (x1, y1), (x2, y2), 255, thickness=-1)
    inter = int(((gt_mask > 0) & (pred_mask > 0)).sum())
    union = int(((gt_mask > 0) | (pred_mask > 0)).sum())
    return inter / union if union > 0 else 0.0
