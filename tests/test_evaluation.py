"""Tests for the GT-driven evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anomaly_inspector.evaluation import (
    GtBox, GtImage, ModeReport, bbox_iou, evaluate_image,
    load_labelme_json, mode_complementarity, pixel_metrics, polygon_iou,
)


# ---------- labelme parsing -------------------------------------------------


def test_load_labelme_parses_rectangles(tmp_path: Path):
    j = {
        "imageWidth": 4096, "imageHeight": 2851,
        "shapes": [
            {"shape_type": "rectangle", "label": "CRACK",
             "points": [[10.0, 20.0], [50.0, 60.0]]},
            {"shape_type": "rectangle", "label": "CRACK",
             "points": [[100, 200], [80, 180]]},   # reversed corners
        ],
    }
    p = tmp_path / "#7.json"
    p.write_text(json.dumps(j), encoding="utf-8")
    gt = load_labelme_json(p)
    assert gt.filename == "#7.bmp"
    assert gt.image_width == 4096 and gt.image_height == 2851
    assert len(gt.boxes) == 2
    assert gt.boxes[0].bbox == (10.0, 20.0, 40.0, 40.0)
    # Reversed corners should normalise to (x, y, w, h) with positive w/h
    assert gt.boxes[1].bbox == (80.0, 180.0, 20.0, 20.0)
    assert gt.is_ng


def test_load_labelme_skips_empty_labels_and_unknown_shapes(tmp_path: Path):
    """Polygons are now supported (see test_load_labelme_parses_polygon...);
    this test guards the *other* drop cases: empty label string and
    shape_type values we don't handle (line, circle, point)."""
    j = {
        "imageWidth": 100, "imageHeight": 100,
        "shapes": [
            {"shape_type": "rectangle", "label": "",
             "points": [[10, 10], [20, 20]]},                      # empty label
            {"shape_type": "line", "label": "CRACK",
             "points": [[0, 0], [10, 10]]},                        # unsupported
            {"shape_type": "point", "label": "CRACK",
             "points": [[5, 5]]},                                  # unsupported
        ],
    }
    p = tmp_path / "x.json"
    p.write_text(json.dumps(j), encoding="utf-8")
    gt = load_labelme_json(p)
    assert len(gt.boxes) == 0
    assert not gt.is_ng


def test_load_labelme_handles_empty_shapes(tmp_path: Path):
    j = {"imageWidth": 100, "imageHeight": 100, "shapes": []}
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(j), encoding="utf-8")
    gt = load_labelme_json(p)
    assert len(gt.boxes) == 0
    assert not gt.is_ng


# ---------- IoU -------------------------------------------------------------


def test_iou_perfect_overlap():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert bbox_iou((0, 0, 5, 5), (10, 10, 5, 5)) == 0.0


def test_iou_partial_overlap():
    # 10x10 vs 10x10 offset by 5 in x: intersection = 5x10 = 50, union = 150
    assert bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50/150)


# ---------- evaluate_image --------------------------------------------------


def _gt(*boxes) -> GtImage:
    return GtImage(filename="t.bmp", image_width=200, image_height=200,
                   boxes=tuple(GtBox("CRACK", b) for b in boxes))


def test_evaluate_image_perfect_match():
    gt = _gt((10.0, 10.0, 20.0, 20.0), (60.0, 60.0, 30.0, 20.0))
    preds = [(10.0, 10.0, 20.0, 20.0), (60.0, 60.0, 30.0, 20.0)]
    ev = evaluate_image(preds, gt, iou_threshold=0.1)
    assert ev.tp == 2
    assert ev.fn == 0
    assert ev.hard_fp == 0


def test_evaluate_image_misses_one_gt():
    gt = _gt((10.0, 10.0, 20.0, 20.0), (60.0, 60.0, 20.0, 20.0))
    preds = [(10.0, 10.0, 20.0, 20.0)]   # only first GT covered
    ev = evaluate_image(preds, gt, iou_threshold=0.1)
    assert ev.tp == 1
    assert ev.fn == 1
    assert ev.hard_fp == 0
    assert ev.fn_indices == [1]


def test_evaluate_image_buckets_far_pred_as_hard_fp():
    gt = _gt((10.0, 10.0, 20.0, 20.0))
    preds = [(150.0, 150.0, 10.0, 10.0)]   # far from GT
    ev = evaluate_image(preds, gt, iou_threshold=0.1,
                        soft_fp_max_centre_distance=20.0)
    assert ev.tp == 0
    assert ev.hard_fp == 1
    assert ev.soft_fp == 0


def test_evaluate_image_buckets_near_pred_as_soft_fp():
    gt = _gt((100.0, 100.0, 20.0, 20.0))
    preds = [(125.0, 100.0, 10.0, 10.0)]   # adjacent but no IoU overlap
    # IoU = 0 (adjacent, not overlapping); centre distance ~ 35 px.
    ev = evaluate_image(preds, gt, iou_threshold=0.1,
                        soft_fp_max_centre_distance=80.0)
    assert ev.tp == 0
    assert ev.hard_fp == 0
    assert ev.soft_fp == 1


def test_evaluate_image_outside_roi_is_always_hard():
    gt = _gt((100.0, 100.0, 20.0, 20.0))
    preds = [(110.0, 110.0, 5.0, 5.0)]    # close to GT but inside an ROI mask
                                          # that excludes that area
    roi = np.zeros((200, 200), dtype=np.uint8)
    roi[80:140, 80:140] = 0   # not in roi
    ev = evaluate_image(preds, gt, iou_threshold=0.5,    # forces no TP match
                        soft_fp_max_centre_distance=200.0,
                        roi_mask=roi)
    # IoU = 25 / (25 + 400 - 25) = 0.0625 < 0.5 -> not TP. Outside ROI -> hard.
    assert ev.tp == 0
    assert ev.hard_fp == 1
    assert ev.soft_fp == 0


def test_evaluate_image_pred_scale_lifts_to_full_res():
    gt = _gt((100.0, 100.0, 40.0, 40.0))
    # Prediction in pred-space (same coords but fed at half-res, so x2):
    preds = [(50.0, 50.0, 20.0, 20.0)]
    ev = evaluate_image(preds, gt, iou_threshold=0.5, pred_scale=2.0)
    assert ev.tp == 1
    assert ev.fn == 0


# ---------- ModeReport aggregate -------------------------------------------


def test_mode_report_metrics():
    rep = ModeReport(mode="absdiff")
    # Image 1: 2 GT, 2 TP, 0 FP, 0 FN
    rep.per_image.append(evaluate_image(
        [(10.0, 10.0, 20.0, 20.0), (60.0, 60.0, 20.0, 20.0)],
        _gt((10.0, 10.0, 20.0, 20.0), (60.0, 60.0, 20.0, 20.0)),
    ))
    # Image 2: 1 GT, 0 TP, 0 FP, 1 FN, 0 hard FP (no preds)
    rep.per_image.append(evaluate_image([], _gt((10.0, 10.0, 20.0, 20.0))))
    # Image 3: 0 GT, 1 hard FP
    rep.per_image.append(evaluate_image(
        [(150.0, 150.0, 10.0, 10.0)], _gt(),
    ))
    assert rep.total_tp == 2
    assert rep.total_fn == 1
    assert rep.total_hard_fp == 1
    assert rep.recall == pytest.approx(2 / 3)
    assert rep.precision == pytest.approx(2 / 3)
    assert rep.f1 == pytest.approx(2 / 3)
    ic = rep.image_classification
    assert ic == {"TP": 1, "FP": 1, "TN": 0, "FN": 1}


def test_mode_complementarity_counts_unique_catches():
    rep_a = ModeReport(mode="a")
    rep_b = ModeReport(mode="b")
    g = _gt((10.0, 10.0, 20.0, 20.0), (60.0, 60.0, 20.0, 20.0))
    rep_a.per_image.append(evaluate_image(
        [(10.0, 10.0, 20.0, 20.0)], g))                  # a catches GT 0
    rep_b.per_image.append(evaluate_image(
        [(60.0, 60.0, 20.0, 20.0)], g))                  # b catches GT 1
    matrix = mode_complementarity({"a": rep_a, "b": rep_b})
    # diagonal = total caught by that mode
    assert matrix["a"]["a"] == 1
    assert matrix["b"]["b"] == 1
    # off-diagonal = caught by row, missed by column
    assert matrix["a"]["b"] == 1
    assert matrix["b"]["a"] == 1


# ---------- polygon parsing ------------------------------------------------


def test_load_labelme_parses_polygon_with_bbox(tmp_path: Path):
    """Polygon GT should preserve the vertex list AND surface a tight
    axis-aligned bbox derived from the vertex extents."""
    pts = [[100.0, 200.0], [300.0, 220.0], [310.0, 280.0],
           [150.0, 290.0], [110.0, 250.0]]
    j = {
        "imageWidth": 4096, "imageHeight": 2851,
        "shapes": [
            {"shape_type": "polygon", "label": "CRACK", "points": pts},
        ],
    }
    p = tmp_path / "#9.json"
    p.write_text(json.dumps(j), encoding="utf-8")
    gt = load_labelme_json(p)
    assert len(gt.boxes) == 1
    g = gt.boxes[0]
    assert g.is_polygon
    assert g.bbox == (100.0, 200.0, 210.0, 90.0)   # min/max extents
    assert g.polygon == tuple((float(x), float(y)) for x, y in pts)


def test_load_labelme_skips_polygons_with_two_vertices(tmp_path: Path):
    """A polygon needs at least 3 vertices; degenerate cases are dropped."""
    j = {
        "imageWidth": 100, "imageHeight": 100,
        "shapes": [
            {"shape_type": "polygon", "label": "CRACK",
             "points": [[10, 10], [50, 50]]},   # only 2 vertices
        ],
    }
    p = tmp_path / "x.json"
    p.write_text(json.dumps(j), encoding="utf-8")
    gt = load_labelme_json(p)
    assert len(gt.boxes) == 0


def test_load_labelme_handles_mixed_rect_and_polygon(tmp_path: Path):
    j = {
        "imageWidth": 100, "imageHeight": 100,
        "shapes": [
            {"shape_type": "rectangle", "label": "CRACK",
             "points": [[10, 10], [30, 30]]},
            {"shape_type": "polygon", "label": "CRACK",
             "points": [[40, 40], [80, 50], [60, 90]]},
        ],
    }
    p = tmp_path / "mixed.json"
    p.write_text(json.dumps(j), encoding="utf-8")
    gt = load_labelme_json(p)
    assert len(gt.boxes) == 2
    assert not gt.boxes[0].is_polygon
    assert gt.boxes[1].is_polygon


# ---------- polygon IoU + pixel metrics ------------------------------------


def test_gtbox_rasterise_polygon():
    g = GtBox(label="CRACK", bbox=(0, 0, 100, 100),
              polygon=((10, 10), (90, 10), (90, 90), (10, 90)))
    mask = g.rasterise(120, 120)
    assert mask.shape == (120, 120)
    assert mask[50, 50] == 255
    assert mask[5, 5] == 0


def test_polygon_iou_matches_bbox_iou_for_rectangles():
    """A square polygon and an equal-area rectangle should give similar IoU
    against the same predicted bbox."""
    g = GtBox(label="CRACK", bbox=(0.0, 0.0, 50.0, 50.0))   # rect
    pred = (25.0, 25.0, 50.0, 50.0)
    bbox = bbox_iou(g.bbox, pred)
    poly_g = GtBox(label="CRACK", bbox=(0.0, 0.0, 50.0, 50.0),
                   polygon=((0, 0), (50, 0), (50, 50), (0, 50)))
    poly = polygon_iou(poly_g, pred, image_h=120, image_w=120)
    assert abs(bbox - poly) < 0.05


def test_polygon_iou_lower_than_bbox_iou_for_thin_diagonal():
    """A thin diagonal crack labelled as polygon vs its loose axis-aligned
    bbox: polygon IoU against the same loose pred bbox should be much
    smaller, because the rect overestimates the defect area."""
    poly = ((10, 10), (90, 90), (95, 85), (15, 5))    # thin diagonal
    g_poly = GtBox(label="CRACK",
                   bbox=(10.0, 5.0, 85.0, 85.0),       # bbox of the diagonal
                   polygon=poly)
    g_rect = GtBox(label="CRACK", bbox=(10.0, 5.0, 85.0, 85.0))
    pred = (10.0, 5.0, 85.0, 85.0)                     # exact bbox match
    bbox = bbox_iou(g_rect.bbox, pred)
    poly = polygon_iou(g_poly, pred, image_h=120, image_w=120)
    assert bbox > 0.99
    assert poly < 0.5      # diagonal fills <50% of its bbox


def test_pixel_metrics_perfect_overlap():
    """The polygon and the predicted bbox describe the same axis-aligned
    region (10..30 inclusive). cv2.fillPoly + cv2.rectangle both fill the
    boundary, so both rasterised masks contain 21*21 = 441 pixels and
    intersect perfectly => IoU=1, Dice=1."""
    g = GtImage(filename="t.bmp", image_width=100, image_height=100,
                boxes=(GtBox("CRACK", (10, 10, 20, 20),
                              polygon=((10, 10), (30, 10), (30, 30), (10, 30))),))
    pm = pixel_metrics([(10.0, 10.0, 20.0, 20.0)], g)
    assert pm.gt_area == pm.pred_area
    assert pm.intersection == pm.gt_area
    assert pm.iou == 1.0
    assert pm.dice == 1.0


def test_pixel_metrics_no_overlap():
    g = GtImage(filename="t.bmp", image_width=100, image_height=100,
                boxes=(GtBox("CRACK", (10, 10, 20, 20),
                              polygon=((10, 10), (30, 10), (30, 30), (10, 30))),))
    pm = pixel_metrics([(60.0, 60.0, 20.0, 20.0)], g)
    assert pm.intersection == 0
    assert pm.iou == 0.0


def test_evaluate_image_use_polygon_iou_changes_match():
    """A loose prediction wrapping a thin diagonal polygon should clear the
    IoU bar with bbox IoU but fail under polygon IoU at the same threshold."""
    poly_g = GtBox(label="CRACK",
                   bbox=(10.0, 5.0, 85.0, 85.0),
                   polygon=((10, 10), (90, 90), (95, 85), (15, 5)))
    gt = GtImage(filename="t.bmp", image_width=120, image_height=120,
                 boxes=(poly_g,))
    pred = [(10.0, 5.0, 85.0, 85.0)]   # exact bbox match
    ev_bbox = evaluate_image(pred, gt, iou_threshold=0.6,
                             use_polygon_iou=False)
    ev_poly = evaluate_image(pred, gt, iou_threshold=0.6,
                             use_polygon_iou=True)
    assert ev_bbox.tp == 1
    assert ev_poly.tp == 0    # polygon IoU below 0.6 => not a match
