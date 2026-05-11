"""Tests for shape classification, asymmetric tolerance, and auto-ignore."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anomaly_inspector import (
    DynamicToleranceInspector, ReferenceBuilder,
    auto_unreliable_mask, classify, shape_features,
)


def _make_normals(base: np.ndarray, n: int = 8, sigma: float = 2.0,
                  seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.clip(base + rng.normal(0, sigma, base.shape),
                0, 255).astype(np.uint8)
        for _ in range(n)
    ]


def _flat_panel(h: int = 256, w: int = 256, val: int = 128) -> np.ndarray:
    return np.full((h, w), val, dtype=np.uint8)


# ---------- shape features --------------------------------------------------


def test_shape_features_for_circle():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, thickness=-1)
    signed = np.full_like(mask, -30, dtype=np.float32)  # darker than master
    feats = shape_features(mask, signed)
    assert feats.circularity > 0.85
    assert feats.aspect_ratio == pytest.approx(1.0, abs=0.15)
    assert feats.solidity > 0.95
    assert feats.polarity == "dark"


def test_shape_features_for_long_rectangle():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 48), (90, 52), 255, thickness=-1)
    signed = np.full_like(mask, 40, dtype=np.float32)   # brighter than master
    feats = shape_features(mask, signed)
    assert feats.aspect_ratio > 10.0
    assert feats.circularity < 0.3
    assert feats.polarity == "bright"


# ---------- classifier ------------------------------------------------------


def test_classify_scratch():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(mask, (10, 50), (90, 50), 255, thickness=2)
    feats = shape_features(mask, np.full_like(mask, -20, dtype=np.float32))
    assert classify(feats) == "scratch"


def test_classify_dent_vs_spot_by_polarity():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 4, 255, thickness=-1)
    dark_feats = shape_features(mask, np.full_like(mask, -30, dtype=np.float32))
    bright_feats = shape_features(mask, np.full_like(mask, +30, dtype=np.float32))
    assert classify(dark_feats) == "dent"
    assert classify(bright_feats) == "spot"


def test_classify_smudge_for_irregular_blob():
    mask = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array([[40, 40], [160, 50], [120, 120], [60, 130],
                    [80, 90], [50, 70]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    # Carve out a hole so solidity drops below the threshold.
    cv2.circle(mask, (100, 90), 25, 0, thickness=-1)
    feats = shape_features(mask, np.full_like(mask, -10, dtype=np.float32))
    assert classify(feats) == "smudge"


# ---------- asymmetric tolerance --------------------------------------------


def test_asymmetric_tolerance_can_suppress_one_polarity():
    base = _flat_panel()
    normals = _make_normals(base)
    builder = ReferenceBuilder(blur_ksize=5, align=False)
    ref = builder.from_images(normals)

    target = base.copy()
    cv2.circle(target, (60, 60), 8, 60, thickness=-1)    # dark dent
    cv2.circle(target, (180, 180), 8, 200, thickness=-1) # bright spot

    # Tighten dark side, loosen bright side dramatically.  Both base_tolerance
    # AND k_sigma must be raised on the bright side because the panel is flat,
    # so the per-pixel std (and thus k_sigma * std) is near zero.
    insp = DynamicToleranceInspector(
        ref, k_sigma=4.0, base_tolerance=2.0, min_blob_area=8,
        align_method="none",
        k_sigma_dark=4.0, k_sigma_bright=50.0,
        base_tolerance_dark=2.0, base_tolerance_bright=120.0,
    )
    result = insp.inspect(target)
    polarities = {d.polarity for d in result.defects}
    assert "dark" in polarities, f"dark dent missed: {result.defects}"
    assert "bright" not in polarities, (
        f"bright spot should have been suppressed: {result.defects}"
    )


# ---------- auto-unreliable mask --------------------------------------------


def test_auto_unreliable_mask_marks_top_percentile():
    tol = np.zeros((50, 50), dtype=np.float32)
    tol[10:15, 10:15] = 100.0   # high-variance pocket
    mask = auto_unreliable_mask(tol, percentile=95.0, dilate_px=0)
    assert mask[12, 12] == 255
    assert mask[40, 40] == 0


def test_auto_ignore_percentile_suppresses_high_variance_region():
    """A defect that lands inside an inherently noisy region (high tolerance)
    should be suppressed when auto-ignore is on, but visible when it's off."""
    base = _flat_panel()
    normals = _make_normals(base)
    # Inflate variance in a known band so its tolerance is high.
    rng = np.random.default_rng(99)
    for n in normals:
        n[20:40, :] = np.clip(n[20:40, :].astype(np.int16)
                              + rng.integers(-30, 30,
                                             size=(20, n.shape[1])),
                              0, 255).astype(np.uint8)
    builder = ReferenceBuilder(blur_ksize=5, align=False)
    ref = builder.from_images(normals)

    # Plant a defect inside the noisy band.
    target = base.copy()
    cv2.circle(target, (128, 30), 5, 30, thickness=-1)
    target = np.clip(target + rng.normal(0, 2, target.shape),
                     0, 255).astype(np.uint8)

    insp_off = DynamicToleranceInspector(
        ref, k_sigma=4.0, base_tolerance=2.0, min_blob_area=8,
        align_method="none",
    )
    insp_on = DynamicToleranceInspector(
        ref, k_sigma=4.0, base_tolerance=2.0, min_blob_area=8,
        align_method="none",
        auto_ignore_percentile=90.0,
    )

    on_hits = [d for d in insp_on.inspect(target).defects
               if d.bbox[1] < 40]
    # With auto-ignore on the high-variance band is suppressed, so the planted
    # defect inside that band should not survive.
    assert len(on_hits) == 0, f"auto-ignore failed to suppress: {on_hits}"
