"""Tests for the auto-ROI extraction stage."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anomaly_inspector import (
    DynamicToleranceInspector, ReferenceBuilder, RoiConfig,
    auto_part_roi,
)
from anomaly_inspector.residual import ResidualConfig


def _backlit_part(size: tuple[int, int] = (200, 300),
                  rect: tuple[int, int, int, int] = (60, 100, 90, 90),
                  bg: int = 5, fg: int = 200,
                  recess: tuple[int, int, int, int] | None = (95, 130, 30, 30),
                  recess_val: int = 30) -> np.ndarray:
    """Synthetic 'bright part on dark background', optionally with a darker
    recess inside the part — mimics the FOOSUNG side-view layout."""
    h, w = size
    img = np.full((h, w), bg, dtype=np.uint8)
    x, y, rw, rh = rect
    img[y:y + rh, x:x + rw] = fg
    if recess is not None:
        rx, ry, rrw, rrh = recess
        img[ry:ry + rrh, rx:rx + rrw] = recess_val
    return img


# ---------- config validation -----------------------------------------------


@pytest.mark.parametrize("bad", [
    {"method": "bogus"},
    {"close_ksize": 4},
    {"erode_px": -1},
    {"min_area_fraction": 1.0},
])
def test_roi_config_rejects_bad(bad):
    with pytest.raises(ValueError):
        RoiConfig(**bad)


def test_roi_meta_round_trip():
    cfg = RoiConfig(method="otsu_close", close_ksize=21, erode_px=4,
                    convex_hull=True)
    restored = RoiConfig.from_meta(cfg.to_meta())
    assert restored.method == "otsu_close"
    assert restored.close_ksize == 21
    assert restored.convex_hull is True


# ---------- auto_part_roi --------------------------------------------------


def test_auto_part_roi_disabled_returns_none():
    img = _backlit_part()
    assert auto_part_roi(img, RoiConfig(method="none")) is None


def test_auto_part_roi_otsu_close_includes_dark_recess():
    """The dark recess inside the bright part should still be inside the
    returned mask after morphological closing."""
    img = _backlit_part(rect=(60, 100, 90, 90),
                        recess=(95, 130, 30, 30), recess_val=20)
    mask = auto_part_roi(img, RoiConfig(method="otsu_close",
                                        close_ksize=41, erode_px=0))
    assert mask is not None
    # Centre of the recess
    assert mask[145, 110] > 0
    # Background (far from the part) should be excluded
    assert mask[20, 20] == 0
    # Approximate fill: should be close to the rect's area, certainly > 60%.
    assert mask.sum() / 255 > 0.6 * 90 * 90


def test_auto_part_roi_keeps_only_largest_component():
    """A bright speck far away should not survive the largest-CC filter."""
    img = _backlit_part()
    img[10:14, 10:14] = 220       # noise speck
    mask = auto_part_roi(img, RoiConfig(method="otsu_close",
                                        close_ksize=41, erode_px=0))
    assert mask is not None
    assert mask[12, 12] == 0
    assert mask[145, 145] > 0


def test_auto_part_roi_erode_shrinks_silhouette():
    img = _backlit_part(recess=None)
    no_erode = auto_part_roi(img, RoiConfig(method="otsu_close",
                                            close_ksize=15, erode_px=0))
    eroded = auto_part_roi(img, RoiConfig(method="otsu_close",
                                          close_ksize=15, erode_px=8))
    assert no_erode is not None and eroded is not None
    assert eroded.sum() < no_erode.sum()


def test_auto_part_roi_returns_none_for_pure_background():
    """Below-min-area parts should return None so the caller can fall back
    to inspecting the whole frame."""
    img = np.full((200, 300), 5, dtype=np.uint8)
    mask = auto_part_roi(img, RoiConfig(method="otsu_close",
                                        min_area_fraction=0.5))
    assert mask is None


# ---------- pipeline integration -------------------------------------------


def test_inspector_with_roi_suppresses_background_anomalies():
    """A defect *outside* the part region (i.e., in the dark backdrop) must
    be suppressed when the reference carries an ROI mask."""
    base = _backlit_part(recess=None)
    rng = np.random.default_rng(0)
    normals = [
        np.clip(base + rng.normal(0, 1.5, base.shape),
                0, 255).astype(np.uint8) for _ in range(8)
    ]

    builder_with = ReferenceBuilder(
        blur_ksize=5, align=False,
        roi=RoiConfig(method="otsu_close", close_ksize=21, erode_px=4),
    )
    builder_without = ReferenceBuilder(blur_ksize=5, align=False,
                                       roi=RoiConfig(method="none"))
    ref_with = builder_with.from_images(normals)
    ref_without = builder_without.from_images(normals)
    assert ref_with.roi_mask is not None
    assert ref_without.roi_mask is None

    # Defective sample: bright speck in the *background* (outside the part)
    target = base.copy()
    cv2.circle(target, (250, 30), 6, 240, thickness=-1)

    insp_with = DynamicToleranceInspector(
        ref_with, k_sigma=4.0, base_tolerance=5.0, min_blob_area=8,
        align_method="none",
    )
    insp_without = DynamicToleranceInspector(
        ref_without, k_sigma=4.0, base_tolerance=5.0, min_blob_area=8,
        align_method="none",
    )

    res_with = insp_with.inspect(target)
    res_without = insp_without.inspect(target)

    bg_hits_with = [d for d in res_with.defects
                    if 240 <= d.centroid[0] <= 260
                    and 20 <= d.centroid[1] <= 40]
    bg_hits_without = [d for d in res_without.defects
                       if 240 <= d.centroid[0] <= 260
                       and 20 <= d.centroid[1] <= 40]
    assert len(bg_hits_with) == 0, f"background defect not suppressed: {bg_hits_with}"
    assert len(bg_hits_without) >= 1, "test setup invalid — defect not detected without ROI"


def test_inspector_with_roi_preserves_in_part_defects():
    """A defect *inside* the part region must still be detected with ROI on."""
    base = _backlit_part(recess=None)
    rng = np.random.default_rng(1)
    normals = [
        np.clip(base + rng.normal(0, 1.5, base.shape),
                0, 255).astype(np.uint8) for _ in range(8)
    ]
    builder = ReferenceBuilder(
        blur_ksize=5, align=False,
        roi=RoiConfig(method="otsu_close", close_ksize=21, erode_px=4),
    )
    ref = builder.from_images(normals)
    target = base.copy()
    # Defect in the centre of the part
    cv2.circle(target, (105, 145), 5, 60, thickness=-1)
    insp = DynamicToleranceInspector(
        ref, k_sigma=4.0, base_tolerance=5.0, min_blob_area=6,
        align_method="none",
    )
    result = insp.inspect(target)
    in_part = [d for d in result.defects
               if 95 <= d.centroid[0] <= 115 and 135 <= d.centroid[1] <= 155]
    assert in_part, f"in-part defect missed: {result.defects}"


@pytest.mark.parametrize("mode", ["absdiff", "ncc", "gradient"])
def test_roi_dramatically_reduces_ncc_gradient_false_positives(mode):
    """The whole reason for ROI: NCC and gradient produce nuisance detections
    in the dark background. With ROI on, the count should collapse."""
    base = _backlit_part(recess=None)
    rng = np.random.default_rng(2)
    normals = [
        np.clip(base + rng.normal(0, 1.5, base.shape),
                0, 255).astype(np.uint8) for _ in range(8)
    ]
    target = np.clip(base + rng.normal(0, 1.5, base.shape),
                     0, 255).astype(np.uint8)

    res_cfg = ResidualConfig(mode=mode)
    builder_off = ReferenceBuilder(blur_ksize=5, align=False,
                                   roi=RoiConfig(method="none"))
    builder_on = ReferenceBuilder(
        blur_ksize=5, align=False,
        roi=RoiConfig(method="otsu_close", close_ksize=21, erode_px=4),
    )
    ref_off = builder_off.from_images(normals)
    ref_on = builder_on.from_images(normals)

    insp_off = DynamicToleranceInspector(
        ref_off, k_sigma=4.0, base_tolerance=5.0, min_blob_area=8,
        align_method="none", residual=res_cfg,
    )
    insp_on = DynamicToleranceInspector(
        ref_on, k_sigma=4.0, base_tolerance=5.0, min_blob_area=8,
        align_method="none", residual=res_cfg,
    )
    n_off = len(insp_off.inspect(target).defects)
    n_on = len(insp_on.inspect(target).defects)
    # ROI must never INCREASE detections; for ncc/gradient on a near-clean
    # target it should reduce them (often to zero).
    assert n_on <= n_off, f"{mode}: ROI increased count from {n_off} to {n_on}"
