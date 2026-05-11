"""Smoke tests against synthetic data — no real images required."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anomaly_inspector import DynamicToleranceInspector, ReferenceBuilder
from anomaly_inspector.alignment import align_translation


@pytest.fixture
def checkerboard() -> np.ndarray:
    img = np.zeros((128, 128), dtype=np.uint8)
    for y in range(0, 128, 16):
        for x in range(0, 128, 16):
            if ((x // 16) + (y // 16)) % 2 == 0:
                img[y:y + 16, x:x + 16] = 200
            else:
                img[y:y + 16, x:x + 16] = 60
    return img


@pytest.fixture
def normals(checkerboard) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        np.clip(checkerboard + rng.normal(0, 2, checkerboard.shape),
                0, 255).astype(np.uint8)
        for _ in range(8)
    ]


def test_reference_shape_matches_input(normals):
    builder = ReferenceBuilder(blur_ksize=5, align=True, dispersion="std")
    ref = builder.from_images(normals)
    assert ref.master.shape == normals[0].shape
    assert ref.tolerance.shape == normals[0].shape
    assert ref.n_samples == len(normals)
    assert ref.method == "std"


def test_clean_image_has_no_defects(normals, checkerboard):
    builder = ReferenceBuilder(blur_ksize=5, align=True)
    ref = builder.from_images(normals)
    inspector = DynamicToleranceInspector(ref, k_sigma=4.0, base_tolerance=5.0,
                                          min_blob_area=10)
    rng = np.random.default_rng(99)
    clean = np.clip(checkerboard + rng.normal(0, 2, checkerboard.shape),
                    0, 255).astype(np.uint8)
    result = inspector.inspect(clean)
    assert not result.is_defective, f"clean image flagged {len(result.defects)} defects"


def test_planted_defect_is_detected(normals, checkerboard):
    builder = ReferenceBuilder(blur_ksize=5, align=True)
    ref = builder.from_images(normals)
    inspector = DynamicToleranceInspector(ref, k_sigma=4.0, base_tolerance=5.0,
                                          min_blob_area=10)

    defective = checkerboard.copy()
    cv2.circle(defective, (64, 64), 6, 20, thickness=-1)

    result = inspector.inspect(defective)
    assert result.is_defective
    # at least one defect overlaps the planted location
    hits = [d for d in result.defects
            if d.bbox[0] <= 64 <= d.bbox[0] + d.bbox[2]
            and d.bbox[1] <= 64 <= d.bbox[1] + d.bbox[3]]
    assert hits, f"no defect overlaps planted location, got {result.defects}"


def test_ignore_mask_suppresses_region(normals, checkerboard):
    builder = ReferenceBuilder(blur_ksize=5, align=True)
    ref = builder.from_images(normals)
    inspector = DynamicToleranceInspector(ref, k_sigma=4.0, base_tolerance=5.0,
                                          min_blob_area=10)

    defective = checkerboard.copy()
    cv2.circle(defective, (64, 64), 6, 20, thickness=-1)

    ignore = np.zeros_like(defective)
    cv2.rectangle(ignore, (50, 50), (80, 80), 255, thickness=-1)

    result = inspector.inspect(defective, ignore_mask=ignore)
    in_box = [d for d in result.defects
              if 50 <= d.centroid[0] <= 80 and 50 <= d.centroid[1] <= 80]
    assert not in_box, f"ignored region still produced defects: {in_box}"


def test_align_translation_recovers_known_shift():
    # Use a non-periodic image so phase correlation is unambiguous.
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    M = np.float32([[1, 0, 3], [0, 1, -2]])
    shifted = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
    res = align_translation(base, shifted)
    assert abs(res.shift[0] - 3) < 0.5
    assert abs(res.shift[1] - (-2)) < 0.5


def test_shape_mismatch_raises(normals):
    builder = ReferenceBuilder(blur_ksize=5, align=True)
    ref = builder.from_images(normals)
    inspector = DynamicToleranceInspector(ref)
    wrong = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError):
        inspector.inspect(wrong)
