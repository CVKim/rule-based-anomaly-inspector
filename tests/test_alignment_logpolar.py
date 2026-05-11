"""Tests for log-polar based rotation+scale alignment."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anomaly_inspector import (
    DynamicToleranceInspector, ReferenceBuilder,
    align_log_polar, estimate_rotation_scale,
)


def _textured_image(size: int = 256, seed: int = 0) -> np.ndarray:
    """Generate a non-periodic, broadband-spectrum test image so phase
    correlation in the log-polar domain has plenty of features to lock onto."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    # Smooth a bit to make it less alias-prone after rotation.
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Add a few large blobs to give the spectrum dominant low-frequency content.
    for _ in range(8):
        cx, cy = rng.integers(20, size - 20, size=2)
        r = int(rng.integers(8, 24))
        v = int(rng.integers(0, 256))
        cv2.circle(img, (int(cx), int(cy)), r, v, thickness=-1)
    return img


def _rotate_scale(img: np.ndarray, deg: float, scale: float = 1.0) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, scale)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


@pytest.mark.parametrize("rot", [-7.0, -2.5, 1.5, 6.0])
def test_estimate_rotation_recovers_known_angle(rot):
    base = _textured_image(seed=1)
    rotated = _rotate_scale(base, rot)
    est_rot, est_scale = estimate_rotation_scale(base, rotated)
    assert abs(est_rot - rot) < 1.0, f"rot mismatch: true={rot}, got={est_rot}"
    assert abs(est_scale - 1.0) < 0.05, f"scale drifted: got={est_scale}"


def test_estimate_scale_recovers_known_zoom():
    base = _textured_image(seed=2)
    scaled = _rotate_scale(base, 0.0, scale=1.05)
    _est_rot, est_scale = estimate_rotation_scale(base, scaled)
    assert abs(est_scale - 1.05) < 0.03, f"scale mismatch: got={est_scale}"


def test_align_log_polar_brings_rotated_image_into_register():
    base = _textured_image(seed=3)
    rotated = _rotate_scale(base, -4.0)

    res = align_log_polar(base, rotated, refine_translation=True)

    pre_diff = float(np.abs(base.astype(np.int16) -
                            rotated.astype(np.int16)).mean())
    post_diff = float(np.abs(base.astype(np.int16) -
                             res.aligned.astype(np.int16)).mean())
    # Alignment should at least halve the mean absolute difference.
    assert post_diff < pre_diff * 0.6, (
        f"alignment did not improve diff: pre={pre_diff:.2f}, post={post_diff:.2f}"
    )
    assert abs(res.rotation_deg - (-4.0)) < 1.0


def test_inspector_logpolar_detects_defect_under_rotation():
    rng = np.random.default_rng(4)
    base = _textured_image(seed=5)

    # 8 mildly noisy "good" samples, all aligned with the master.
    normals = []
    for _ in range(8):
        n = np.clip(base + rng.normal(0, 2, base.shape),
                    0, 255).astype(np.uint8)
        normals.append(n)

    # Defective sample = base + small dark spot, rotated by 3 degrees.
    defective = base.copy()
    cv2.circle(defective, (160, 110), 5, 20, thickness=-1)
    defective = _rotate_scale(defective, 3.0)

    builder = ReferenceBuilder(blur_ksize=5, align=True)
    ref = builder.from_images(normals)
    insp = DynamicToleranceInspector(ref, k_sigma=4.0, base_tolerance=5.0,
                                     min_blob_area=10,
                                     align_method="logpolar+phase")
    result = insp.inspect(defective)

    # The recovered rotation should be close to 3 degrees.
    assert abs(result.rotation_deg - 3.0) < 1.5, \
        f"rotation not recovered: {result.rotation_deg}"
    assert result.is_defective, "no defect detected after log-polar alignment"


def test_inspector_rejects_unknown_align_method():
    base = _textured_image(seed=6)
    ref = ReferenceBuilder().from_images([base, base, base])
    with pytest.raises(ValueError):
        DynamicToleranceInspector(ref, align_method="bogus")
