"""Tests for the photometric normalization module."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from anomaly_inspector import (
    DynamicToleranceInspector, PhotometricCorrector, ReferenceBuilder,
    flat_field_divide, top_hat, clahe,
)


@pytest.fixture
def flat_panel() -> np.ndarray:
    rng = np.random.default_rng(0)
    panel = np.full((128, 128), 128, dtype=np.uint8)
    return np.clip(panel + rng.normal(0, 2, panel.shape), 0, 255).astype(np.uint8)


def _gradient_illumination(h: int = 128, w: int = 128) -> np.ndarray:
    """Linear left-to-right brightness ramp from 60 to 200."""
    ramp = np.linspace(60, 200, w, dtype=np.float32)
    return np.tile(ramp, (h, 1))


def test_flat_field_flattens_brightness_ramp():
    illum = _gradient_illumination()
    rng = np.random.default_rng(1)
    img = np.clip(illum + rng.normal(0, 2, illum.shape), 0, 255).astype(np.uint8)

    raw_std = float(img.astype(np.float32).std())
    corrected = flat_field_divide(img, sigma=21.0)
    corrected_std = float(corrected.astype(np.float32).std())

    # The ramp dominates the original std.  After flat-field, the std should
    # collapse to roughly noise level (well below the original).
    assert corrected_std < raw_std * 0.3, (
        f"flat-field did not flatten ramp: raw={raw_std:.1f}, after={corrected_std:.1f}"
    )


def test_top_hat_white_isolates_bright_spot():
    bg = _gradient_illumination().astype(np.uint8)
    img = bg.copy()
    cv2.circle(img, (64, 64), 4, 255, thickness=-1)

    out = top_hat(img, ksize=31, polarity="white")
    # The spot should still be visible; the background ramp should be gone.
    assert out[64, 64] > 100
    # Far from the spot, the top-hat output should be near zero.
    assert out[10, 10] < 30
    assert out[10, 110] < 30


def test_clahe_increases_local_contrast():
    flat = np.full((128, 128), 128, dtype=np.uint8)
    flat[40:88, 40:88] = 132   # very faint square
    out = clahe(flat, clip_limit=4.0, tile_grid=8)
    # Inside-vs-outside contrast should be amplified.
    inside = float(out[40:88, 40:88].mean())
    outside = float(np.concatenate([out[:40, :].ravel(),
                                    out[88:, :].ravel()]).mean())
    assert abs(inside - outside) > 4


def test_corrector_meta_round_trip():
    c = PhotometricCorrector(method="flat_field", sigma=15.0)
    meta = c.to_meta()
    restored = PhotometricCorrector.from_meta(meta)
    assert restored.method == "flat_field"
    assert restored.sigma == pytest.approx(15.0)


def test_corrector_rejects_unknown_method():
    with pytest.raises(ValueError):
        PhotometricCorrector(method="bogus")  # type: ignore[arg-type]


def test_inspector_warns_on_photometric_mismatch(flat_panel, caplog):
    builder = ReferenceBuilder(blur_ksize=5, align=True,
                               photometric=PhotometricCorrector(method="flat_field"))
    ref = builder.from_images([flat_panel.copy() for _ in range(5)])

    # Our package logger has propagate=False by default to avoid double-logging
    # when embedded in a host app.  Enable it for the duration of the test so
    # caplog (which attaches at the root) can see the warning.
    logger = logging.getLogger("anomaly_inspector")
    prior = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="anomaly_inspector"):
            DynamicToleranceInspector(
                ref, photometric=PhotometricCorrector(method="none"),
            )
    finally:
        logger.propagate = prior
    assert any("photometric method" in m for m in caplog.messages), \
        f"warning not emitted: {caplog.messages}"


def test_pipeline_with_flat_field_detects_planted_defect_under_drift():
    """Whole-pipeline check: with strong illumination drift the raw pipeline
    is overwhelmed; flat-field correction recovers the defect."""
    rng = np.random.default_rng(2)
    base = _gradient_illumination().astype(np.uint8)

    normals = []
    for _ in range(8):
        noisy = np.clip(base + rng.normal(0, 2, base.shape),
                        0, 255).astype(np.uint8)
        normals.append(noisy)

    defective = base.copy()
    cv2.circle(defective, (64, 64), 5, 30, thickness=-1)
    defective = np.clip(defective + rng.normal(0, 2, defective.shape),
                        0, 255).astype(np.uint8)

    builder = ReferenceBuilder(
        blur_ksize=5, align=False,    # rig already aligned; skip phase corr
        photometric=PhotometricCorrector(method="flat_field", sigma=21.0),
    )
    ref = builder.from_images(normals)
    # Disable target-side alignment too: after flat-fielding, the panel is
    # nearly featureless and phase correlation would lock onto the defect
    # itself, producing nonsense shifts.
    insp = DynamicToleranceInspector(ref, k_sigma=4.0, base_tolerance=5.0,
                                     min_blob_area=8, align_method="none")
    result = insp.inspect(defective)
    assert result.is_defective
    hits = [d for d in result.defects
            if d.bbox[0] <= 64 <= d.bbox[0] + d.bbox[2]
            and d.bbox[1] <= 64 <= d.bbox[1] + d.bbox[3]]
    assert hits, f"defect not recovered after flat-field: {result.defects}"
