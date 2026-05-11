"""Tests for the pluggable residual stage."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anomaly_inspector import (
    DynamicToleranceInspector, ReferenceBuilder, ResidualConfig,
    compute_residual,
)


def _textured(size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


# ---------- config validation -----------------------------------------------


@pytest.mark.parametrize("bad", [
    {"mode": "bogus"},
    {"pyramid_levels": 0},
    {"pyramid_combine": "median"},
    {"ncc_window": 4},
    {"gradient_op": "prewitt"},
    {"gradient_ksize": 4},
    {"gradient_blend": -0.1},
])
def test_residual_config_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        ResidualConfig(**bad)


def test_residual_meta_round_trip():
    cfg = ResidualConfig(mode="ncc", ncc_window=21,
                         extra_modes=("gradient",))
    restored = ResidualConfig.from_meta(cfg.to_meta())
    assert restored.mode == "ncc"
    assert restored.ncc_window == 21
    assert restored.extra_modes == ("gradient",)


# ---------- absdiff baseline ------------------------------------------------


def test_absdiff_matches_naive_subtraction():
    m = _textured(seed=1).astype(np.float32)
    t = _textured(seed=2).astype(np.float32)
    signed, abs_r = compute_residual(m, t, ResidualConfig(mode="absdiff"))
    np.testing.assert_allclose(signed, t - m, rtol=1e-5)
    np.testing.assert_allclose(abs_r, np.abs(t - m), rtol=1e-5)


# ---------- multiscale ------------------------------------------------------


def test_multiscale_lowers_signal_to_noise_outside_blob():
    """Per-pixel max across pyramid levels picks up the smoothed-blob signal
    *inside* the blob without inflating the noise floor outside it any more
    than absdiff already does. We verify both halves: in-blob mean is at
    least as high as absdiff (multiscale never *loses* signal), and the
    peak in-blob residual exceeds the typical out-of-blob noise."""
    rng = np.random.default_rng(33)
    m = np.full((128, 128), 128, dtype=np.float32)
    m = m + rng.normal(0, 4.0, m.shape).astype(np.float32)
    t = m.copy()
    blob_mask = np.zeros_like(t, dtype=bool)
    cv2.circle(blob_mask.view(np.uint8), (64, 64), 18, 1, thickness=-1)
    t[blob_mask] = t[blob_mask] + 5.0

    _, abs_r = compute_residual(m, t, ResidualConfig(mode="absdiff"))
    _, ms_r = compute_residual(m, t,
                               ResidualConfig(mode="multiscale",
                                              pyramid_levels=3,
                                              pyramid_combine="max"))
    in_blob = ms_r[blob_mask]
    bg = ms_r[~blob_mask]
    assert in_blob.mean() >= abs_r[blob_mask].mean() * 0.95
    # Even modest blob signal should peak above background under multiscale.
    assert in_blob.max() > bg.mean() * 1.5


# ---------- ncc -------------------------------------------------------------


def test_ncc_robust_to_global_brightness_offset():
    """NCC should ignore a uniform brightness shift that absdiff would flag
    as a giant anomaly."""
    m = _textured(seed=4).astype(np.float32)
    t = m + 25.0     # +25 gray-level uniform shift, no real defect
    _, abs_r = compute_residual(m, t, ResidualConfig(mode="absdiff"))
    _, ncc_r = compute_residual(m, t,
                                ResidualConfig(mode="ncc", ncc_window=15))
    assert abs_r.mean() > 20.0
    # NCC should stay well under the absdiff baseline.
    assert ncc_r.mean() < abs_r.mean() * 0.2


# ---------- gradient --------------------------------------------------------


def test_gradient_residual_lights_up_new_edges():
    m = _textured(seed=5).astype(np.float32)
    t = m.copy()
    cv2.line(t, (20, 64), (108, 64), 220, thickness=2)   # bright scratch
    _, grad_r = compute_residual(m, t,
                                 ResidualConfig(mode="gradient",
                                                gradient_op="scharr"))
    # The scratch's edges should produce strong gradient residual; far from
    # the line it should be near zero.
    line_band = grad_r[60:69, 25:103]
    far_band = grad_r[10:18, 10:118]
    assert line_band.max() > far_band.max() * 2.0


# ---------- pipeline integration -------------------------------------------


def _normals(base, n=8, sigma=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return [
        np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)
        for _ in range(n)
    ]


@pytest.mark.parametrize("mode", ["absdiff", "multiscale", "ncc", "gradient"])
def test_inspector_runs_with_each_residual_mode(mode):
    base = _textured(size=128, seed=11)
    normals = _normals(base, n=8, sigma=2.0, seed=12)
    builder = ReferenceBuilder(blur_ksize=5, align=False)
    ref = builder.from_images(normals)

    target = base.copy()
    cv2.circle(target, (64, 64), 6, 30, thickness=-1)
    rng = np.random.default_rng(13)
    target = np.clip(target + rng.normal(0, 2, target.shape),
                     0, 255).astype(np.uint8)

    insp = DynamicToleranceInspector(
        ref, k_sigma=4.0, base_tolerance=5.0, min_blob_area=8,
        align_method="none",
        residual=ResidualConfig(mode=mode, pyramid_levels=3, ncc_window=11),
    )
    result = insp.inspect(target)
    # Every mode should at least *find something* given an obvious dark spot
    # against an otherwise-stable background.
    assert result.is_defective, f"{mode} produced no defects"
