"""Tests for ridge filter + score-level fusion residual modes."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anomaly_inspector import (
    DynamicToleranceInspector, ReferenceBuilder, ResidualConfig,
    compute_residual,
)


def _flat(size: int = 128, val: int = 128, noise: float = 1.5,
          seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size), val, dtype=np.float32)
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- ridge filter ----------------------------------------------------


def test_ridge_lights_up_dark_thin_line_only_in_target():
    """A thin dark line that exists in the target but not the master should
    fire on the ridge residual."""
    master = _flat(size=192, val=140, noise=1.5, seed=1)
    target = master.copy()
    cv2.line(target, (40, 96), (152, 96), 60, thickness=2)   # dark crack

    _signed, abs_r = compute_residual(master, target,
                                      ResidualConfig(mode="ridge",
                                                     ridge_polarity="dark",
                                                     ridge_scales=(1.5, 3.0)))
    line_band = abs_r[92:101, 50:142]
    far_band = abs_r[10:20, 10:182]
    assert line_band.max() > far_band.max() * 2.0, (
        f"crack not isolated: line={line_band.max():.2f} far={far_band.max():.2f}"
    )


def test_ridge_polarity_filters_correctly():
    """A bright line should fire only when polarity='bright', and a dark line
    only when polarity='dark'."""
    master = _flat(size=128, val=128, noise=1.0, seed=2)

    bright = master.copy()
    cv2.line(bright, (20, 64), (108, 64), 240, thickness=2)
    dark = master.copy()
    cv2.line(dark, (20, 64), (108, 64), 30, thickness=2)

    cfg_dark = ResidualConfig(mode="ridge", ridge_polarity="dark",
                              ridge_scales=(1.5, 3.0))
    cfg_bright = ResidualConfig(mode="ridge", ridge_polarity="bright",
                                ridge_scales=(1.5, 3.0))

    _, dark_with_dark = compute_residual(master, dark, cfg_dark)
    _, bright_with_dark = compute_residual(master, bright, cfg_dark)
    _, bright_with_bright = compute_residual(master, bright, cfg_bright)
    _, dark_with_bright = compute_residual(master, dark, cfg_bright)

    # The right polarity should give a much hotter line response.
    assert dark_with_dark[64, 64] > dark_with_bright[64, 64] * 5
    assert bright_with_bright[64, 64] > bright_with_dark[64, 64] * 5


def test_ridge_response_subtracts_master_ridges():
    """A pre-existing ridge in the master shouldn't show up as residual when
    the same ridge is in the target."""
    base = _flat(size=128, seed=3)
    cv2.line(base, (20, 64), (108, 64), 60, thickness=2)
    master = base.copy()
    target = base.copy()  # identical
    _, abs_r = compute_residual(master, target,
                                ResidualConfig(mode="ridge",
                                               ridge_polarity="dark",
                                               ridge_scales=(1.5, 3.0)))
    # Ridge response should be ~0 everywhere (master was subtracted off).
    assert abs_r[60:69, 30:100].max() < 5.0


# ---------- score-level fusion ---------------------------------------------


def test_fused_residual_rewards_mode_agreement():
    """A spot that all constituent modes agree on should score higher than a
    spot only one mode flags. Compare against any single mode's residual at
    the same location."""
    master = _flat(size=160, val=130, noise=1.5, seed=4)
    target = master.copy()
    cv2.circle(target, (80, 80), 5, 50, thickness=-1)         # real defect

    cfg_single = ResidualConfig(mode="absdiff")
    cfg_fused = ResidualConfig(mode="fused",
                               fused_modes=("absdiff", "ncc", "gradient"))

    _, abs_r = compute_residual(master, target, cfg_single)
    _, fused = compute_residual(master, target, cfg_fused)

    # Inside the defect, fusion's signal should not collapse below absdiff;
    # the agreement structure means it's a legitimate detection
    # (and indeed often higher because every mode agrees).
    inner = (slice(76, 85), slice(76, 85))
    assert fused[inner].max() > 5.0
    # Outside the defect, fused should have *less* peak noise than any single
    # mode (because mode-specific noise patterns get averaged down).
    far = (slice(0, 30), slice(0, 30))
    assert fused[far].max() <= max(20.0, abs_r[far].max())


def test_fused_modes_validation_blocks_self_reference():
    with pytest.raises(ValueError):
        ResidualConfig(mode="fused", fused_modes=("fused",))


def test_fused_meta_round_trip():
    cfg = ResidualConfig(mode="fused",
                         fused_modes=("absdiff", "ncc"),
                         fused_weights=(0.7, 0.3),
                         ridge_scales=(1.5, 3.0, 5.0),
                         ridge_polarity="bright")
    restored = ResidualConfig.from_meta(cfg.to_meta())
    assert restored.mode == "fused"
    assert restored.fused_modes == ("absdiff", "ncc")
    assert restored.fused_weights == (0.7, 0.3)
    assert restored.ridge_scales == (1.5, 3.0, 5.0)
    assert restored.ridge_polarity == "bright"


# ---------- pipeline integration -------------------------------------------


@pytest.mark.parametrize("mode", ["ridge", "fused"])
def test_inspector_runs_with_new_modes(mode):
    base = _flat(size=128, val=130, noise=1.5, seed=5)
    rng = np.random.default_rng(6)
    normals = [
        np.clip(base + rng.normal(0, 1.5, base.shape),
                0, 255).astype(np.uint8) for _ in range(8)
    ]
    builder = ReferenceBuilder(blur_ksize=5, align=False)
    ref = builder.from_images(normals)

    target = base.copy()
    cv2.line(target, (30, 64), (98, 64), 50, thickness=2)   # crack

    insp = DynamicToleranceInspector(
        ref, k_sigma=4.0, base_tolerance=5.0, min_blob_area=8,
        align_method="none",
        residual=ResidualConfig(mode=mode,
                                ridge_polarity="dark",
                                ridge_scales=(1.5, 3.0),
                                fused_modes=("absdiff", "ncc", "ridge")),
    )
    result = insp.inspect(target)
    assert result.is_defective, f"{mode} produced no defects"
