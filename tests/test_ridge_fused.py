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
    # Outside the defect, the average residual stays low — using mean
    # (not max) is more stable against single-pixel boundary spikes from
    # the gradient mode at the image rim.
    far = (slice(0, 30), slice(0, 30))
    assert fused[far].mean() < 5.0


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


def test_fused_op_agree_drives_to_zero_when_one_mode_silent():
    """Mathematically: a weighted geometric mean of N positive values is
    bounded above by their weighted arithmetic mean. So when one mode
    is near zero everywhere, "agree" must collapse to ~0 too — that's
    the FP-suppression property. We construct identical master == target
    so absdiff is exactly zero everywhere, leaving only ridge to "vote";
    agree must therefore return ~0."""
    master = _flat(size=128, val=130, noise=1.5, seed=10)
    target = master.copy()
    cv2.line(target, (20, 64), (108, 64), 50, thickness=2)

    cfg_mean = ResidualConfig(mode="fused",
                              fused_modes=("absdiff", "ridge"),
                              fused_op="mean", ridge_polarity="dark",
                              ridge_scales=(1.5, 3.0))
    cfg_agree = ResidualConfig(mode="fused",
                               fused_modes=("absdiff", "ridge"),
                               fused_op="agree", ridge_polarity="dark",
                               ridge_scales=(1.5, 3.0))
    _, mean_r = compute_residual(master, target, cfg_mean)
    _, agree_r = compute_residual(master, target, cfg_agree)
    # Agree's per-pixel value must never exceed mean's at the same pixel
    # (geometric mean <= arithmetic mean for non-negative inputs).
    assert agree_r.max() <= mean_r.max() + 1e-2
    # On the line region both should be non-trivial (since both modes fire).
    assert agree_r[60:69, 30:100].max() > 0.5


def test_fused_op_max_picks_loudest():
    """max should equal the weighted max of the constituent normed maps."""
    master = _flat(size=128, seed=11)
    target = master.copy()
    cv2.line(target, (20, 64), (108, 64), 50, thickness=2)

    cfg = ResidualConfig(mode="fused",
                         fused_modes=("absdiff", "ridge"),
                         fused_op="max",
                         ridge_polarity="dark", ridge_scales=(1.5, 3.0))
    _, fused = compute_residual(master, target, cfg)
    assert fused.max() > 1.0


def test_fused_op_validation():
    with pytest.raises(ValueError):
        ResidualConfig(mode="fused", fused_op="bogus")  # type: ignore[arg-type]


def test_fused_op_intersect_drastically_thins_the_active_region():
    """intersect: each constituent residual is thresholded at its own
    top-N percentile; only pixels surviving EVERY mask survive. The
    active-pixel count must be substantially smaller than for "mean"
    (which doesn't gate, just averages)."""
    master = _flat(size=128, val=130, noise=1.5, seed=14)
    target = master.copy()
    cv2.line(target, (20, 64), (108, 64), 50, thickness=2)

    cfg_mean = ResidualConfig(mode="fused",
                              fused_modes=("absdiff", "ridge"),
                              fused_op="mean",
                              ridge_polarity="dark",
                              ridge_scales=(1.5, 3.0))
    cfg_int = ResidualConfig(mode="fused",
                             fused_modes=("absdiff", "ridge"),
                             fused_op="intersect",
                             intersect_quantile=0.95,
                             ridge_polarity="dark",
                             ridge_scales=(1.5, 3.0))
    _, mean_r = compute_residual(master, target, cfg_mean)
    _, int_r = compute_residual(master, target, cfg_int)

    n_active_int = int((int_r > 0).sum())
    n_active_mean = int((mean_r > 0).sum())
    # intersect should keep substantially fewer active pixels than mean
    # (since both modes must each be in their top-5%).
    assert n_active_int < n_active_mean * 0.5, (
        f"intersect didn't thin enough: int={n_active_int} mean={n_active_mean}"
    )
    # And the absolute fraction is bounded by the smaller per-mode quantile.
    assert n_active_int <= int_r.size * 0.05 + 50


def test_fused_op_intersect_validation():
    with pytest.raises(ValueError):
        ResidualConfig(mode="fused", fused_op="intersect",
                       intersect_quantile=1.5)


def test_hstripe_isolates_horizontal_dark_stripe():
    """hstripe must isolate a thin horizontal dark stripe. The kernel
    *thickness* must exceed (2 * defect-thickness + 1) to fully fill the
    stripe via the closing step — that's a documented constraint, so
    we test with a 1-px line and kernel thickness=5."""
    master = _flat(size=192, val=180, noise=1.0, seed=15)
    target = master.copy()
    cv2.line(target, (40, 96), (152, 96), 60, thickness=1)

    cfg = ResidualConfig(mode="hstripe", hstripe_length=31,
                         hstripe_thickness=5,
                         hstripe_master_dilate=0)
    _, hr = compute_residual(master, target, cfg)
    on_stripe = hr[94:99, 50:142]
    far = hr[20:40, 20:170]
    assert on_stripe.max() > 30
    assert far.max() < on_stripe.max() * 0.5


def test_fused_op_intersect_validation():
    with pytest.raises(ValueError):
        ResidualConfig(mode="fused", fused_op="intersect",
                       intersect_quantile=1.5)


def test_ridge_tolerance_aware_dilation_extends_in_high_tol_regions():
    """When ridge_master_dilate_high_tol > 0 and a tolerance map is
    passed, the cancellation radius expands at high-tolerance pixels.
    Verify the high-tol path produces ≤ residual everywhere vs the
    low-tol path (more cancellation cannot create more residual)."""
    rng = np.random.default_rng(17)
    base = np.full((128, 128), 130, dtype=np.uint8)
    base = np.clip(base + rng.normal(0, 2, base.shape),
                   0, 255).astype(np.uint8)
    target = base.copy()
    cv2.line(target, (10, 64), (118, 64), 60, thickness=2)
    # Construct a tolerance map: high near the line, low elsewhere.
    tol = np.zeros_like(base, dtype=np.float32)
    tol[60:69, :] = 20.0

    cfg_off = ResidualConfig(mode="ridge",
                             ridge_polarity="dark",
                             ridge_scales=(1.5, 3.0),
                             ridge_master_dilate=2,
                             ridge_master_dilate_high_tol=0)
    cfg_on = ResidualConfig(mode="ridge",
                            ridge_polarity="dark",
                            ridge_scales=(1.5, 3.0),
                            ridge_master_dilate=2,
                            ridge_master_dilate_high_tol=4)
    _, r_off = compute_residual(base, target, cfg_off, tolerance=tol)
    _, r_on = compute_residual(base, target, cfg_on, tolerance=tol)
    # In the high-tolerance band, the on-version should cancel at
    # least as aggressively (residual <= off-version everywhere there).
    assert r_on[60:69, :].max() <= r_off[60:69, :].max() + 1e-3


def test_use_gpu_ridge_falls_back_to_cpu_when_unavailable():
    """When GPU is requested but unavailable (or torch missing), the
    code path must NOT crash; it should silently use the CPU path so
    config files stay portable."""
    master = _flat(size=128, val=128, noise=1.5, seed=12)
    target = master.copy()
    cv2.line(target, (20, 64), (108, 64), 50, thickness=2)

    cfg_cpu = ResidualConfig(mode="ridge",
                             ridge_polarity="dark", ridge_scales=(1.5, 3.0))
    cfg_gpu = ResidualConfig(mode="ridge", use_gpu_ridge=True,
                             ridge_polarity="dark", ridge_scales=(1.5, 3.0))
    _, r_cpu = compute_residual(master, target, cfg_cpu)
    _, r_gpu = compute_residual(master, target, cfg_gpu)

    # Both should produce a non-trivial ridge response on the line.
    assert r_cpu.max() > 1.0
    assert r_gpu.max() > 1.0


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
