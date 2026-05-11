"""Pluggable residual maps for anomaly inspection.

The original v0.1 inspector used a single residual: ``|target - master|``.
This module generalises that step so the inspector can pick (or combine):

* ``absdiff``     — the original absolute difference. Cheap, signed-aware.
* ``multiscale``  — Gaussian-pyramid absdiff, fused via per-pixel max so a
                    single defect that only shows at one scale still scores.
* ``ncc``         — local normalised cross-correlation residual,
                    ``1 - NCC(target, master)``. Robust to global brightness
                    or contrast drift the photometric stage didn't catch.
* ``gradient``    — Sobel/Scharr gradient-magnitude difference, sensitive to
                    edge-shape defects (scratches, missing chamfers) that low
                    intensity contrast hides.

All residual functions return a float32 map in the same gray-level units as
the master, so the existing ``base_tolerance + k_sigma * std`` thresholding
keeps the same operator-facing meaning. NCC and gradient maps are scaled to
that range during construction so the user doesn't have to retune k_sigma
when switching modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np


Mode = Literal["absdiff", "multiscale", "ncc", "gradient"]
_VALID_MODES: tuple[Mode, ...] = ("absdiff", "multiscale", "ncc", "gradient")

# What an ``absdiff`` residual of "obvious defect" looks like in 8-bit units —
# used as the reference scale when normalising NCC / gradient outputs into
# the same numerical neighbourhood.
_GRAY_REFERENCE_SCALE = 255.0


@dataclass(frozen=True)
class ResidualConfig:
    """Selects and parameterises the residual stage."""

    mode: Mode = "absdiff"

    # multiscale
    pyramid_levels: int = 3                 # number of Gaussian-pyramid levels
    pyramid_combine: Literal["max", "mean"] = "max"

    # ncc
    ncc_window: int = 15                    # square window side (odd)

    # gradient
    gradient_op: Literal["sobel", "scharr"] = "scharr"
    gradient_ksize: int = 3
    # When non-zero the gradient residual is *added* to the absdiff residual
    # with this weight, instead of replacing it. Keeps intensity-defect
    # sensitivity while gaining edge-defect sensitivity.
    gradient_blend: float = 0.0

    # combine multiple residual modes by max — set to a tuple of modes to
    # union them all. Empty = use ``mode`` only.
    extra_modes: tuple[Mode, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"unknown mode '{self.mode}', expected one of {_VALID_MODES}")
        if self.pyramid_levels < 1:
            raise ValueError("pyramid_levels must be >= 1")
        if self.pyramid_combine not in {"max", "mean"}:
            raise ValueError("pyramid_combine must be 'max' or 'mean'")
        if self.ncc_window < 3 or self.ncc_window % 2 == 0:
            raise ValueError("ncc_window must be an odd integer >= 3")
        if self.gradient_op not in {"sobel", "scharr"}:
            raise ValueError("gradient_op must be 'sobel' or 'scharr'")
        if self.gradient_ksize not in {1, 3, 5, 7}:
            raise ValueError("gradient_ksize must be 1, 3, 5, or 7")
        if self.gradient_blend < 0:
            raise ValueError("gradient_blend must be >= 0")
        for m in self.extra_modes:
            if m not in _VALID_MODES:
                raise ValueError(f"unknown extra mode '{m}'")

    def to_meta(self) -> dict:
        return {
            "mode": self.mode,
            "pyramid_levels": int(self.pyramid_levels),
            "pyramid_combine": self.pyramid_combine,
            "ncc_window": int(self.ncc_window),
            "gradient_op": self.gradient_op,
            "gradient_ksize": int(self.gradient_ksize),
            "gradient_blend": float(self.gradient_blend),
            "extra_modes": list(self.extra_modes),
        }

    @classmethod
    def from_meta(cls, meta: dict | None) -> "ResidualConfig":
        if not meta:
            return cls()
        extras = meta.get("extra_modes") or ()
        return cls(
            mode=meta.get("mode", "absdiff"),
            pyramid_levels=int(meta.get("pyramid_levels", 3)),
            pyramid_combine=meta.get("pyramid_combine", "max"),
            ncc_window=int(meta.get("ncc_window", 15)),
            gradient_op=meta.get("gradient_op", "scharr"),
            gradient_ksize=int(meta.get("gradient_ksize", 3)),
            gradient_blend=float(meta.get("gradient_blend", 0.0)),
            extra_modes=tuple(extras),
        )


def compute_residual(master: np.ndarray, target: np.ndarray,
                     config: ResidualConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(signed_residual, abs_residual)`` for the configured mode.

    Both maps are float32 in the same gray-level space as ``master`` so the
    inspector can apply the same dynamic-tolerance thresholding regardless
    of which residual mode is in play.

    The signed residual is meaningful for ``absdiff`` (and its multiscale
    variant); for ``ncc`` and ``gradient`` it is identical to the absolute
    residual since the underlying quantity has no natural sign.
    """
    m = master.astype(np.float32)
    t = target.astype(np.float32)

    primary_signed, primary_abs = _residual_one(m, t, config.mode, config)

    if config.gradient_blend > 0 and config.mode != "gradient":
        _g_signed, g_abs = _residual_one(m, t, "gradient", config)
        primary_abs = primary_abs + config.gradient_blend * g_abs
        # Signed residual loses meaning when blended with an unsigned source;
        # keep the original sign mask for any downstream polarity check.

    if config.extra_modes:
        for extra_mode in config.extra_modes:
            if extra_mode == config.mode:
                continue
            _signed, abs_extra = _residual_one(m, t, extra_mode, config)
            primary_abs = np.maximum(primary_abs, abs_extra)

    return primary_signed, primary_abs


# ---------- internals --------------------------------------------------------


def _residual_one(master: np.ndarray, target: np.ndarray,
                  mode: Mode, config: ResidualConfig) -> tuple[np.ndarray, np.ndarray]:
    if mode == "absdiff":
        signed = target - master
        return signed, np.abs(signed)

    if mode == "multiscale":
        return _multiscale_residual(master, target, config)

    if mode == "ncc":
        ncc_abs = _local_ncc_residual(master, target, config.ncc_window)
        return ncc_abs.copy(), ncc_abs

    if mode == "gradient":
        grad_abs = _gradient_residual(master, target, config.gradient_op,
                                      config.gradient_ksize)
        return grad_abs.copy(), grad_abs

    raise AssertionError(f"unhandled mode '{mode}'")  # pragma: no cover


def _multiscale_residual(master: np.ndarray, target: np.ndarray,
                         config: ResidualConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build a Gaussian pyramid for both, residual at each level, fuse upward.

    A small spot vanishes from coarser levels but lights up the finest one;
    a wide blob barely shows at the finest scale but dominates a coarser one.
    Per-pixel max captures both without forcing the operator to commit to a
    single blur scale.
    """
    levels = config.pyramid_levels
    h, w = master.shape

    m_pyr = [master]
    t_pyr = [target]
    for _ in range(levels - 1):
        m_pyr.append(cv2.pyrDown(m_pyr[-1]))
        t_pyr.append(cv2.pyrDown(t_pyr[-1]))

    fused_signed: np.ndarray | None = None
    fused_abs: np.ndarray | None = None
    for lvl, (m_l, t_l) in enumerate(zip(m_pyr, t_pyr)):
        signed = (t_l - m_l).astype(np.float32)
        abs_l = np.abs(signed)
        if lvl > 0:
            signed = cv2.resize(signed, (w, h), interpolation=cv2.INTER_LINEAR)
            abs_l = cv2.resize(abs_l, (w, h), interpolation=cv2.INTER_LINEAR)

        if fused_abs is None:
            fused_signed = signed
            fused_abs = abs_l
            continue
        if config.pyramid_combine == "max":
            mask = abs_l > fused_abs
            fused_abs = np.where(mask, abs_l, fused_abs)
            fused_signed = np.where(mask, signed, fused_signed)
        else:   # mean
            fused_abs = (fused_abs + abs_l) / 2.0
            fused_signed = (fused_signed + signed) / 2.0

    assert fused_signed is not None and fused_abs is not None
    return fused_signed.astype(np.float32), fused_abs.astype(np.float32)


def _local_ncc_residual(master: np.ndarray, target: np.ndarray,
                        window: int) -> np.ndarray:
    """``(1 - NCC) * 0.5 * 255`` — local normalised cross-correlation residual.

    Computes a per-pixel correlation coefficient inside a square window
    using running sums (separable box filters), then maps NCC ∈ [-1, 1]
    into an 8-bit-equivalent positive residual where 0 means "perfect
    local correlation" and 255 means "anti-correlated patch".
    """
    k = (window, window)
    inv_n = 1.0 / (window * window)

    mean_m = cv2.boxFilter(master, ddepth=cv2.CV_32F, ksize=k,
                           normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_t = cv2.boxFilter(target, ddepth=cv2.CV_32F, ksize=k,
                           normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_mm = cv2.boxFilter(master * master, ddepth=cv2.CV_32F, ksize=k,
                            normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_tt = cv2.boxFilter(target * target, ddepth=cv2.CV_32F, ksize=k,
                            normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_mt = cv2.boxFilter(master * target, ddepth=cv2.CV_32F, ksize=k,
                            normalize=True, borderType=cv2.BORDER_REFLECT)

    var_m = np.maximum(mean_mm - mean_m * mean_m, 0)
    var_t = np.maximum(mean_tt - mean_t * mean_t, 0)
    cov = mean_mt - mean_m * mean_t

    denom = np.sqrt(var_m * var_t) + 1e-3
    ncc = cov / denom
    ncc = np.clip(ncc, -1.0, 1.0)

    residual = (1.0 - ncc) * 0.5 * _GRAY_REFERENCE_SCALE
    # Suppress the boundary band where the box filter spills into reflected
    # pixels; otherwise the rim glows uniformly bright.
    pad = window // 2
    if pad > 0:
        residual[:pad, :] = 0
        residual[-pad:, :] = 0
        residual[:, :pad] = 0
        residual[:, -pad:] = 0
    _ = inv_n  # kept for clarity; box filter's normalize=True already does it
    return residual.astype(np.float32)


def _gradient_residual(master: np.ndarray, target: np.ndarray,
                       op: str, ksize: int) -> np.ndarray:
    """Difference of gradient magnitudes, scaled into the gray-level range."""
    if op == "scharr":
        gx_m = cv2.Scharr(master, cv2.CV_32F, 1, 0)
        gy_m = cv2.Scharr(master, cv2.CV_32F, 0, 1)
        gx_t = cv2.Scharr(target, cv2.CV_32F, 1, 0)
        gy_t = cv2.Scharr(target, cv2.CV_32F, 0, 1)
    else:
        gx_m = cv2.Sobel(master, cv2.CV_32F, 1, 0, ksize=ksize)
        gy_m = cv2.Sobel(master, cv2.CV_32F, 0, 1, ksize=ksize)
        gx_t = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=ksize)
        gy_t = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=ksize)

    mag_m = cv2.magnitude(gx_m, gy_m)
    mag_t = cv2.magnitude(gx_t, gy_t)
    diff = np.abs(mag_t - mag_m)

    # The gradient operators output values that grow with kernel size; rescale
    # so the bulk of the distribution lives in the same neighbourhood as the
    # master's intensity range. We use a percentile so a single hot edge
    # doesn't compress the rest of the map.
    scale_ref = float(np.percentile(np.maximum(mag_m, mag_t), 99.0))
    if scale_ref > 1e-3:
        diff = diff / scale_ref * _GRAY_REFERENCE_SCALE
    return diff.astype(np.float32)
