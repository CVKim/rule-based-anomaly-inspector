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


Mode = Literal["absdiff", "multiscale", "ncc", "gradient", "ridge", "fused"]
_VALID_MODES: tuple[Mode, ...] = ("absdiff", "multiscale", "ncc", "gradient",
                                  "ridge", "fused")

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

    # ridge (Frangi-style multi-scale Hessian tubeness)
    ridge_scales: tuple[float, ...] = (1.0, 2.0, 4.0)
    ridge_polarity: Literal["dark", "bright", "both"] = "dark"
    ridge_beta: float = 0.5      # blob-vs-tube discrimination
    ridge_c: float = 15.0        # noise threshold; higher = more conservative
    # Spatial-dilation radius (px) for the master ridge response before
    # subtraction. Master ridges that drift by <= this radius in the target
    # (sub-pixel alignment slop, micro-warps) still cancel cleanly. Set 0 to
    # disable for clean synthetic tests.
    ridge_master_dilate: int = 3

    # fused: per-pixel mean of normalised residuals across these modes.
    # Defaults to "the three discriminative modes" so a single
    # ResidualConfig(mode="fused") works out of the box.
    fused_modes: tuple[Mode, ...] = ("absdiff", "ncc", "ridge")
    fused_weights: tuple[float, ...] = ()    # empty = uniform

    # combine multiple residual modes by max — set to a tuple of modes to
    # union them all. Empty = use ``mode`` only. Distinct from ``fused_modes``:
    # extra_modes uses MAX over **abs** residuals (catches anything any mode
    # flagged); fused averages NORMALISED residuals (rewards mode agreement).
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
        if not self.ridge_scales or any(s <= 0 for s in self.ridge_scales):
            raise ValueError("ridge_scales must be a non-empty tuple of positive floats")
        if self.ridge_polarity not in {"dark", "bright", "both"}:
            raise ValueError("ridge_polarity must be 'dark' | 'bright' | 'both'")
        if self.ridge_beta <= 0 or self.ridge_c <= 0:
            raise ValueError("ridge_beta and ridge_c must be > 0")
        if self.ridge_master_dilate < 0:
            raise ValueError("ridge_master_dilate must be >= 0")
        for m in self.fused_modes:
            if m not in _VALID_MODES or m == "fused":
                raise ValueError(f"unknown / illegal fused mode '{m}'")
        if self.fused_weights and len(self.fused_weights) != len(self.fused_modes):
            raise ValueError("fused_weights length must match fused_modes")
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
            "ridge_scales": [float(s) for s in self.ridge_scales],
            "ridge_polarity": self.ridge_polarity,
            "ridge_beta": float(self.ridge_beta),
            "ridge_c": float(self.ridge_c),
            "ridge_master_dilate": int(self.ridge_master_dilate),
            "fused_modes": list(self.fused_modes),
            "fused_weights": list(self.fused_weights),
            "extra_modes": list(self.extra_modes),
        }

    @classmethod
    def from_meta(cls, meta: dict | None) -> "ResidualConfig":
        if not meta:
            return cls()
        extras = meta.get("extra_modes") or ()
        ridge_scales = meta.get("ridge_scales") or (1.0, 2.0, 4.0)
        fused_modes = meta.get("fused_modes") or ("absdiff", "ncc", "ridge")
        fused_weights = meta.get("fused_weights") or ()
        return cls(
            mode=meta.get("mode", "absdiff"),
            pyramid_levels=int(meta.get("pyramid_levels", 3)),
            pyramid_combine=meta.get("pyramid_combine", "max"),
            ncc_window=int(meta.get("ncc_window", 15)),
            gradient_op=meta.get("gradient_op", "scharr"),
            gradient_ksize=int(meta.get("gradient_ksize", 3)),
            gradient_blend=float(meta.get("gradient_blend", 0.0)),
            ridge_scales=tuple(float(s) for s in ridge_scales),
            ridge_polarity=meta.get("ridge_polarity", "dark"),
            ridge_beta=float(meta.get("ridge_beta", 0.5)),
            ridge_c=float(meta.get("ridge_c", 15.0)),
            ridge_master_dilate=int(meta.get("ridge_master_dilate", 3)),
            fused_modes=tuple(fused_modes),
            fused_weights=tuple(float(w) for w in fused_weights),
            extra_modes=tuple(extras),
        )


_FUSABLE: tuple[Mode, ...] = ("absdiff", "multiscale", "ncc", "gradient", "ridge")


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

    if mode == "ridge":
        ridge_abs = _ridge_residual(master, target, config)
        return ridge_abs.copy(), ridge_abs

    if mode == "fused":
        fused = _fused_residual(master, target, config)
        return fused.copy(), fused

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


# ---------- ridge (Frangi-style multi-scale Hessian) -----------------------


def _ridge_response(img: np.ndarray, sigma: float,
                    beta: float, c: float,
                    polarity: str) -> np.ndarray:
    """Frangi tubeness at a single scale.

    Computes the Hessian via Gaussian-derivative convolution, then for each
    pixel the eigenvalues ``|lambda1| <= |lambda2|``. A line-like (tubular)
    structure has |lambda1| ~ 0 and large |lambda2|; a blob has both large
    and similar; flat regions both small. Vesselness:

        Rb = lambda1 / lambda2          (blob-vs-line discriminator)
        S  = sqrt(lambda1^2 + lambda2^2) (structureness — kills noise)
        V  = exp(-Rb^2 / (2*beta^2)) * (1 - exp(-S^2 / (2*c^2)))

    With sign filter on lambda2: dark line -> lambda2 > 0, bright -> < 0.
    """
    f = img.astype(np.float32)
    # Gaussian smoothing for differentiation at scale sigma
    smoothed = cv2.GaussianBlur(f, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT)
    # Hessian components (sigma^2 normalisation per Lindeberg)
    Hxx = cv2.Sobel(smoothed, cv2.CV_32F, 2, 0, ksize=3) * (sigma ** 2)
    Hyy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 2, ksize=3) * (sigma ** 2)
    Hxy = cv2.Sobel(smoothed, cv2.CV_32F, 1, 1, ksize=3) * (sigma ** 2)

    # Closed-form 2x2 eigenvalues
    tmp = np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy ** 2 + 1e-12)
    lam1 = 0.5 * (Hxx + Hyy - tmp)
    lam2 = 0.5 * (Hxx + Hyy + tmp)

    # Order so |lam1| <= |lam2|
    abs1 = np.abs(lam1)
    abs2 = np.abs(lam2)
    swap = abs1 > abs2
    a = np.where(swap, lam2, lam1)
    b = np.where(swap, lam1, lam2)   # |b| >= |a|

    Rb = a / (b + 1e-6)
    S = np.sqrt(a * a + b * b)
    V = np.exp(-(Rb * Rb) / (2 * beta * beta)) * (
        1.0 - np.exp(-(S * S) / (2 * c * c))
    )

    if polarity == "dark":
        V = np.where(b > 0, V, 0.0)
    elif polarity == "bright":
        V = np.where(b < 0, V, 0.0)
    # both: leave as is
    V = np.where(np.isfinite(V), V, 0.0)
    return V.astype(np.float32)


def _ridge_residual(master: np.ndarray, target: np.ndarray,
                    config: ResidualConfig) -> np.ndarray:
    """Multi-scale ridge response **difference** between target and master.

    The Frangi response by itself flags every legitimate ridge in the master
    (slot edges, machined grooves) too; subtracting the master response means
    we only keep ridges that appeared (or strengthened) in the target —
    i.e. crack candidates.
    """
    polarity = config.ridge_polarity
    # Per-scale response for both, take per-pixel max across scales
    def _multi_scale(img: np.ndarray) -> np.ndarray:
        per_scale = [
            _ridge_response(img, s, config.ridge_beta, config.ridge_c, polarity)
            for s in config.ridge_scales
        ]
        return np.maximum.reduce(per_scale)

    Vm = _multi_scale(master)
    Vt = _multi_scale(target)

    # Spatial-dilation cancellation: a legitimate ridge in the master that
    # drifts by a few pixels in the target (sub-pixel alignment slop, line-
    # scan jitter, micro-warp) would otherwise leave a residual of
    # ``Vt - 0`` because Vm was zero at the SHIFTED location. Take the
    # neighbourhood-max of Vm so any nearby master ridge cancels first.
    if config.ridge_master_dilate > 0:
        d = config.ridge_master_dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * d + 1, 2 * d + 1))
        Vm = cv2.dilate(Vm, kernel)

    diff = np.maximum(Vt - Vm, 0.0)   # only "new" ridges

    # Frangi vesselness lives in [0, 1]. Rescale into the gray-level range so
    # the dynamic-tolerance threshold's base + k_sigma * std stays meaningful.
    diff_scaled = diff * _GRAY_REFERENCE_SCALE
    return diff_scaled.astype(np.float32)


# ---------- score-level fusion ---------------------------------------------


def _fused_residual(master: np.ndarray, target: np.ndarray,
                    config: ResidualConfig) -> np.ndarray:
    """Per-pixel weighted mean of normalised residuals across multiple modes.

    Each constituent residual is min/max-normalised on a robust percentile
    range (1st-99th) before averaging so a single mode with a wider numeric
    spread doesn't dominate. The result is rescaled to the gray-level
    range so the inspector's dynamic threshold still applies.

    Compared to ``extra_modes`` (per-pixel max over abs residuals), fusion
    REWARDS mode agreement instead of taking the loudest signal: a real
    defect tends to light up multiple modes, while mode-specific noise
    (e.g. NCC's edge-halo noise) only fires in one — so fusion implicitly
    cuts FPs while keeping recall.
    """
    # Default to "the three most discriminative" if none specified
    modes = config.fused_modes or ("absdiff", "ncc", "ridge")
    weights = (config.fused_weights or (1.0,) * len(modes))
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()

    accum = np.zeros_like(master, dtype=np.float32)
    for w, m in zip(weights, modes):
        if m == "fused":
            raise ValueError("fused mode cannot reference itself")
        _signed, abs_r = _residual_one(master, target, m, config)
        normed = _robust_normalise(abs_r)
        accum += w * normed

    return (accum * _GRAY_REFERENCE_SCALE).astype(np.float32)


def _robust_normalise(arr: np.ndarray,
                      lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Clip to the ``[lo, hi]`` percentile band and rescale to ``[0, 1]``."""
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
