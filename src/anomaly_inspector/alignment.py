"""Image alignment utilities — translation (phase correlation), ECC, and
log-polar based rotation+scale recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    aligned: np.ndarray
    warp_matrix: np.ndarray
    method: str
    shift: Tuple[float, float]  # (dx, dy) — translation portion only
    rotation_deg: float = 0.0
    scale: float = 1.0


def align_translation(master: np.ndarray, target: np.ndarray) -> AlignmentResult:
    """Sub-pixel translation alignment via phase correlation.

    `master` and `target` may be uint8 or float; both are coerced to float32
    on the same scale before correlation.
    """
    m = np.float32(master)
    t = np.float32(target)
    (dx, dy), _response = cv2.phaseCorrelate(m, t)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    h, w = target.shape[:2]
    aligned = cv2.warpAffine(target, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return AlignmentResult(aligned=aligned, warp_matrix=M,
                           method="phase", shift=(float(dx), float(dy)))


def align_ecc(master: np.ndarray, target: np.ndarray,
              motion: str = "euclidean",
              max_iter: int = 50, eps: float = 1e-4,
              init: np.ndarray | None = None) -> AlignmentResult:
    """ECC (Enhanced Correlation Coefficient) alignment, default euclidean
    motion model (translation + rotation). Falls back gracefully if ECC fails
    to converge by returning the input as-is with identity warp."""
    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY,
    }
    if motion not in motion_map:
        raise ValueError(f"unsupported motion '{motion}'")
    motion_type = motion_map[motion]

    if motion_type == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32) if init is None else init.astype(np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32) if init is None else init.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    m = np.float32(master) / 255.0 if master.dtype == np.uint8 else np.float32(master)
    t = np.float32(target) / 255.0 if target.dtype == np.uint8 else np.float32(target)

    try:
        _, warp = cv2.findTransformECC(m, t, warp, motion_type, criteria, None, 5)
    except cv2.error:
        h, w = target.shape[:2]
        return AlignmentResult(aligned=target.copy(), warp_matrix=warp,
                               method=f"ecc-{motion}-failed", shift=(0.0, 0.0))

    h, w = target.shape[:2]
    if motion_type == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(target, warp, (w, h),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REPLICATE)
    else:
        aligned = cv2.warpAffine(target, warp, (w, h),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)

    dx = float(warp[0, 2]) if warp.shape[0] >= 2 else 0.0
    dy = float(warp[1, 2]) if warp.shape[0] >= 2 else 0.0
    return AlignmentResult(aligned=aligned, warp_matrix=warp,
                           method=f"ecc-{motion}", shift=(dx, dy))


# ---------- log-polar (rotation + scale) -------------------------------------


def _windowed_magnitude_spectrum(img: np.ndarray) -> np.ndarray:
    """|FFT| of a Hann-windowed image, fft-shifted to centre DC.

    Windowing suppresses the cross-shaped spectral artifact you get from the
    image boundary discontinuity, which otherwise dominates the log-polar
    correlation and makes the rotation estimate noisy.
    """
    h, w = img.shape
    win_y = np.hanning(h).astype(np.float32)
    win_x = np.hanning(w).astype(np.float32)
    window = np.outer(win_y, win_x)
    windowed = img.astype(np.float32) * window
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    return np.abs(spectrum).astype(np.float32)


def estimate_rotation_scale(master: np.ndarray,
                            target: np.ndarray) -> tuple[float, float]:
    """Recover (rotation_degrees, scale) of `target` w.r.t. `master`.

    Implements the classic Reddy & Chatterji 1996 method: the magnitude
    spectrum is translation-invariant, and a log-polar warp turns rotation
    into a vertical shift and scale into a horizontal shift.  A single phase
    correlation in log-polar space then recovers both at once.

    Returned values describe how the *target* differs from the master, i.e.
    rotating the target by ``-rotation_degrees`` and scaling by ``1/scale``
    should bring it back into register.
    """
    if master.shape != target.shape:
        raise ValueError(f"shape mismatch: {master.shape} vs {target.shape}")

    m_mag = _windowed_magnitude_spectrum(master)
    t_mag = _windowed_magnitude_spectrum(target)

    # A high-pass on the magnitude spectrum kills the DC plateau that would
    # otherwise dominate the log-polar correlation.
    m_mag = _highpass_filter(m_mag)
    t_mag = _highpass_filter(t_mag)

    h, w = m_mag.shape
    centre = (w / 2.0, h / 2.0)
    radius = float(min(centre))
    flags = cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR

    m_polar = cv2.warpPolar(m_mag, (w, h), centre, radius, flags)
    t_polar = cv2.warpPolar(t_mag, (w, h), centre, radius, flags)

    (shift_x, shift_y), _resp = cv2.phaseCorrelate(m_polar.astype(np.float32),
                                                   t_polar.astype(np.float32))

    # Vertical shift maps to rotation (angle wraps over the full image height).
    rotation_deg = -shift_y * 360.0 / h
    # Horizontal shift maps to log-radius. Note: scaling the spatial image by
    # `s` shrinks the magnitude spectrum by `1/s`, so the log-polar shift is
    # `-log(s)`. Hence the negation below — without it we'd report 1/s.
    log_base = np.log(radius) / w
    scale = float(np.exp(-shift_x * log_base))
    return float(rotation_deg), scale


def align_log_polar(master: np.ndarray, target: np.ndarray,
                    refine_translation: bool = True) -> AlignmentResult:
    """Recover rotation + scale + translation between master and target.

    Stages:
    1. ``estimate_rotation_scale`` recovers (theta, s).
    2. The target is warped by R(-theta) * 1/s about its centre to undo it.
    3. Optional ``refine_translation`` runs a final phase correlation to clean
       up any residual sub-pixel translation.

    Returns an ``AlignmentResult`` whose ``warp_matrix`` is the 2x3 affine
    actually applied.
    """
    if master.shape != target.shape:
        raise ValueError(f"shape mismatch: {master.shape} vs {target.shape}")

    rot_deg, scale = estimate_rotation_scale(master, target)

    h, w = target.shape[:2]
    centre = (w / 2.0, h / 2.0)
    # Inverse rotation+scale: rotate by -rot, scale by 1/s.
    inv_scale = 1.0 / scale if abs(scale) > 1e-6 else 1.0
    rot_matrix = cv2.getRotationMatrix2D(centre, -rot_deg, inv_scale)
    derotated = cv2.warpAffine(target, rot_matrix, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

    if refine_translation:
        (dx, dy), _ = cv2.phaseCorrelate(np.float32(master), np.float32(derotated))
        T = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned = cv2.warpAffine(derotated, T, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        # Compose the two affines: T ∘ rot_matrix.
        full = _compose_affines(T, rot_matrix)
        return AlignmentResult(aligned=aligned, warp_matrix=full,
                               method="logpolar+phase",
                               shift=(float(dx), float(dy)),
                               rotation_deg=float(rot_deg), scale=float(scale))

    return AlignmentResult(aligned=derotated, warp_matrix=rot_matrix,
                           method="logpolar", shift=(0.0, 0.0),
                           rotation_deg=float(rot_deg), scale=float(scale))


def _highpass_filter(spectrum: np.ndarray) -> np.ndarray:
    """Cosine high-pass commonly used together with log-polar registration."""
    h, w = spectrum.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    eta = np.cos(np.pi * (xx / (w - 1) - 0.5)) * np.cos(np.pi * (yy / (h - 1) - 0.5))
    return spectrum * (1.0 - eta) * (2.0 - eta)


def _compose_affines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the 2x3 affine equivalent to applying ``b`` then ``a``."""
    a3 = np.vstack([a, [0.0, 0.0, 1.0]])
    b3 = np.vstack([b, [0.0, 0.0, 1.0]])
    return (a3 @ b3)[:2, :].astype(np.float32)
