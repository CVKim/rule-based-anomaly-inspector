"""Image alignment utilities — translation (phase correlation) and ECC."""

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
