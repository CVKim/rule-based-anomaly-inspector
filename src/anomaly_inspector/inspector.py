"""Dynamic-tolerance defect inspector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .alignment import align_ecc, align_log_polar, align_translation
from .classification import (
    Category, ShapeFeatures, auto_unreliable_mask, classify, shape_features,
)
from .photometric import PhotometricCorrector
from .reference import Reference
from .residual import ResidualConfig, compute_residual
from .utils import get_logger


@dataclass
class DefectInfo:
    bbox: tuple[int, int, int, int]    # x, y, w, h
    area: int
    centroid: tuple[float, float]
    mean_diff: float
    max_diff: float
    category: Category = "unknown"
    polarity: str = "dark"             # "dark" | "bright"
    aspect_ratio: float = 1.0
    circularity: float = 0.0
    solidity: float = 0.0


@dataclass
class InspectionResult:
    aligned: np.ndarray            # uint8, aligned target
    diff: np.ndarray               # float32, |target - master|
    threshold_map: np.ndarray      # float32, per-pixel threshold
    anomaly_mask: np.ndarray       # uint8, 0/255
    defects: list[DefectInfo] = field(default_factory=list)
    shift: tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0
    scale: float = 1.0
    align_method: str = "phase"

    @property
    def is_defective(self) -> bool:
        return len(self.defects) > 0


class DynamicToleranceInspector:
    """Pixel-level rule-based anomaly inspector.

    Parameters
    ----------
    k_sigma:
        Tolerance multiplier on the per-pixel dispersion map.
    base_tolerance:
        Floor added to every pixel's threshold (in gray-level units). Prevents
        regions where dispersion ~= 0 from over-detecting against tiny lighting
        ripples.
    min_blob_area:
        Connected components smaller than this many pixels are dropped.
    blur_ksize:
        Gaussian blur kernel applied to the target before differencing. Must
        match what the reference was built with.
    align_method:
        "phase" (translation only, fast) or "phase+ecc" (refine with euclidean
        ECC after phase correlation). "none" disables alignment.
    morph_ksize:
        Kernel size for open + close.
    """

    def __init__(self,
                 reference: Reference,
                 k_sigma: float = 4.0,
                 base_tolerance: float = 5.0,
                 min_blob_area: int = 30,
                 blur_ksize: int = 5,
                 align_method: str = "phase",
                 morph_ksize: int = 3,
                 photometric: PhotometricCorrector | None = None,
                 k_sigma_dark: float | None = None,
                 k_sigma_bright: float | None = None,
                 base_tolerance_dark: float | None = None,
                 base_tolerance_bright: float | None = None,
                 auto_ignore_percentile: float | None = None,
                 classify_defects: bool = True,
                 residual: ResidualConfig | None = None):
        if k_sigma <= 0:
            raise ValueError("k_sigma must be > 0")
        if base_tolerance < 0:
            raise ValueError("base_tolerance must be >= 0")
        if align_method not in {"none", "phase", "phase+ecc",
                                "logpolar", "logpolar+phase"}:
            raise ValueError(f"unknown align_method '{align_method}'")
        if blur_ksize % 2 == 0 or blur_ksize < 1:
            raise ValueError("blur_ksize must be a positive odd integer")
        if morph_ksize < 1:
            raise ValueError("morph_ksize must be >= 1")
        if auto_ignore_percentile is not None and not (
                0.0 < auto_ignore_percentile < 100.0):
            raise ValueError("auto_ignore_percentile must be in (0, 100)")

        self.ref = reference
        self.k_sigma = float(k_sigma)
        self.base_tolerance = float(base_tolerance)
        self.min_blob_area = int(min_blob_area)
        self.blur_ksize = int(blur_ksize)
        self.align_method = align_method
        self.morph_ksize = int(morph_ksize)
        # Asymmetric thresholds: tighten the side that matters more.  When
        # not given, both default back to k_sigma so behaviour matches v0.1.
        self.k_sigma_dark = float(k_sigma_dark) if k_sigma_dark is not None else self.k_sigma
        self.k_sigma_bright = float(k_sigma_bright) if k_sigma_bright is not None else self.k_sigma
        self.base_tolerance_dark = (
            float(base_tolerance_dark) if base_tolerance_dark is not None
            else self.base_tolerance
        )
        self.base_tolerance_bright = (
            float(base_tolerance_bright) if base_tolerance_bright is not None
            else self.base_tolerance
        )
        self.auto_ignore_percentile = (
            float(auto_ignore_percentile)
            if auto_ignore_percentile is not None else None
        )
        self.classify_defects = bool(classify_defects)
        self.residual_config = residual or ResidualConfig()
        self._auto_ignore_cache: np.ndarray | None = None
        # Default to whatever photometric normalizer the reference was built
        # with — they MUST match, or the master and the target will live in
        # different brightness spaces.
        self.photometric = photometric or reference.photometric
        if self.photometric.method != reference.photometric.method:
            get_logger().warning(
                "photometric method on inspector (%s) differs from reference (%s); "
                "this will likely inflate the diff map",
                self.photometric.method, reference.photometric.method,
            )
        self.log = get_logger()

    # ---------- public ---------------------------------------------------

    def inspect(self, target: np.ndarray,
                ignore_mask: Optional[np.ndarray] = None) -> InspectionResult:
        """Run the full inspection pipeline on a single grayscale image.

        Parameters
        ----------
        target:
            HxW uint8 grayscale image, same shape as the reference master.
        ignore_mask:
            Optional uint8 mask the same shape as `target`. Any non-zero pixel
            is excluded from the anomaly mask (useful for stamped text,
            barcodes, fiducials, etc.).
        """
        self._validate(target, ignore_mask)

        prepared = self._preprocess(target)
        aligned, shift, method, rotation_deg, scale = self._align(prepared)

        signed, diff = compute_residual(self.ref.master, aligned.astype(np.float32),
                                        self.residual_config)

        # Symmetric threshold (used for the headline ``threshold_map`` field)
        # plus split bright/dark thresholds applied during binarization.
        threshold_map = self.base_tolerance + self.k_sigma * self.ref.tolerance
        bright_thresh = self.base_tolerance_bright + self.k_sigma_bright * self.ref.tolerance
        dark_thresh = self.base_tolerance_dark + self.k_sigma_dark * self.ref.tolerance
        # ``signed`` only carries true polarity for absdiff/multiscale modes;
        # for ncc/gradient it equals ``diff`` (always >= 0), which makes the
        # ``-signed > dark_thresh`` branch never fire — the bright threshold
        # then acts as a single magnitude check, the right behaviour for
        # sign-less residuals.
        bright_hits = signed > bright_thresh
        dark_hits = -signed > dark_thresh
        anomaly_mask = ((bright_hits | dark_hits).astype(np.uint8)) * 255

        full_ignore = self._combined_ignore(ignore_mask, target.shape)
        if full_ignore is not None:
            inverse = cv2.bitwise_not(full_ignore)
            anomaly_mask = cv2.bitwise_and(anomaly_mask, inverse)

        anomaly_mask = self._morph(anomaly_mask)
        defects = self._blobs(anomaly_mask, diff, signed)

        return InspectionResult(
            aligned=aligned, diff=diff, threshold_map=threshold_map,
            anomaly_mask=anomaly_mask, defects=defects,
            shift=shift, rotation_deg=rotation_deg, scale=scale,
            align_method=method,
        )

    # ---------- internals ------------------------------------------------

    def _validate(self, target: np.ndarray,
                  ignore_mask: Optional[np.ndarray]) -> None:
        if target.ndim != 2:
            raise ValueError(f"target must be 2D grayscale, got shape {target.shape}")
        if target.shape != self.ref.shape:
            raise ValueError(
                f"target shape {target.shape} != reference shape {self.ref.shape}"
            )
        if ignore_mask is not None and ignore_mask.shape != target.shape:
            raise ValueError(
                f"ignore_mask shape {ignore_mask.shape} != target shape {target.shape}"
            )

    def _preprocess(self, target: np.ndarray) -> np.ndarray:
        if self.photometric.method != "none":
            target = self.photometric.apply(target)
        if self.blur_ksize > 1:
            return cv2.GaussianBlur(target, (self.blur_ksize, self.blur_ksize), 0)
        return target

    def _align(self, target: np.ndarray) -> tuple[np.ndarray, tuple[float, float],
                                                   str, float, float]:
        if self.align_method == "none":
            return target, (0.0, 0.0), "none", 0.0, 1.0

        if self.align_method in {"logpolar", "logpolar+phase"}:
            refine = self.align_method == "logpolar+phase"
            res = align_log_polar(self.ref.master, target,
                                  refine_translation=refine)
            return (res.aligned, res.shift, res.method,
                    res.rotation_deg, res.scale)

        phase = align_translation(self.ref.master, target)
        if self.align_method == "phase":
            return phase.aligned, phase.shift, phase.method, 0.0, 1.0

        # phase + ecc
        # Use the translation result to seed the ECC warp matrix.
        init = np.float32([[1, 0, -phase.shift[0]],
                           [0, 1, -phase.shift[1]]])
        ecc = align_ecc(self.ref.master, target, motion="euclidean", init=init)
        return ecc.aligned, ecc.shift, f"phase+{ecc.method}", 0.0, 1.0

    def _morph(self, mask: np.ndarray) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (self.morph_ksize, self.morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _blobs(self, mask: np.ndarray, diff: np.ndarray,
               signed: np.ndarray) -> list[DefectInfo]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        out: list[DefectInfo] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_blob_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            region = diff[y:y + h, x:x + w]
            label_region = labels[y:y + h, x:x + w] == i
            vals = region[label_region]
            mean_diff = float(vals.mean()) if vals.size else 0.0
            max_diff = float(vals.max()) if vals.size else 0.0

            category: Category = "unknown"
            polarity = "dark"
            aspect = 1.0
            circ = 0.0
            solidity = 0.0
            if self.classify_defects:
                blob_mask = ((labels == i).astype(np.uint8)) * 255
                feats = shape_features(blob_mask, signed)
                category = classify(feats)
                polarity = feats.polarity
                aspect = feats.aspect_ratio
                circ = feats.circularity
                solidity = feats.solidity

            out.append(DefectInfo(
                bbox=(x, y, w, h), area=area,
                centroid=(float(centroids[i, 0]), float(centroids[i, 1])),
                mean_diff=mean_diff, max_diff=max_diff,
                category=category, polarity=polarity,
                aspect_ratio=aspect, circularity=circ, solidity=solidity,
            ))
        return out

    @staticmethod
    def _binarize_mask(mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        return bin_mask

    def _combined_ignore(self, user_mask: Optional[np.ndarray],
                         shape: tuple[int, int]) -> Optional[np.ndarray]:
        """OR all ignore-source masks: user-supplied, auto-unreliable
        (high-tolerance percentile), and out-of-ROI (everything outside the
        part region recorded in the reference)."""
        masks: list[np.ndarray] = []
        if user_mask is not None:
            masks.append(self._binarize_mask(user_mask))
        if self.auto_ignore_percentile is not None:
            if self._auto_ignore_cache is None:
                self._auto_ignore_cache = auto_unreliable_mask(
                    self.ref.tolerance,
                    percentile=self.auto_ignore_percentile,
                )
            masks.append(self._auto_ignore_cache)
        if self.ref.roi_mask is not None:
            # ROI mask = where the part *is*; the inverse is where to ignore.
            masks.append(cv2.bitwise_not(self.ref.roi_mask))
        if not masks:
            return None
        out = masks[0]
        for m in masks[1:]:
            out = cv2.bitwise_or(out, m)
        return out
