"""Master + tolerance map builder from a set of known-good images."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .alignment import align_ecc, align_translation
from .photometric import PhotometricCorrector
from .roi import RoiConfig, auto_part_roi
from .utils import get_logger, load_gray, stack_images


@dataclass
class Reference:
    master: np.ndarray       # float32, H x W — per-pixel median
    tolerance: np.ndarray    # float32, H x W — per-pixel std (or MAD-derived)
    method: str              # "std" or "mad"
    n_samples: int
    photometric: PhotometricCorrector = field(default_factory=lambda: PhotometricCorrector())
    roi: RoiConfig = field(default_factory=lambda: RoiConfig())
    roi_mask: np.ndarray | None = None    # uint8 HxW or None when method='none'

    @property
    def shape(self) -> tuple[int, int]:
        return self.master.shape  # type: ignore[return-value]


class ReferenceBuilder:
    """Build a Reference from multiple known-good images.

    The build process:
    1. Load images as grayscale.
    2. Optionally align each one to the first sample (phase correlation) so
       residual translation does not inflate the tolerance map.
    3. Apply a light Gaussian blur to suppress sensor noise.
    4. Compute per-pixel median (master) and per-pixel dispersion (tolerance).

    The dispersion can be either standard deviation (`std`, fast and standard)
    or `1.4826 * MAD` (robust against a few mislabeled or contaminated samples).
    """

    def __init__(self,
                 blur_ksize: int = 5,
                 align: bool = True,
                 align_method: str = "phase",
                 dispersion: str = "std",
                 photometric: PhotometricCorrector | None = None,
                 roi: RoiConfig | None = None):
        if blur_ksize % 2 == 0 or blur_ksize < 1:
            raise ValueError("blur_ksize must be a positive odd integer")
        if dispersion not in {"std", "mad"}:
            raise ValueError("dispersion must be 'std' or 'mad'")
        if align_method not in {"phase", "phase+ecc"}:
            raise ValueError("align_method must be 'phase' | 'phase+ecc'")
        self.blur_ksize = blur_ksize
        self.align = align
        self.align_method = align_method
        self.dispersion = dispersion
        self.photometric = photometric or PhotometricCorrector(method="none")
        self.roi = roi or RoiConfig(method="none")
        self.log = get_logger()

    # ---------- public ---------------------------------------------------

    def from_paths(self, paths: Iterable[str | Path]) -> Reference:
        imgs = []
        for p in paths:
            img = load_gray(p)
            imgs.append(img)
        if not imgs:
            raise ValueError("no images provided")
        return self.from_images(imgs)

    def from_images(self, images: list[np.ndarray]) -> Reference:
        if len(images) < 2:
            self.log.warning("only %d sample(s) — tolerance map will be degenerate",
                             len(images))

        prepped = [self._preprocess(img) for img in images]

        if self.align and len(prepped) >= 2:
            anchor = prepped[0]
            aligned = [anchor]
            for i, img in enumerate(prepped[1:], start=1):
                phase = align_translation(anchor, img)
                if self.align_method == "phase":
                    aligned.append(phase.aligned)
                    self.log.debug("sample %d shift=(%.3f, %.3f)",
                                   i, *phase.shift)
                else:   # "phase+ecc" — robust against cross-session drift
                    init = np.float32([[1, 0, -phase.shift[0]],
                                       [0, 1, -phase.shift[1]]])
                    ecc = align_ecc(anchor, img, motion="euclidean",
                                    init=init)
                    aligned.append(ecc.aligned)
                    self.log.debug("sample %d phase=(%.3f,%.3f) ecc=(%.3f,%.3f)",
                                   i, *phase.shift, *ecc.shift)
            prepped = aligned

        stack = stack_images(prepped).astype(np.float32)
        master = np.median(stack, axis=0).astype(np.float32)
        tolerance = self._dispersion(stack, master)

        # ROI extraction works best on the RAW (pre-photometric) median: the
        # flat-field / top-hat normalisers compress the part/background
        # dynamic range, which collapses Otsu's bimodal histogram.
        if self.roi.method != "none":
            raw_imgs = [self._raw_for_roi(img) for img in images]
            raw_stack = stack_images(raw_imgs).astype(np.float32)
            raw_master = np.median(raw_stack, axis=0).astype(np.float32)
            roi_mask = auto_part_roi(raw_master, self.roi)
        else:
            roi_mask = None
        roi_frac = (float((roi_mask > 0).sum()) / roi_mask.size
                    if roi_mask is not None else 1.0)
        self.log.info("built reference: shape=%s, n=%d, dispersion=%s, photometric=%s, "
                      "roi=%s (%.1f%% of frame)",
                      master.shape, len(images), self.dispersion,
                      self.photometric.method, self.roi.method, roi_frac * 100)
        return Reference(master=master, tolerance=tolerance,
                         method=self.dispersion, n_samples=len(images),
                         photometric=self.photometric,
                         roi=self.roi, roi_mask=roi_mask)

    # ---------- internals ------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Photometric normalization runs BEFORE the noise blur so the blur
        # smooths over any small artifacts that the normalizer introduces.
        if self.photometric.method != "none":
            img = self.photometric.apply(img)
        if self.blur_ksize > 1:
            img = cv2.GaussianBlur(img, (self.blur_ksize, self.blur_ksize), 0)
        return img

    def _raw_for_roi(self, img: np.ndarray) -> np.ndarray:
        """Photometric-free preprocessed image used only for ROI extraction.
        Same gray + light-blur path as ``_preprocess``, minus the photometric
        step that would crush the part/background contrast."""
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize > 1:
            img = cv2.GaussianBlur(img, (self.blur_ksize, self.blur_ksize), 0)
        return img

    def _dispersion(self, stack: np.ndarray, master: np.ndarray) -> np.ndarray:
        if self.dispersion == "std":
            return np.std(stack, axis=0).astype(np.float32)
        # MAD: median absolute deviation, scaled to match std of a normal dist.
        mad = np.median(np.abs(stack - master[None, ...]), axis=0)
        return (1.4826 * mad).astype(np.float32)
