"""Master + tolerance map builder from a set of known-good images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .alignment import align_translation
from .utils import get_logger, load_gray, stack_images


@dataclass
class Reference:
    master: np.ndarray       # float32, H x W — per-pixel median
    tolerance: np.ndarray    # float32, H x W — per-pixel std (or MAD-derived)
    method: str              # "std" or "mad"
    n_samples: int

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
                 dispersion: str = "std"):
        if blur_ksize % 2 == 0 or blur_ksize < 1:
            raise ValueError("blur_ksize must be a positive odd integer")
        if dispersion not in {"std", "mad"}:
            raise ValueError("dispersion must be 'std' or 'mad'")
        self.blur_ksize = blur_ksize
        self.align = align
        self.dispersion = dispersion
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
                res = align_translation(anchor, img)
                aligned.append(res.aligned)
                self.log.debug("sample %d shift=(%.3f, %.3f)", i, *res.shift)
            prepped = aligned

        stack = stack_images(prepped).astype(np.float32)
        master = np.median(stack, axis=0).astype(np.float32)
        tolerance = self._dispersion(stack, master)

        self.log.info("built reference: shape=%s, n=%d, method=%s",
                      master.shape, len(images), self.dispersion)
        return Reference(master=master, tolerance=tolerance,
                         method=self.dispersion, n_samples=len(images))

    # ---------- internals ------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
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
