"""Photometric (illumination) normalization.

Industrial vision rigs — especially rotary stages with off-axis ring lights —
rarely produce a perfectly flat illumination field. Even a small low-frequency
brightness drift across the panel will dominate the per-pixel difference and
either drown the real defects or force the user to crank ``k_sigma`` so high
that subtle defects disappear.

This module provides three rule-based normalizers that all run on uint8 (or
float32) grayscale images and return uint8 in the same dynamic range:

* ``flat_field_divide``  — divide by a heavily Gaussian-blurred copy. The blur
  estimates the slowly-varying illumination; division flattens it.
* ``top_hat``            — morphological top-hat. Subtracts a structural opening
  (white top-hat) or closing (black top-hat) of the image to suppress
  large-scale background variations while keeping small high-contrast features.
* ``clahe``              — Contrast Limited Adaptive Histogram Equalization.
  Useful when both brightness and contrast drift across the field.

The same ``PhotometricCorrector`` instance is used at reference-build time and
at inspect time so the master and the target see identical normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


Method = Literal["none", "flat_field", "top_hat_white", "top_hat_black", "clahe"]
_VALID: tuple[Method, ...] = ("none", "flat_field", "top_hat_white",
                              "top_hat_black", "clahe")


@dataclass(frozen=True)
class PhotometricCorrector:
    """Stateless normalizer — describes which correction to apply.

    Parameters
    ----------
    method:
        Which normalization to apply.  ``"none"`` is a passthrough.
    sigma:
        Gaussian sigma used by ``flat_field``.  Should be large compared to the
        typical defect size so the blur represents illumination, not structure.
    ksize:
        Structuring-element size for the morphological top-hat.  Should also be
        clearly larger than the defect to be detected.
    clip_limit, tile_grid:
        CLAHE parameters.  ``clip_limit`` ~2-4 is a good starting point.
    """

    method: Method = "none"
    sigma: float = 31.0
    ksize: int = 51
    clip_limit: float = 2.0
    tile_grid: int = 8

    def __post_init__(self) -> None:
        if self.method not in _VALID:
            raise ValueError(f"unknown method '{self.method}', expected one of {_VALID}")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self.ksize < 3 or self.ksize % 2 == 0:
            raise ValueError("ksize must be an odd integer >= 3")
        if self.clip_limit <= 0:
            raise ValueError("clip_limit must be > 0")
        if self.tile_grid < 1:
            raise ValueError("tile_grid must be >= 1")

    def apply(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 2:
            raise ValueError(f"expected 2D grayscale, got shape {img.shape}")

        if self.method == "none":
            return img if img.dtype == np.uint8 else _to_uint8(img)
        if self.method == "flat_field":
            return flat_field_divide(img, sigma=self.sigma)
        if self.method == "top_hat_white":
            return top_hat(img, ksize=self.ksize, polarity="white")
        if self.method == "top_hat_black":
            return top_hat(img, ksize=self.ksize, polarity="black")
        if self.method == "clahe":
            return clahe(img, clip_limit=self.clip_limit, tile_grid=self.tile_grid)
        raise AssertionError(f"unhandled method '{self.method}'")  # pragma: no cover

    def to_meta(self) -> dict:
        """Serializable description so the reference can store it."""
        return {
            "method": self.method,
            "sigma": float(self.sigma),
            "ksize": int(self.ksize),
            "clip_limit": float(self.clip_limit),
            "tile_grid": int(self.tile_grid),
        }

    @classmethod
    def from_meta(cls, meta: dict | None) -> "PhotometricCorrector":
        if not meta:
            return cls(method="none")
        return cls(
            method=meta.get("method", "none"),
            sigma=float(meta.get("sigma", 31.0)),
            ksize=int(meta.get("ksize", 51)),
            clip_limit=float(meta.get("clip_limit", 2.0)),
            tile_grid=int(meta.get("tile_grid", 8)),
        )


# ---------- functional API ---------------------------------------------------


def flat_field_divide(img: np.ndarray, sigma: float = 31.0,
                      eps: float = 1.0) -> np.ndarray:
    """Divide-by-blur flat-field correction.

    Estimates the illumination field as a heavy Gaussian low-pass of ``img``,
    then ``corrected = img / illum * mean(illum)``.  Result is rescaled back to
    the original mean so downstream thresholds are interpretable in 8-bit units.
    """
    f = img.astype(np.float32)
    illum = cv2.GaussianBlur(f, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma,
                             borderType=cv2.BORDER_REFLECT)
    illum = np.maximum(illum, eps)
    target_mean = float(illum.mean())
    corrected = f / illum * target_mean
    return _to_uint8(corrected)


def top_hat(img: np.ndarray, ksize: int = 51,
            polarity: Literal["white", "black"] = "white") -> np.ndarray:
    """Morphological top-hat.

    ``polarity="white"`` highlights bright features on a dark background
    (``img - opening``).  ``polarity="black"`` highlights dark features on a
    bright background (``closing - img``).

    Practically: pick whichever matches the *defect* polarity for the strongest
    contrast.  For a generic "anything that deviates" workflow, prefer the
    flat-field normalizer instead.
    """
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("ksize must be an odd integer >= 3")
    src = img if img.dtype == np.uint8 else _to_uint8(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if polarity == "white":
        return cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
    if polarity == "black":
        return cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
    raise ValueError(f"polarity must be 'white' or 'black', got '{polarity}'")


def clahe(img: np.ndarray, clip_limit: float = 2.0,
          tile_grid: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization."""
    src = img if img.dtype == np.uint8 else _to_uint8(img)
    op = cv2.createCLAHE(clipLimit=float(clip_limit),
                         tileGridSize=(int(tile_grid), int(tile_grid)))
    return op.apply(src)


# ---------- internals --------------------------------------------------------


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Clip to [0, 255] and cast.  Preserves dtype semantics for downstream."""
    return np.clip(img, 0.0, 255.0).astype(np.uint8)
