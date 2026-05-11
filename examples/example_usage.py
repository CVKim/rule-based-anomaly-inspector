"""End-to-end smoke demo on synthetic data.

Generates a clean checkerboard 'good' set under simulated illumination drift,
then plants three different defect types (dent, spot, scratch) on a copy that
has *also* been rotated by 2.5°.  Runs the v0.2 pipeline with photometric
flat-field correction, log-polar rotation alignment, and shape classification.

Run from project root:
    python examples/example_usage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_inspector import (  # noqa: E402
    DynamicToleranceInspector, PhotometricCorrector, ReferenceBuilder,
)
from anomaly_inspector.visualization import side_by_side  # noqa: E402


def make_checkerboard(h: int = 256, w: int = 256, cell: int = 32) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                img[y:y + cell, x:x + cell] = 200
            else:
                img[y:y + cell, x:x + cell] = 60
    return img


def add_illumination_drift(img: np.ndarray) -> np.ndarray:
    """Multiply by a smooth diagonal ramp so the panel is brighter top-left
    than bottom-right — mimics an off-axis ring light."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = 0.7 + 0.6 * (1.0 - (xx + yy) / (h + w))
    return np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REPLICATE)


def main() -> None:
    rng = np.random.default_rng(42)
    base = make_checkerboard()

    # 10 'good' samples with mild noise, ~1 px jitter, and illumination drift
    normals: list[np.ndarray] = []
    for _ in range(10):
        noisy = np.clip(base + rng.normal(0, 3, base.shape),
                        0, 255).astype(np.uint8)
        dx = int(rng.integers(-1, 2))
        dy = int(rng.integers(-1, 2))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(noisy, M, (noisy.shape[1], noisy.shape[0]),
                                 borderMode=cv2.BORDER_REPLICATE)
        normals.append(add_illumination_drift(shifted))

    builder = ReferenceBuilder(
        blur_ksize=5, align=True, dispersion="mad",
        photometric=PhotometricCorrector(method="flat_field", sigma=41.0),
    )
    ref = builder.from_images(normals)

    # Defective sample: dark dent + bright spot + diagonal scratch, rotated 2.5°
    defective = base.copy()
    cv2.circle(defective, (90, 110), 7, 30, thickness=-1)        # dent (dark)
    cv2.circle(defective, (180, 80), 4, 240, thickness=-1)       # spot (bright)
    cv2.line(defective, (40, 40), (90, 200), 240, thickness=2)   # scratch
    defective = add_illumination_drift(defective)
    defective = rotate(defective, 2.5)
    defective = np.clip(defective + rng.normal(0, 3, defective.shape),
                        0, 255).astype(np.uint8)

    inspector = DynamicToleranceInspector(
        ref, k_sigma=4.0, base_tolerance=5.0, min_blob_area=10,
        align_method="logpolar+phase",
        auto_ignore_percentile=99.0,
    )
    result = inspector.inspect(defective)

    out_dir = ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "panel.png"), side_by_side(result, ref.master))
    cv2.imwrite(str(out_dir / "mask.png"), result.anomaly_mask)

    print(f"found {len(result.defects)} defect(s); "
          f"shift={result.shift}  rotation={result.rotation_deg:.2f}°  "
          f"scale={result.scale:.4f}")
    for i, d in enumerate(result.defects, start=1):
        print(f"  [{i}] {d.category}/{d.polarity:6}  bbox={d.bbox}  "
              f"area={d.area:4d}  aspect={d.aspect_ratio:.1f}  "
              f"circ={d.circularity:.2f}  max_diff={d.max_diff:.1f}")
    print(f"overlays written to {out_dir}")


if __name__ == "__main__":
    main()
