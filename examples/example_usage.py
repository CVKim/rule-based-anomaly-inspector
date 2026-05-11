"""End-to-end smoke demo on synthetic data.

Generates a clean checkerboard 'good' set, then plants a small dark blob on a
copy and inspects it. Produces `examples/output/` overlays so you can eyeball
the pipeline without real fab data.

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

from anomaly_inspector import DynamicToleranceInspector, ReferenceBuilder  # noqa: E402
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


def main() -> None:
    rng = np.random.default_rng(42)
    base = make_checkerboard()

    # 10 'good' samples with mild noise and ~1 px translation jitter
    normals: list[np.ndarray] = []
    for _ in range(10):
        noisy = base + rng.normal(0, 3, base.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        dx = int(rng.integers(-1, 2))
        dy = int(rng.integers(-1, 2))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(noisy, M, (noisy.shape[1], noisy.shape[0]),
                                 borderMode=cv2.BORDER_REPLICATE)
        normals.append(shifted)

    # Build reference
    builder = ReferenceBuilder(blur_ksize=5, align=True, dispersion="std")
    ref = builder.from_images(normals)

    # Defective sample: clean image + dark circular blob + small bright scratch
    defective = base.copy()
    cv2.circle(defective, (110, 130), 7, 30, thickness=-1)
    cv2.line(defective, (180, 60), (200, 75), 240, thickness=2)
    defective = np.clip(defective + rng.normal(0, 3, defective.shape),
                        0, 255).astype(np.uint8)

    inspector = DynamicToleranceInspector(
        ref, k_sigma=4.0, base_tolerance=5.0, min_blob_area=10,
        blur_ksize=5, align_method="phase",
    )
    result = inspector.inspect(defective)

    out_dir = ROOT / "examples" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "panel.png"), side_by_side(result, ref.master))
    cv2.imwrite(str(out_dir / "mask.png"), result.anomaly_mask)

    print(f"found {len(result.defects)} defect(s); shift={result.shift}")
    for i, d in enumerate(result.defects, start=1):
        print(f"  [{i}] bbox={d.bbox} area={d.area} max_diff={d.max_diff:.1f}")
    print(f"overlays written to {out_dir}")


if __name__ == "__main__":
    main()
