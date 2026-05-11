"""Sweep N residual algorithms across a folder of test images.

Builds one shared reference from a folder of known-good images, then for each
selected residual mode (``absdiff``, ``multiscale``, ``ncc``, ``gradient``)
runs the full inspection pipeline on every test image and writes a six-cell
panel (image | heatmap | mask pred | pred conf fg | pred conf bg | overlay)
plus a per-mode ``summary.csv``.

Output structure:

    <output_root>/
        absdiff/
            <stem>_panel.png
            summary.csv
        multiscale/
        ncc/
        gradient/

The script handles non-ASCII Windows paths (Korean, Japanese, ...) and very
large source images: panels are downscaled per cell to keep total disk usage
under ~5 MB per source image.

Usage:
    python scripts/run_inference_panels.py \
        --normal "H:/.../FS_측면/10회 반복" \
        --test   "H:/.../FS_측면" \
        --output outputs/foosung_side \
        --modes  absdiff multiscale ncc gradient
"""

from __future__ import annotations

# IMPORTANT: scrub the script's own folder out of sys.path *before* any other
# imports — otherwise our sibling ``scripts/inspect.py`` shadows the
# stdlib ``inspect`` module, breaking numpy/dataclasses on import.
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]

import argparse  # noqa: E402
import time      # noqa: E402

import cv2       # noqa: E402
import numpy as np  # noqa: E402

ROOT = _HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from anomaly_inspector import (  # noqa: E402
    DynamicToleranceInspector, PhotometricCorrector, ReferenceBuilder,
    ResidualConfig, make_panel,
)
from anomaly_inspector.utils import (  # noqa: E402
    SUPPORTED_EXTS, ensure_dir, get_logger, imread_unicode, imwrite_unicode,
)


SUPPORTED = SUPPORTED_EXTS


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED)


def load_gray_unicode(path: Path) -> np.ndarray:
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"failed to read {path}")
    return img


def make_inspector(ref, mode: str, args) -> DynamicToleranceInspector:
    res_cfg = ResidualConfig(
        mode=mode,                            # type: ignore[arg-type]
        pyramid_levels=args.pyramid_levels,
        ncc_window=args.ncc_window,
        gradient_op=args.gradient_op,         # type: ignore[arg-type]
        gradient_blend=args.gradient_blend,
    )
    return DynamicToleranceInspector(
        ref,
        k_sigma=args.k_sigma,
        base_tolerance=args.base_tolerance,
        min_blob_area=args.min_blob_area,
        blur_ksize=args.blur_ksize,
        align_method=args.align_method,
        morph_ksize=args.morph_ksize,
        auto_ignore_percentile=args.auto_ignore_percentile,
        residual=res_cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--normal", required=True, type=Path,
                        help="Folder of known-good (normal) images.")
    parser.add_argument("--test", required=True, type=Path,
                        help="Folder of images to inspect.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output root directory; one subdir per mode.")
    parser.add_argument("--modes", nargs="+",
                        default=["absdiff", "multiscale", "ncc", "gradient"],
                        choices=["absdiff", "multiscale", "ncc", "gradient"],
                        help="Residual modes to sweep.")

    # Reference build
    parser.add_argument("--photometric", default="flat_field",
                        choices=["none", "flat_field", "top_hat_white",
                                 "top_hat_black", "clahe"])
    parser.add_argument("--photo-sigma", type=float, default=51.0,
                        help="flat_field sigma — should be >> defect size.")
    parser.add_argument("--photo-ksize", type=int, default=51,
                        help="top-hat structuring-element size.")
    parser.add_argument("--dispersion", default="mad", choices=["std", "mad"])
    parser.add_argument("--ref-blur", type=int, default=5)
    parser.add_argument("--ref-align", dest="ref_align",
                        action="store_true", default=True,
                        help="Phase-correlate normals to the first sample (default).")
    parser.add_argument("--no-ref-align", dest="ref_align", action="store_false")

    # Inspect
    parser.add_argument("--k-sigma", type=float, default=4.0)
    parser.add_argument("--base-tolerance", type=float, default=8.0)
    parser.add_argument("--min-blob-area", type=int, default=80)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--align-method", default="phase",
                        choices=["none", "phase", "phase+ecc",
                                 "logpolar", "logpolar+phase"])
    parser.add_argument("--morph-ksize", type=int, default=5)
    parser.add_argument("--auto-ignore-percentile", type=float, default=99.5)

    # Residual options
    parser.add_argument("--pyramid-levels", type=int, default=3)
    parser.add_argument("--ncc-window", type=int, default=21)
    parser.add_argument("--gradient-op", default="scharr",
                        choices=["sobel", "scharr"])
    parser.add_argument("--gradient-blend", type=float, default=0.0)

    # Speed/memory
    parser.add_argument("--max-input-width", type=int, default=2048,
                        help="Downsample huge inputs to this width before "
                             "inspecting (keeps aspect). Set to 0 to disable.")
    parser.add_argument("--max-cell-width", type=int, default=520,
                        help="Per-panel-cell display width.")

    args = parser.parse_args()

    log = get_logger()
    log.info("normal: %s", args.normal)
    log.info("test:   %s", args.test)
    log.info("modes:  %s", ", ".join(args.modes))

    if not args.normal.exists():
        raise SystemExit(f"normal folder not found: {args.normal}")
    if not args.test.exists():
        raise SystemExit(f"test folder not found: {args.test}")

    normal_paths = list_images(args.normal)
    test_paths = list_images(args.test)
    if not normal_paths:
        raise SystemExit(f"no images in {args.normal}")
    if not test_paths:
        raise SystemExit(f"no images in {args.test}")
    log.info("loaded %d normal images, %d test images",
             len(normal_paths), len(test_paths))

    # ------------------------------------------------------------------
    # Reference build (shared across all modes)
    # ------------------------------------------------------------------
    log.info("loading normals...")
    normals: list[np.ndarray] = []
    target_shape: tuple[int, int] | None = None
    for p in normal_paths:
        img = load_gray_unicode(p)
        if args.max_input_width and img.shape[1] > args.max_input_width:
            scale = args.max_input_width / img.shape[1]
            new_h = int(round(img.shape[0] * scale))
            img = cv2.resize(img, (args.max_input_width, new_h),
                             interpolation=cv2.INTER_AREA)
        if target_shape is None:
            target_shape = img.shape
        elif img.shape != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]),
                             interpolation=cv2.INTER_AREA)
        normals.append(img)
    log.info("normals shape: %s", target_shape)

    photometric = PhotometricCorrector(
        method=args.photometric,                    # type: ignore[arg-type]
        sigma=args.photo_sigma,
        ksize=args.photo_ksize if args.photo_ksize % 2 else args.photo_ksize + 1,
    )
    builder = ReferenceBuilder(
        blur_ksize=args.ref_blur,
        align=args.ref_align,
        dispersion=args.dispersion,
        photometric=photometric,
    )
    t0 = time.time()
    ref = builder.from_images(normals)
    log.info("reference built in %.2fs (n=%d, dispersion=%s, photometric=%s)",
             time.time() - t0, ref.n_samples, ref.method,
             photometric.method)

    # ------------------------------------------------------------------
    # Run each mode against every test image
    # ------------------------------------------------------------------
    out_root = ensure_dir(args.output)
    summary_columns = (
        "filename,n_defects,shift_x,shift_y,rotation_deg,scale,"
        "categories,inference_ms"
    )

    for mode in args.modes:
        mode_dir = ensure_dir(out_root / mode)
        log.info("=== mode: %s ===", mode)
        inspector = make_inspector(ref, mode, args)
        rows = [summary_columns]

        for p in test_paths:
            target = load_gray_unicode(p)
            if target.shape != target_shape:
                target = cv2.resize(target, (target_shape[1], target_shape[0]),
                                    interpolation=cv2.INTER_AREA)

            t1 = time.time()
            result = inspector.inspect(target)
            ms = (time.time() - t1) * 1000.0

            cats = ";".join(f"{d.category}/{d.polarity}"
                            for d in result.defects) or "-"
            log.info("  %-22s defects=%-3d  rot=%+.2f°  scale=%.4f  "
                     "shift=(%+.2f,%+.2f)  %.0fms  %s",
                     p.name, len(result.defects),
                     result.rotation_deg, result.scale,
                     result.shift[0], result.shift[1], ms, cats)

            panel = make_panel(target, result,
                               max_cell_width=args.max_cell_width,
                               title=f"{p.name}  |  mode={mode}  "
                                     f"|  defects={len(result.defects)}  "
                                     f"|  rot={result.rotation_deg:+.2f}°")
            imwrite_unicode(mode_dir / f"{p.stem}_panel.png", panel)

            rows.append(
                f"{p.name},{len(result.defects)},"
                f"{result.shift[0]:.3f},{result.shift[1]:.3f},"
                f"{result.rotation_deg:.3f},{result.scale:.5f},"
                f"{cats},{ms:.0f}"
            )

        (mode_dir / "summary.csv").write_text(
            "\n".join(rows), encoding="utf-8")

    log.info("done — outputs in %s", out_root)


if __name__ == "__main__":
    main()
