"""GT-driven hyperparameter tuner.

For each requested residual mode, computes the residual map ONCE per image
(the expensive step), caches it, then sweeps the cheap post-processing knobs
(``k_sigma``, ``base_tolerance``, ``min_blob_area``, ``morph_ksize``) and
scores each combination against the labelme GT.

Per-mode outputs:

* ``recommend.yaml``: best F1 operating point (drop-in config block)
* ``pareto.csv``:    every combination with metrics, sortable
* ``pareto.png``:    recall-vs-FP/image scatter, the Pareto frontier in red

Usage:
    python scripts/tune.py \\
        --normal "H:/.../FS_측면/10회 반복" \\
        --test   "H:/.../FS_측면" \\
        --gt     "H:/.../2026-05-12" \\
        --output outputs/tune_v1 \\
        --modes  absdiff ridge fused \\
        --max-input-width 0
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]

import argparse  # noqa: E402
import itertools  # noqa: E402
import json      # noqa: E402
import time      # noqa: E402

import cv2       # noqa: E402
import numpy as np  # noqa: E402

ROOT = _HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from anomaly_inspector import (  # noqa: E402
    DynamicToleranceInspector, PhotometricCorrector, ReferenceBuilder,
    ResidualConfig, RoiConfig,
)
from anomaly_inspector.classification import auto_unreliable_mask  # noqa: E402
from anomaly_inspector.evaluation import (  # noqa: E402
    ModeReport, evaluate_image, load_gt_folder,
)
from anomaly_inspector.residual import compute_residual  # noqa: E402
from anomaly_inspector.utils import (  # noqa: E402
    SUPPORTED_EXTS, ensure_dir, get_logger, imread_unicode, imwrite_unicode,
)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def load_gray_unicode(path: Path) -> np.ndarray:
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"failed to read {path}")
    return img


# ---------- cached per-image residual cache --------------------------------


def cached_residual(ref, prepared_target: np.ndarray, aligned: np.ndarray,
                    mode: str, res_kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute and return (signed, abs) residual. We don't actually cache
    here because residual depends on mode + ridge/fused params; the caller
    is expected to iterate modes in the outer loop and re-call this once
    per image."""
    cfg = ResidualConfig(mode=mode, **res_kwargs)            # type: ignore[arg-type]
    return compute_residual(ref.master, aligned.astype(np.float32), cfg,
                            tolerance=ref.tolerance)


def quick_threshold_blob(signed: np.ndarray, diff: np.ndarray,
                         tolerance: np.ndarray,
                         k_sigma: float, base_tolerance: float,
                         min_blob_area: int, morph_ksize: int,
                         ignore_mask: np.ndarray | None) -> list[tuple[float, float, float, float]]:
    """The cheap downstream pipeline: per-pixel threshold -> morphology
    -> connected components -> bboxes. Symmetric thresholds (same dark /
    bright) for the tuner; asymmetric is exposed in the inspector API for
    operators who need it."""
    thresh = base_tolerance + k_sigma * tolerance
    mask = (np.abs(signed) > thresh).astype(np.uint8) * 255
    # Also fire on either polarity in case signed != diff (NCC/gradient/ridge
    # all have signed == diff so this is just defensive).
    mask |= (diff > thresh).astype(np.uint8) * 255

    if ignore_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(ignore_mask))

    if morph_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[tuple[float, float, float, float]] = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_blob_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append((float(x), float(y), float(w), float(h)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal", required=True, type=Path, nargs="+",
                        help="One or more folders of known-good images.")
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", nargs="+",
                        default=["absdiff", "ridge", "fused"],
                        choices=["absdiff", "multiscale", "ncc", "gradient",
                                 "ridge", "fused"])

    # Reference build (fixed across the sweep)
    parser.add_argument("--max-input-width", type=int, default=0,
                        help="0 = native resolution (recommended for thin cracks).")
    parser.add_argument("--photometric", default="flat_field")
    parser.add_argument("--photo-sigma", type=float, default=51.0)
    parser.add_argument("--dispersion", default="mad")
    parser.add_argument("--ref-blur", type=int, default=5)
    parser.add_argument("--roi", default="otsu_close")
    parser.add_argument("--roi-close-ksize", type=int, default=81)
    parser.add_argument("--roi-erode-px", type=int, default=12)

    # Per-image inspect-side fixed knobs
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--align-method", default="phase")
    parser.add_argument("--ncc-window", type=int, default=21)
    parser.add_argument("--pyramid-levels", type=int, default=3)
    parser.add_argument("--ridge-polarity", default="dark")
    parser.add_argument("--ridge-scales", nargs="+", type=float,
                        default=[1.5, 3.0, 5.0])
    parser.add_argument("--ridge-master-dilate", type=int, default=3)
    parser.add_argument("--fused-modes", nargs="+",
                        default=["absdiff", "ncc", "ridge"])

    # Grid over the cheap knobs
    parser.add_argument("--k-sigma-grid", nargs="+", type=float,
                        default=[2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
    parser.add_argument("--base-tol-grid", nargs="+", type=float,
                        default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--min-blob-grid", nargs="+", type=int,
                        default=[40, 80, 150, 300])
    parser.add_argument("--morph-grid", nargs="+", type=int,
                        default=[3, 5])
    parser.add_argument("--auto-ignore-grid", nargs="+", type=float,
                        default=[-1.0, 99.0, 99.5, 99.9],
                        help="-1 means 'off'")

    # Eval
    parser.add_argument("--iou-threshold", type=float, default=0.05)
    parser.add_argument("--soft-fp-distance-px", type=float, default=120.0)
    parser.add_argument("--fp-budget", type=float, default=10.0,
                        help="Pareto recommendation: best F1 among configs "
                             "with hard_FP/img <= this.")
    parser.add_argument("--polygon-iou", action="store_true",
                        help="Use polygon-IoU for matching when GT polygons "
                             "are present (slower but more accurate).")

    args = parser.parse_args()
    log = get_logger()

    # ------------------------------------------------------------------
    # GT + image lists
    # ------------------------------------------------------------------
    gt = load_gt_folder(args.gt)
    test_paths = [p for p in list_images(args.test) if p.name in gt]
    if not test_paths:
        raise SystemExit("no test images matched any GT file")
    log.info("test images: %d (NG=%d OK=%d)", len(test_paths),
             sum(1 for p in test_paths if gt[p.name].is_ng),
             sum(1 for p in test_paths if not gt[p.name].is_ng))

    # ------------------------------------------------------------------
    # Build shared reference
    # ------------------------------------------------------------------
    log.info("loading normals...")
    normal_paths: list[Path] = []
    for nd in args.normal:
        nd_paths = list_images(nd)
        log.info("  normal source %s: %d images", nd, len(nd_paths))
        normal_paths.extend(nd_paths)
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

    photo = PhotometricCorrector(method=args.photometric,                # type: ignore[arg-type]
                                 sigma=args.photo_sigma)
    roi_cfg = RoiConfig(method=args.roi, close_ksize=args.roi_close_ksize,  # type: ignore[arg-type]
                        erode_px=args.roi_erode_px)
    t0 = time.time()
    ref = ReferenceBuilder(blur_ksize=args.ref_blur, align=True,
                           dispersion=args.dispersion,
                           photometric=photo, roi=roi_cfg).from_images(normals)
    log.info("reference built in %.1fs (shape=%s, roi=%s)",
             time.time() - t0, ref.master.shape, args.roi)

    a_gt = next(iter(gt.values()))
    full_res = (a_gt.image_height, a_gt.image_width)
    pred_scale = full_res[1] / target_shape[1]

    # ------------------------------------------------------------------
    # Pre-compute per-image residual maps ONCE per mode
    # ------------------------------------------------------------------
    log.info("loading + aligning test images...")
    targets_aligned: dict[str, np.ndarray] = {}
    inspector_for_align = DynamicToleranceInspector(
        ref, align_method=args.align_method, blur_ksize=args.blur_ksize,
    )
    for p in test_paths:
        img = load_gray_unicode(p)
        if img.shape != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]),
                             interpolation=cv2.INTER_AREA)
        prep = inspector_for_align._preprocess(img)
        aligned, *_ = inspector_for_align._align(prep)
        targets_aligned[p.name] = aligned

    out_root = ensure_dir(args.output)

    # ------------------------------------------------------------------
    # Per-mode sweep
    # ------------------------------------------------------------------
    summary_rows = ["mode,k_sigma,base_tol,min_blob,morph,auto_ignore,"
                    "TP,FN,hardFP,softFP,precision,recall,F1,hardFP_per_img"]

    for mode in args.modes:
        log.info("=== mode: %s ===  computing residuals...", mode)
        res_kwargs: dict = {}
        if mode == "ridge":
            res_kwargs = dict(
                ridge_polarity=args.ridge_polarity,
                ridge_scales=tuple(args.ridge_scales),
                ridge_master_dilate=args.ridge_master_dilate,
            )
        elif mode == "ncc":
            res_kwargs = dict(ncc_window=args.ncc_window)
        elif mode == "multiscale":
            res_kwargs = dict(pyramid_levels=args.pyramid_levels)
        elif mode == "fused":
            res_kwargs = dict(
                fused_modes=tuple(args.fused_modes),
                ridge_polarity=args.ridge_polarity,
                ridge_scales=tuple(args.ridge_scales),
                ridge_master_dilate=args.ridge_master_dilate,
                ncc_window=args.ncc_window,
            )

        cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        t0 = time.time()
        for p in test_paths:
            signed, diff = cached_residual(ref, None, targets_aligned[p.name],
                                           mode, res_kwargs)
            cache[p.name] = (signed, diff)
        log.info("  residuals computed in %.1fs", time.time() - t0)

        rows: list[dict] = []
        grid = list(itertools.product(args.k_sigma_grid, args.base_tol_grid,
                                      args.min_blob_grid, args.morph_grid,
                                      args.auto_ignore_grid))
        log.info("  grid size: %d", len(grid))

        for (k_sigma, base_tol, min_blob, morph, auto_ig) in grid:
            # Build the ignore mask for this auto_ig setting
            ignore_extra: np.ndarray | None = None
            if auto_ig > 0:
                ignore_extra = auto_unreliable_mask(ref.tolerance,
                                                    percentile=auto_ig)
            if ref.roi_mask is not None:
                inv_roi = cv2.bitwise_not(ref.roi_mask)
                ignore_extra = (cv2.bitwise_or(ignore_extra, inv_roi)
                                if ignore_extra is not None else inv_roi)

            rep = ModeReport(mode=mode, iou_threshold=args.iou_threshold)
            for p in test_paths:
                signed, diff = cache[p.name]
                preds = quick_threshold_blob(
                    signed, diff, ref.tolerance,
                    k_sigma=k_sigma, base_tolerance=base_tol,
                    min_blob_area=min_blob, morph_ksize=morph,
                    ignore_mask=ignore_extra,
                )
                ev = evaluate_image(
                    pred_boxes=preds, gt=gt[p.name],
                    iou_threshold=args.iou_threshold,
                    soft_fp_max_centre_distance=args.soft_fp_distance_px,
                    roi_mask=ref.roi_mask, pred_scale=pred_scale,
                    use_polygon_iou=args.polygon_iou,
                )
                rep.per_image.append(ev)
            row = {
                "k_sigma": k_sigma, "base_tol": base_tol,
                "min_blob": min_blob, "morph": morph,
                "auto_ignore": auto_ig,
                "tp": rep.total_tp, "fn": rep.total_fn,
                "hard_fp": rep.total_hard_fp, "soft_fp": rep.total_soft_fp,
                "precision": rep.precision, "recall": rep.recall, "f1": rep.f1,
                "hard_fp_per_img": rep.hard_fp_per_image,
            }
            rows.append(row)
            summary_rows.append(
                f"{mode},{k_sigma},{base_tol},{min_blob},{morph},{auto_ig},"
                f"{rep.total_tp},{rep.total_fn},{rep.total_hard_fp},"
                f"{rep.total_soft_fp},"
                f"{_fmt(rep.precision)},{_fmt(rep.recall)},{_fmt(rep.f1)},"
                f"{rep.hard_fp_per_image:.2f}"
            )

        # Per-mode best by F1 within FP budget
        valid = [r for r in rows
                 if r["hard_fp_per_img"] <= args.fp_budget
                 and r["f1"] == r["f1"]]
        valid.sort(key=lambda r: (-r["f1"], r["hard_fp_per_img"]))
        if valid:
            best = valid[0]
        else:
            # No config fit the budget — take the highest F1 we did find.
            with_f1 = [r for r in rows if r["f1"] == r["f1"]]
            with_f1.sort(key=lambda r: -r["f1"])
            best = with_f1[0] if with_f1 else rows[0]

        log.info("  best within FP budget %.1f/img: F1=%.3f  R=%.3f  P=%.3f"
                 "  hardFP/img=%.2f  k=%.1f base=%.1f minblob=%d morph=%d ai=%.1f",
                 args.fp_budget, _f(best["f1"]), _f(best["recall"]),
                 _f(best["precision"]),
                 best["hard_fp_per_img"],
                 best["k_sigma"], best["base_tol"], best["min_blob"],
                 best["morph"], best["auto_ignore"])

        # Write per-mode recommend yaml
        rec = {
            "mode": mode,
            "k_sigma": best["k_sigma"],
            "base_tolerance": best["base_tol"],
            "min_blob_area": best["min_blob"],
            "morph_ksize": best["morph"],
            "auto_ignore_percentile": (None if best["auto_ignore"] <= 0
                                        else best["auto_ignore"]),
            "metrics": {
                "tp": best["tp"], "fn": best["fn"],
                "hard_fp": best["hard_fp"], "soft_fp": best["soft_fp"],
                "precision": _f(best["precision"]),
                "recall": _f(best["recall"]),
                "f1": _f(best["f1"]),
                "hard_fp_per_img": best["hard_fp_per_img"],
            },
        }
        mode_dir = ensure_dir(out_root / mode)
        (mode_dir / "recommend.json").write_text(
            json.dumps(rec, indent=2), encoding="utf-8")

        # Per-mode pareto.csv
        cols = ["k_sigma", "base_tol", "min_blob", "morph", "auto_ignore",
                "tp", "fn", "hard_fp", "soft_fp",
                "precision", "recall", "f1", "hard_fp_per_img"]
        with open(mode_dir / "pareto.csv", "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(
                    f"{r[c]:.4f}" if isinstance(r[c], float) and r[c] == r[c]
                    else str(r[c]) for c in cols) + "\n")

        # Per-mode Pareto plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [r["hard_fp_per_img"] for r in rows if r["f1"] == r["f1"]]
            ys = [r["recall"] for r in rows if r["f1"] == r["f1"]]
            f1s = [r["f1"] for r in rows if r["f1"] == r["f1"]]
            if xs:
                fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
                sc = ax.scatter(xs, ys, c=f1s, cmap="viridis", s=18, alpha=0.7)
                ax.scatter([best["hard_fp_per_img"]], [best["recall"]],
                           c="red", s=120, marker="*", edgecolors="black",
                           label=f"best F1={_f(best['f1']):.3f}", zorder=10)
                ax.axvline(args.fp_budget, color="gray", linestyle="--",
                           alpha=0.5, label=f"FP budget = {args.fp_budget}")
                ax.set_xlabel("hard FP / image")
                ax.set_ylabel("recall (vs GT bboxes)")
                ax.set_title(f"{mode}: recall vs hard FP (color = F1)")
                ax.legend(loc="lower right")
                plt.colorbar(sc, ax=ax, label="F1")
                fig.tight_layout()
                fig.savefig(mode_dir / "pareto.png")
                plt.close(fig)
        except ImportError:
            log.warning("matplotlib not available; skipping pareto plots")

    (out_root / "summary.csv").write_text("\n".join(summary_rows),
                                          encoding="utf-8")
    log.info("done. recommend.json / pareto.csv / pareto.png per mode in %s",
             out_root)


def _fmt(x: float) -> str:
    return "" if x != x else f"{x:.4f}"


def _f(x: float) -> float:
    return -1.0 if x != x else float(x)


if __name__ == "__main__":
    main()
