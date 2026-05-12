"""GT-driven evaluation harness.

For each requested residual mode:

1. Build one shared reference from the normals folder.
2. Inspect every test image at (optionally downsampled) inference resolution.
3. Match the predicted bboxes to the labelme GT bboxes via IoU and bucket
   leftover predictions as ``hard FP`` (counts toward precision) or
   ``soft FP`` (likely unlabelled — surfaced but excluded from precision).
4. Write a ``report.json`` with full per-image + aggregate metrics, a
   ``summary.csv`` with mode × metric grid, and a comparison panel per
   image colouring TP cyan, FP red (hard) / orange (soft), FN yellow X,
   and GT in green.

Usage:
    python scripts/evaluate.py \
        --normal "H:/.../FS_측면/10회 반복" \
        --test   "H:/.../FS_측면" \
        --gt     "H:/.../2026-05-12" \
        --output outputs/eval_v1 \
        --modes  absdiff multiscale ncc gradient
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]

import argparse  # noqa: E402
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
from anomaly_inspector.evaluation import (  # noqa: E402
    ModeReport, boxes_from_defects, evaluate_image, load_gt_folder,
    mode_complementarity,
)
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


def render_comparison_panel(image: np.ndarray,
                            gt_boxes_full: list[tuple[float, float, float, float]],
                            pred_boxes_pred: list[tuple[float, float, float, float]],
                            ev,
                            pred_scale: float,
                            roi_mask: np.ndarray | None,
                            max_width: int = 1800,
                            title: str = "") -> np.ndarray:
    """One-cell panel: original image + GT (green), TP (cyan), hard FP (red),
    soft FP (orange), FN marker (yellow X). All boxes drawn in pred-space
    coordinates (pred_scale=1) or full-res (then scaled down for display)."""
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 200, 0), 2)

    matched_pred_idx = {pi for (pi, _, _) in ev.matched_pairs}

    def _draw(rect: tuple[float, float, float, float],
              color: tuple[int, int, int], thickness: int = 3,
              label: str | None = None):
        x, y, w, h = rect
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(vis, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # GT (green, dashed look via thin border)
    for g in gt_boxes_full:
        _draw(g, (0, 255, 0), 2, "GT")

    for pi, p in enumerate(pred_boxes_pred):
        p_full = (p[0] * pred_scale, p[1] * pred_scale,
                  p[2] * pred_scale, p[3] * pred_scale)
        if pi in matched_pred_idx:
            _draw(p_full, (255, 200, 0), 3, "TP")    # cyan-ish
        elif pi in ev.soft_fp_pred_indices:
            _draw(p_full, (0, 165, 255), 2, "soft")  # orange
        else:
            _draw(p_full, (0, 0, 255), 2, "FP")      # red

    # FN markers (yellow X across the GT centre)
    for gi in ev.fn_indices:
        g = gt_boxes_full[gi]
        cx = int(round(g[0] + g[2] / 2.0))
        cy = int(round(g[1] + g[3] / 2.0))
        s = 16
        cv2.line(vis, (cx - s, cy - s), (cx + s, cy + s), (0, 255, 255), 3)
        cv2.line(vis, (cx + s, cy - s), (cx - s, cy + s), (0, 255, 255), 3)
        cv2.putText(vis, "FN", (cx + s + 4, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if vis.shape[1] > max_width:
        scale = max_width / vis.shape[1]
        vis = cv2.resize(vis, (max_width, int(vis.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)

    if title:
        bar = np.full((40, vis.shape[1], 3), 28, dtype=np.uint8)
        cv2.putText(bar, title, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        vis = np.vstack([bar, vis])

    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path,
                        help="Folder of labelme JSONs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", nargs="+",
                        default=["absdiff", "multiscale", "ncc", "gradient",
                                 "ridge", "fused"],
                        choices=["absdiff", "multiscale", "ncc", "gradient",
                                 "ridge", "fused"])

    parser.add_argument("--photometric", default="flat_field",
                        choices=["none", "flat_field", "top_hat_white",
                                 "top_hat_black", "clahe"])
    parser.add_argument("--photo-sigma", type=float, default=51.0)
    parser.add_argument("--dispersion", default="mad", choices=["std", "mad"])
    parser.add_argument("--ref-blur", type=int, default=5)
    parser.add_argument("--roi", default="otsu_close",
                        choices=["none", "otsu_close", "fixed_threshold"])
    parser.add_argument("--roi-close-ksize", type=int, default=81)
    parser.add_argument("--roi-erode-px", type=int, default=12)

    parser.add_argument("--k-sigma", type=float, default=4.0)
    parser.add_argument("--base-tolerance", type=float, default=8.0)
    parser.add_argument("--min-blob-area", type=int, default=40)
    parser.add_argument("--blur-ksize", type=int, default=5)
    parser.add_argument("--align-method", default="phase",
                        choices=["none", "phase", "phase+ecc",
                                 "logpolar", "logpolar+phase"])
    parser.add_argument("--morph-ksize", type=int, default=5)
    parser.add_argument("--auto-ignore-percentile", type=float, default=99.5)

    parser.add_argument("--pyramid-levels", type=int, default=3)
    parser.add_argument("--ncc-window", type=int, default=21)
    parser.add_argument("--gradient-op", default="scharr",
                        choices=["sobel", "scharr"])
    parser.add_argument("--gradient-blend", type=float, default=0.0)

    parser.add_argument("--max-input-width", type=int, default=1600,
                        help="Downsample huge inputs to this width before "
                             "inspecting (set 0 to disable).")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    parser.add_argument("--soft-fp-distance-px", type=float, default=120.0,
                        help="FPs within this many full-res pixels of any GT "
                             "are bucketed as 'soft' (likely unlabelled).")
    parser.add_argument("--no-panels", action="store_true",
                        help="Skip the per-image comparison panels (faster).")

    args = parser.parse_args()
    log = get_logger()

    # ------------------------------------------------------------------
    # Load GT + image lists
    # ------------------------------------------------------------------
    gt = load_gt_folder(args.gt)
    log.info("loaded %d GT files (NG=%d, OK=%d)", len(gt),
             sum(1 for g in gt.values() if g.is_ng),
             sum(1 for g in gt.values() if not g.is_ng))

    test_paths = list_images(args.test)
    test_paths = [p for p in test_paths if p.name in gt]
    if not test_paths:
        raise SystemExit(f"no test images matched any GT file in {args.gt}")
    log.info("matched %d test images to GT", len(test_paths))

    normal_paths = list_images(args.normal)
    log.info("normals: %d", len(normal_paths))

    # ------------------------------------------------------------------
    # Build reference (one shared, all modes)
    # ------------------------------------------------------------------
    log.info("loading + downsampling normals...")
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

    photometric = PhotometricCorrector(
        method=args.photometric, sigma=args.photo_sigma,            # type: ignore[arg-type]
    )
    roi_cfg = RoiConfig(
        method=args.roi, close_ksize=args.roi_close_ksize,           # type: ignore[arg-type]
        erode_px=args.roi_erode_px,
    )
    builder = ReferenceBuilder(
        blur_ksize=args.ref_blur, align=True, dispersion=args.dispersion,
        photometric=photometric, roi=roi_cfg,
    )
    t0 = time.time()
    ref = builder.from_images(normals)
    log.info("reference built in %.1fs (shape=%s, roi=%s)",
             time.time() - t0, ref.master.shape, roi_cfg.method)

    # Use the first GT we have to anchor full-res coordinates; everyone is
    # 4096x2851 in this dataset but we read it from the JSON to be safe.
    a_gt = next(iter(gt.values()))
    full_res = (a_gt.image_height, a_gt.image_width)
    pred_scale = full_res[1] / target_shape[1]   # x-axis; aspect preserved
    log.info("pred_scale = %.4f (%dpx wide -> %dpx wide)",
             pred_scale, target_shape[1], full_res[1])

    # ------------------------------------------------------------------
    # Per-mode inspect + evaluate
    # ------------------------------------------------------------------
    out_root = ensure_dir(args.output)
    reports: dict[str, ModeReport] = {}

    # Pre-load test images at inference scale (one shot, reused across modes)
    targets: dict[str, np.ndarray] = {}
    for p in test_paths:
        img = load_gray_unicode(p)
        if img.shape != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]),
                             interpolation=cv2.INTER_AREA)
        targets[p.name] = img

    for mode in args.modes:
        log.info("=== mode: %s ===", mode)
        inspector = make_inspector(ref, mode, args)
        rep = ModeReport(mode=mode, iou_threshold=args.iou_threshold)
        mode_dir = ensure_dir(out_root / mode)

        for p in test_paths:
            img = targets[p.name]
            t1 = time.time()
            result = inspector.inspect(img)
            ms = (time.time() - t1) * 1000.0

            pred_boxes = boxes_from_defects(result.defects)
            ev = evaluate_image(
                pred_boxes=pred_boxes,
                gt=gt[p.name],
                iou_threshold=args.iou_threshold,
                soft_fp_max_centre_distance=args.soft_fp_distance_px,
                roi_mask=ref.roi_mask,
                pred_scale=pred_scale,
            )
            rep.per_image.append(ev)
            log.info("  %-12s  GT=%d  TP=%d  FN=%d  hard_FP=%d  soft_FP=%d  %.0fms",
                     p.name, ev.n_gt, ev.tp, ev.fn, ev.hard_fp, ev.soft_fp, ms)

            if not args.no_panels:
                gt_full = [g.bbox for g in gt[p.name].boxes]
                title = (f"{p.name}  |  mode={mode}  |  "
                         f"TP={ev.tp}  FN={ev.fn}  hardFP={ev.hard_fp}  softFP={ev.soft_fp}")
                panel = render_comparison_panel(
                    img, gt_boxes_full=gt_full,
                    pred_boxes_pred=pred_boxes, ev=ev,
                    pred_scale=pred_scale,
                    roi_mask=ref.roi_mask, title=title,
                )
                imwrite_unicode(mode_dir / f"{p.stem}_eval.png", panel)

        reports[mode] = rep
        ic = rep.image_classification
        log.info("  -> bbox: P=%.3f  R=%.3f  F1=%.3f  hardFP/img=%.2f  softFP_total=%d",
                 _nan(rep.precision), _nan(rep.recall), _nan(rep.f1),
                 rep.hard_fp_per_image, rep.total_soft_fp)
        log.info("  -> image classification: TP=%d FP=%d TN=%d FN=%d",
                 ic["TP"], ic["FP"], ic["TN"], ic["FN"])

    # ------------------------------------------------------------------
    # Write report.json + summary.csv + complementarity matrix
    # ------------------------------------------------------------------
    full_report = {
        "args": {k: (str(v) if isinstance(v, Path) else v)
                  for k, v in vars(args).items()},
        "n_test_images": len(test_paths),
        "modes": {m: rep.to_dict() for m, rep in reports.items()},
    }
    (out_root / "report.json").write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Summary CSV (mode x headline metric)
    csv_lines = [
        "mode,bbox_TP,bbox_FN,bbox_hardFP,bbox_softFP,precision,recall,F1,"
        "hardFP_per_img,img_TP,img_FP,img_TN,img_FN"
    ]
    for m, rep in reports.items():
        ic = rep.image_classification
        csv_lines.append(
            f"{m},{rep.total_tp},{rep.total_fn},{rep.total_hard_fp},"
            f"{rep.total_soft_fp},"
            f"{_csv(rep.precision)},{_csv(rep.recall)},{_csv(rep.f1)},"
            f"{rep.hard_fp_per_image:.2f},"
            f"{ic['TP']},{ic['FP']},{ic['TN']},{ic['FN']}"
        )
    (out_root / "summary.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    # Mode complementarity (which mode catches what others miss)
    matrix = mode_complementarity(reports)
    cm_lines = ["from \\ to," + ",".join(reports.keys())]
    for a, row in matrix.items():
        cm_lines.append(f"{a}," + ",".join(str(row[b]) for b in reports))
    (out_root / "mode_complementarity.csv").write_text("\n".join(cm_lines),
                                                       encoding="utf-8")

    log.info("done. report.json + summary.csv + comparison panels in %s", out_root)


def _nan(x: float) -> float:
    return -1.0 if x != x else x


def _csv(x: float) -> str:
    return "" if x != x else f"{x:.4f}"


if __name__ == "__main__":
    main()
