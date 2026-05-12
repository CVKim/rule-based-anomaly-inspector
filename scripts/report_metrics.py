"""Render a self-contained Markdown report from one or more eval runs.

Reads ``report.json`` from each input directory and emits a single
``report.md`` that collects:

* Run metadata (args used, image counts, GT type)
* Bbox-level metrics table (TP/FN/hardFP/softFP/precision/recall/F1)
* Image-level NG/OK confusion matrix per mode
* Pixel-level metrics (when polygon GT was used)
* Mode-complementarity matrix
* Per-image breakdown for each mode

Usage:
    python scripts/report_metrics.py \\
        --inputs outputs/eval_rect outputs/eval_polygon outputs/eval_v4_gpu \\
        --names  Rect Polygon "GPU+agree" \\
        --out    outputs/report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]

import argparse  # noqa: E402
import json      # noqa: E402


def _f(x, fmt: str = ".4f") -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
        return f"{v:{fmt}}"
    except (TypeError, ValueError):
        return str(x)


def fmt_run(name: str, report: dict) -> str:
    out = [f"## {name}", ""]
    args = report.get("args", {})
    out.append(f"- modes: `{', '.join(report['modes'].keys())}`")
    out.append(f"- max_input_width: `{args.get('max_input_width')}` "
               f"(0 = native)")
    out.append(f"- thresholds: `k_sigma={args.get('k_sigma')}` "
               f"`base_tolerance={args.get('base_tolerance')}` "
               f"`min_blob_area={args.get('min_blob_area')}`")
    out.append(f"- align_method: `{args.get('align_method')}`")
    out.append(f"- ROI: `{args.get('roi')}` (close_ksize=`{args.get('roi_close_ksize')}`, "
               f"erode_px=`{args.get('roi_erode_px')}`)")
    out.append(f"- residual options: pyramid_levels=`{args.get('pyramid_levels')}` "
               f"ncc_window=`{args.get('ncc_window')}` "
               f"gradient_op=`{args.get('gradient_op')}`")
    if args.get("gpu_ridge"):
        out.append("- **GPU ridge (torch CUDA): on**")
    if args.get("fused_op") and args.get("fused_op") != "mean":
        out.append(f"- fused_op: `{args.get('fused_op')}`  "
                   f"fused_modes: `{args.get('fused_modes')}`")
    if args.get("polygon_iou"):
        out.append("- polygon-IoU matching: on")
    out.append(f"- iou_threshold: `{args.get('iou_threshold')}`  "
               f"soft_fp_distance_px: `{args.get('soft_fp_distance_px')}`")
    out.append("")

    # Bbox table
    out.append("### Bbox-level metrics")
    out.append("")
    out.append("| mode | TP | FN | hardFP | softFP | precision | recall | F1 | hardFP/img |")
    out.append("|------|----|----|--------|--------|-----------|--------|-----|------------|")
    for m, rep in report["modes"].items():
        bb = rep["bbox"]
        out.append(
            f"| {m} | {bb['tp']} | {bb['fn']} | {bb['hard_fp']} | "
            f"{bb['soft_fp']} | {_f(bb['precision'])} | {_f(bb['recall'])} | "
            f"{_f(bb['f1'])} | {bb['hard_fp_per_image']:.2f} |"
        )
    out.append("")

    # Image-level table
    out.append("### Image-level NG/OK confusion")
    out.append("")
    out.append("| mode | TP (NG hit) | FP (false alarm) | TN (clean OK) | FN (missed NG) |")
    out.append("|------|------------:|-----------------:|--------------:|---------------:|")
    for m, rep in report["modes"].items():
        ic = rep["image"]
        out.append(f"| {m} | {ic['TP']} | {ic['FP']} | {ic['TN']} | {ic['FN']} |")
    out.append("")

    # Pixel metrics if available
    pix = report.get("pixel_metrics") or {}
    if pix:
        out.append("### Pixel-level metrics (from polygon GT)")
        out.append("")
        out.append("| mode | IoU | Dice | pix_recall | pix_precision | inter | gt_area | pred_area |")
        out.append("|------|-----|------|------------|---------------|-------|---------|-----------|")
        for m, p in pix.items():
            out.append(
                f"| {m} | {_f(p.get('iou'), '.3f')} | "
                f"{_f(p.get('dice'), '.3f')} | "
                f"{_f(p.get('recall'), '.3f')} | "
                f"{_f(p.get('precision'), '.3f')} | "
                f"{int(p.get('intersection', 0))} | "
                f"{int(p.get('gt_area', 0))} | "
                f"{int(p.get('pred_area', 0))} |"
            )
        out.append("")

    # Per-image breakdown (one table per mode)
    out.append("### Per-image breakdown")
    out.append("")
    for m, rep in report["modes"].items():
        out.append(f"#### {m}")
        out.append("")
        out.append("| image | n_GT | TP | FN | hardFP | softFP | recall | precision | F1 |")
        out.append("|-------|------|----|----|--------|--------|--------|-----------|-----|")
        for img in rep["per_image"]:
            out.append(
                f"| {img['filename']} | {img['n_gt']} | {img['tp']} | "
                f"{img['fn']} | {img['hard_fp']} | {img['soft_fp']} | "
                f"{_f(img['recall'])} | {_f(img['precision'])} | "
                f"{_f(img['f1'])} |"
            )
        out.append("")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="One or more eval output directories.")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names matching --inputs.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Path to write the markdown file.")
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.inputs):
        raise SystemExit("--names length must match --inputs length")
    names = args.names or [str(p.name) for p in args.inputs]

    sections = ["# Anomaly inspector — evaluation report", ""]
    sections.append(f"_runs:_ {' • '.join(names)}")
    sections.append("")

    for name, d in zip(names, args.inputs):
        rp = d / "report.json"
        if not rp.exists():
            sections.append(f"## {name}\n\n_skipped — no report.json in {d}_\n")
            continue
        with open(rp, "r", encoding="utf-8") as f:
            report = json.load(f)
        sections.append(fmt_run(name, report))
        sections.append("---")
        sections.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
