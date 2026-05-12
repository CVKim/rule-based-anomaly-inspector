"""Compare two evaluation runs (typically Rect-GT vs Polygon-GT) side-by-side.

Reads ``report.json`` from each input directory and prints a unified
summary table covering image-level NG/OK confusion, bbox-level
TP/FN/hardFP/softFP/precision/recall/F1, and pixel-level IoU/Dice/recall/
precision (where available). Also writes ``compare.csv`` with the joined
table for spreadsheet use.

Usage:
    python scripts/compare_eval.py \\
        --rect    outputs/eval_rect \\
        --polygon outputs/eval_polygon \\
        --out     outputs/compare_rect_vs_polygon
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]

import argparse  # noqa: E402
import json      # noqa: E402


def load_report(d: Path) -> dict:
    p = d / "report.json"
    if not p.exists():
        raise SystemExit(f"missing report.json in {d}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _f(x) -> str:
    if x is None:
        return "  -  "
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rect", required=True, type=Path,
                        help="Eval output directory built against Rect GT.")
    parser.add_argument("--polygon", required=True, type=Path,
                        help="Eval output directory built against Polygon GT.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Where to write compare.csv.")
    args = parser.parse_args()

    rect = load_report(args.rect)
    poly = load_report(args.polygon)

    rect_modes = set(rect["modes"])
    poly_modes = set(poly["modes"])
    common = sorted(rect_modes & poly_modes)
    only_rect = sorted(rect_modes - poly_modes)
    only_poly = sorted(poly_modes - rect_modes)

    rect_pixel = rect.get("pixel_metrics", {}) or {}
    poly_pixel = poly.get("pixel_metrics", {}) or {}

    # ---------- console table ------------------------------------------
    header = (f"{'mode':<11}"
              f"{'  | TP':>6} {'FN':>4} {'hFP':>5} {'sFP':>4}"
              f"  | {'P':>6} {'R':>6} {'F1':>6}"
              f"  | {'pIoU':>6} {'pDice':>7} {'pRec':>6} {'pPrec':>7}")
    sep = "-" * len(header)

    def _line_for(mode: str, src: str) -> str:
        rep = rect["modes"][mode] if src == "rect" else poly["modes"][mode]
        bb = rep["bbox"]
        pix = (rect_pixel if src == "rect" else poly_pixel).get(mode, {})
        return (f"{mode + '/' + src:<11}"
                f"  | {bb['tp']:>4} {bb['fn']:>4} {bb['hard_fp']:>5} {bb['soft_fp']:>4}"
                f"  | {_f(bb['precision']):>6} {_f(bb['recall']):>6} {_f(bb['f1']):>6}"
                f"  | {_f(pix.get('iou')):>6} {_f(pix.get('dice')):>7} "
                f"{_f(pix.get('recall')):>6} {_f(pix.get('precision')):>7}")

    print(header)
    print(sep)
    for m in common:
        print(_line_for(m, "rect"))
        print(_line_for(m, "poly"))
        rep_r = rect["modes"][m]["bbox"]
        rep_p = poly["modes"][m]["bbox"]
        d_tp = rep_p["tp"] - rep_r["tp"]
        d_recall = ((rep_p["recall"] or 0.0) - (rep_r["recall"] or 0.0))
        print(f"{'  delta':<11}"
              f"  | {d_tp:>+4} {rep_p['fn'] - rep_r['fn']:>+4} "
              f"{rep_p['hard_fp'] - rep_r['hard_fp']:>+5} "
              f"{rep_p['soft_fp'] - rep_r['soft_fp']:>+4}"
              f"  | {'':>6} {d_recall:>+6.3f} {'':>6}"
              f"  |")
        print(sep)

    if only_rect or only_poly:
        print()
        if only_rect:
            print(f"only in rect: {only_rect}")
        if only_poly:
            print(f"only in polygon: {only_poly}")

    # ---------- compare.csv -------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    csv_lines = ["mode,gt_kind,bbox_TP,bbox_FN,bbox_hardFP,bbox_softFP,"
                 "precision,recall,F1,pix_IoU,pix_Dice,pix_recall,pix_prec"]
    for m in common:
        for src, rep_full, pix_table in [
            ("rect", rect, rect_pixel),
            ("polygon", poly, poly_pixel),
        ]:
            bb = rep_full["modes"][m]["bbox"]
            pix = pix_table.get(m, {})
            csv_lines.append(
                f"{m},{src},{bb['tp']},{bb['fn']},{bb['hard_fp']},"
                f"{bb['soft_fp']},{_f(bb['precision'])},{_f(bb['recall'])},"
                f"{_f(bb['f1'])},{_f(pix.get('iou'))},{_f(pix.get('dice'))},"
                f"{_f(pix.get('recall'))},{_f(pix.get('precision'))}")
    (args.out / "compare.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    print(f"\nwrote {args.out / 'compare.csv'}")


if __name__ == "__main__":
    main()
