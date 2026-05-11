"""Typer-based CLI entrypoints used by the console scripts in pyproject.toml."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import typer

from . import ReferenceBuilder, DynamicToleranceInspector
from .photometric import PhotometricCorrector
from .utils import (
    ensure_dir, get_logger, list_images, load_config, load_gray,
    load_reference, save_reference,
)
from .visualization import side_by_side


build_app = typer.Typer(add_completion=False, help="Build a reference (master + tolerance map).")
inspect_app = typer.Typer(add_completion=False, help="Run anomaly inspection on images.")


@build_app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i",
                               help="Folder of known-good images."),
    output: Path = typer.Option(..., "--output", "-o",
                                help="Path to write the .npz reference."),
    config: Optional[Path] = typer.Option(None, "--config", "-c",
                                          help="Optional YAML config."),
    blur_ksize: int = typer.Option(5, "--blur-ksize"),
    align: bool = typer.Option(True, "--align/--no-align"),
    dispersion: str = typer.Option("std", "--dispersion",
                                   help="'std' or 'mad'."),
    photometric: str = typer.Option("none", "--photometric",
                                    help="'none' | 'flat_field' | 'top_hat_white' | 'top_hat_black' | 'clahe'."),
) -> None:
    """Build a reference from a folder of known-good images."""
    log = get_logger()
    cfg = load_config(config) if config else {}
    ref_cfg = cfg.get("reference", {})
    blur_ksize = ref_cfg.get("blur_ksize", blur_ksize)
    align = ref_cfg.get("align", align)
    dispersion = ref_cfg.get("dispersion", dispersion)
    photo_cfg = ref_cfg.get("photometric", {})
    if isinstance(photo_cfg, str):
        photo_cfg = {"method": photo_cfg}
    photo_cfg.setdefault("method", photometric)
    corrector = PhotometricCorrector.from_meta(photo_cfg)

    paths = list_images(input)
    if not paths:
        raise typer.BadParameter(f"no images in {input}")
    log.info("loading %d images from %s", len(paths), input)

    builder = ReferenceBuilder(blur_ksize=blur_ksize, align=align,
                               dispersion=dispersion, photometric=corrector)
    ref = builder.from_paths(paths)
    save_reference(output, ref.master, ref.tolerance,
                   meta={"n_samples": ref.n_samples, "dispersion": ref.method,
                         "blur_ksize": blur_ksize,
                         "photometric": corrector.to_meta()})
    log.info("wrote reference to %s (n=%d, shape=%s, photometric=%s)",
             output, ref.n_samples, ref.master.shape, corrector.method)


@inspect_app.command()
def run(
    reference: Path = typer.Option(..., "--reference", "-r",
                                   help="Path to .npz reference."),
    input: Path = typer.Option(..., "--input", "-i",
                               help="Image file or folder of images."),
    output: Path = typer.Option(..., "--output", "-o",
                                help="Output directory for overlays."),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    k_sigma: float = typer.Option(4.0, "--k-sigma"),
    base_tolerance: float = typer.Option(5.0, "--base-tolerance"),
    min_blob_area: int = typer.Option(15, "--min-blob-area"),
    blur_ksize: int = typer.Option(5, "--blur-ksize"),
    align_method: str = typer.Option("phase", "--align-method",
                                     help="'none' | 'phase' | 'phase+ecc' | "
                                          "'logpolar' | 'logpolar+phase'."),
    morph_ksize: int = typer.Option(3, "--morph-ksize"),
    k_sigma_dark: Optional[float] = typer.Option(None, "--k-sigma-dark",
                                                 help="Asymmetric: tighten threshold for dark anomalies."),
    k_sigma_bright: Optional[float] = typer.Option(None, "--k-sigma-bright",
                                                   help="Asymmetric: tighten threshold for bright anomalies."),
    base_tolerance_dark: Optional[float] = typer.Option(None, "--base-tolerance-dark"),
    base_tolerance_bright: Optional[float] = typer.Option(None, "--base-tolerance-bright"),
    auto_ignore_percentile: Optional[float] = typer.Option(
        None, "--auto-ignore-percentile",
        help="Auto-mask pixels above this tolerance percentile (e.g. 99.0)."),
    classify_defects: bool = typer.Option(True, "--classify/--no-classify",
                                          help="Classify each blob as scratch/spot/dent/smudge."),
) -> None:
    """Inspect one image (or every image in a folder) and write overlays."""
    log = get_logger()
    cfg = load_config(config) if config else {}
    insp_cfg = cfg.get("inspect", {})
    k_sigma = insp_cfg.get("k_sigma", k_sigma)
    base_tolerance = insp_cfg.get("base_tolerance", base_tolerance)
    min_blob_area = insp_cfg.get("min_blob_area", min_blob_area)
    blur_ksize = insp_cfg.get("blur_ksize", blur_ksize)
    align_method = insp_cfg.get("align_method", align_method)
    morph_ksize = insp_cfg.get("morph_ksize", morph_ksize)
    k_sigma_dark = insp_cfg.get("k_sigma_dark", k_sigma_dark)
    k_sigma_bright = insp_cfg.get("k_sigma_bright", k_sigma_bright)
    base_tolerance_dark = insp_cfg.get("base_tolerance_dark", base_tolerance_dark)
    base_tolerance_bright = insp_cfg.get("base_tolerance_bright", base_tolerance_bright)
    auto_ignore_percentile = insp_cfg.get("auto_ignore_percentile",
                                          auto_ignore_percentile)
    classify_defects = insp_cfg.get("classify_defects", classify_defects)

    master, tolerance, meta = load_reference(reference)
    log.info("loaded reference: shape=%s, meta=%s", master.shape, meta)

    from .reference import Reference
    ref = Reference(master=master, tolerance=tolerance,
                    method=meta.get("dispersion", "std"),
                    n_samples=meta.get("n_samples", 0),
                    photometric=PhotometricCorrector.from_meta(meta.get("photometric")))

    inspector = DynamicToleranceInspector(
        ref, k_sigma=k_sigma, base_tolerance=base_tolerance,
        min_blob_area=min_blob_area, blur_ksize=blur_ksize,
        align_method=align_method, morph_ksize=morph_ksize,
        k_sigma_dark=k_sigma_dark, k_sigma_bright=k_sigma_bright,
        base_tolerance_dark=base_tolerance_dark,
        base_tolerance_bright=base_tolerance_bright,
        auto_ignore_percentile=auto_ignore_percentile,
        classify_defects=classify_defects,
    )

    if input.is_dir():
        paths = list_images(input)
    elif input.is_file():
        paths = [input]
    else:
        raise typer.BadParameter(f"input does not exist: {input}")

    ensure_dir(output)
    summary_lines = ["filename,n_defects,shift_x,shift_y,rotation_deg,scale,categories"]
    for p in paths:
        img = load_gray(p)
        result = inspector.inspect(img)
        vis = side_by_side(result, ref.master)
        out_path = output / f"{p.stem}_result.png"
        cv2.imwrite(str(out_path), vis)
        cat_summary = ";".join(f"{d.category}/{d.polarity}"
                               for d in result.defects) or "-"
        log.info("%-40s defects=%d shift=(%.2f, %.2f) rot=%.2f scale=%.4f categories=%s",
                 p.name, len(result.defects), *result.shift,
                 result.rotation_deg, result.scale, cat_summary)
        summary_lines.append(
            f"{p.name},{len(result.defects)},"
            f"{result.shift[0]:.3f},{result.shift[1]:.3f},"
            f"{result.rotation_deg:.3f},{result.scale:.5f},{cat_summary}"
        )

    (output / "summary.csv").write_text("\n".join(summary_lines), encoding="utf-8")
    log.info("done — wrote %d overlays to %s", len(paths), output)


def build_reference_main() -> None:
    build_app()


def inspect_main() -> None:
    inspect_app()
