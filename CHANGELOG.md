# Changelog

## v0.3.1

Bug-fix release driven by the FOOSUNG side-view dataset, where the part
occupies <20% of the frame against a near-black background.

### Added

- **Auto part-region (ROI) extraction** (``anomaly_inspector.roi``)
  - ``RoiConfig`` selects the strategy: ``otsu_close`` (Otsu + morph close
    + largest CC + boundary erode), ``fixed_threshold``, or ``none``.
  - ``auto_part_roi`` returns ``None`` when no clear part is found
    (covers <``min_area_fraction`` or >``max_area_fraction`` of the frame),
    so the inspector falls back to whole-frame inspection cleanly.
  - Built once at reference time, persisted in the .npz, applied as an
    ignore mask in the inspector — affects every residual mode.
  - ``--roi``, ``--roi-close-ksize``, ``--roi-erode-px`` CLI flags and a
    new ``reference.roi:`` config block.
  - Panel viz now outlines the ROI in green on the ``image`` cell so
    operators see exactly what was inspected.

### Fixed

- ROI is now derived from the **raw** (pre-photometric) median; the
  flat-field / top-hat normalisers crush the part/background contrast
  Otsu needs and would otherwise return a degenerate full-frame mask.
- ``save_reference`` / ``load_reference`` round-trip the ROI mask through
  the .npz alongside master + tolerance + meta.

### Real-data impact (FOOSUNG side-view, 10 normals + 10 test frames)

False-positive collapse with no change in true-positive recall:

| mode       | before (no ROI) | after (ROI on) |
|------------|-----------------|-----------------|
| absdiff    | 4.6 avg / image | 3.3 avg / image |
| multiscale | 7.2             | 5.2             |
| ncc        | 30.1            | 16.0            |
| gradient   | 56.9            | 31.8            |

NCC and gradient become operationally usable; absdiff/multiscale gain a
modest cleanup on the part outline.

### Tests

15 new tests in ``test_roi.py`` covering config validation, recess/CC/
erode behaviour, degenerate-input handling, and pipeline integration
(background defects suppressed, in-part defects preserved, all four
residual modes verified to never *increase* count when ROI is enabled).
Suite total: **60 tests, all green**.

## v0.3.0

Pluggable residual stage + reporting tooling. Default residual mode stays
``absdiff`` so v0.2 behaviour is bit-for-bit preserved.

### Added

- **Residual module** (``anomaly_inspector.residual``)
  - ``ResidualConfig`` dataclass selecting one of ``absdiff`` /
    ``multiscale`` / ``ncc`` / ``gradient`` plus mode-specific knobs.
  - ``compute_residual`` returns ``(signed, abs)`` maps in gray-level units
    so the dynamic-tolerance threshold keeps its operator-facing meaning
    regardless of the underlying metric.
  - ``gradient_blend`` knob ADDS the gradient residual on top of the
    primary residual with a configurable weight; ``extra_modes`` fuses
    additional modes via per-pixel max.
  - ``--residual``, ``--pyramid-levels``, ``--ncc-window``, ``--gradient-op``,
    ``--gradient-blend`` CLI flags and a matching ``residual:`` config block.

- **Six-panel debug visualisation** (``anomaly_inspector.panel.make_panel``)
  - Layout: ``image | heatmap | mask pred | pred conf fg | pred conf bg | overlay``.
  - Caps each cell width to keep PNG files manageable for 12 MP+ sources;
    overall panel is downsized losslessly.

- **Multi-mode inference sweep script**
  ``scripts/run_inference_panels.py``
  - Builds one shared reference and runs the full pipeline under each
    requested residual mode, writing ``<output>/<mode>/<stem>_panel.png``
    plus a per-mode ``summary.csv`` (defect count, alignment, categories,
    inference latency).
  - Drops the script's own folder from ``sys.path`` *before* any other
    imports so the sibling ``scripts/inspect.py`` doesn't shadow stdlib
    ``inspect`` (which previously broke ``numpy``/``dataclasses`` import).

- **Unicode-safe Windows I/O** (``utils.imread_unicode`` / ``imwrite_unicode``)
  - ``cv2.imread``/``imwrite`` mangle non-ASCII paths through the active
    ANSI codepage on Windows. The new helpers go via
    ``np.fromfile`` + ``cv2.imdecode``/``cv2.imencode`` and round-trip
    Korean / Japanese / Chinese paths cleanly. ``load_gray`` and the CLI
    now use them.

### Changed

- ``DynamicToleranceInspector`` accepts ``residual: ResidualConfig`` and
  delegates the diff/threshold step to ``compute_residual``. The bright/
  dark asymmetric thresholds still apply on the *signed* output of
  absdiff/multiscale; for ncc/gradient (which are sign-less) only the
  bright threshold fires, which is the right behaviour for those metrics.

### Tests

- 16 new tests in ``test_residual.py`` covering config validation,
  per-mode residual semantics (NCC stable under brightness shift,
  gradient lights up on new edges, multiscale catches wide low-contrast
  blobs), and a parametrised pipeline round-trip across all four modes.
  Suite total: **45 tests, all green**.

## v0.2.0

Three rule-based extensions on top of the v0.1.0 core. All v0.1 defaults are
preserved — turning each new feature on is opt-in.

### Added

- **Photometric normalization** (`anomaly_inspector.photometric`)
  - `PhotometricCorrector` with `flat_field`, `top_hat_white`,
    `top_hat_black`, and `clahe` strategies.
  - The chosen method is persisted in the `.npz` reference and applied
    identically by `ReferenceBuilder` and `DynamicToleranceInspector`,
    with a runtime warning if a mismatch is constructed.
  - `--photometric` CLI flag and `reference.photometric:` config block.

- **Log-polar rotation + scale alignment**
  (`anomaly_inspector.alignment.align_log_polar`,
  `estimate_rotation_scale`)
  - Reddy & Chatterji 1996 method on a Hann-windowed, high-passed
    magnitude spectrum, with an optional final phase-correlation pass for
    sub-pixel translation.
  - New `align_method` values `"logpolar"` and `"logpolar+phase"` on the
    inspector.
  - `InspectionResult` now exposes `rotation_deg` and `scale`; CSV
    summary gains the matching columns.

- **Asymmetric tolerance**
  - `k_sigma_dark` / `k_sigma_bright` and matching
    `base_tolerance_dark` / `base_tolerance_bright` parameters on the
    inspector. Both default to the symmetric value, preserving v0.1
    behaviour.
  - Mask construction now uses signed difference, not just absolute
    value, so the bright and dark sides can be tuned independently.

- **Auto-ignore mask**
  (`anomaly_inspector.classification.auto_unreliable_mask`)
  - Optional `auto_ignore_percentile` flag on the inspector marks the
    top-N% of the tolerance map as "don't trust", typically catching
    high-variance edge halos without a hand-painted ROI.
  - Combined with any user-supplied `ignore_mask` via `bitwise_or`.

- **Blob classification**
  (`anomaly_inspector.classification.shape_features`, `classify`)
  - Each `DefectInfo` now carries `category`
    (`scratch / spot / dent / smudge / unknown`), `polarity`
    (`dark / bright`), and the underlying `aspect_ratio`, `circularity`,
    `solidity` descriptors.
  - `draw_defects` color-codes by category; CSV summary gains a
    `categories` column.
  - Toggle via `classify_defects` (default on).

### Changed

- `DefectInfo` and `InspectionResult` gained fields. Existing positional
  / keyword construction still works; new fields default sensibly.
- CSV summary header is now
  `filename,n_defects,shift_x,shift_y,rotation_deg,scale,categories`.

### Tests

- 17 new tests across `test_photometric.py`,
  `test_alignment_logpolar.py`, and `test_classification.py`. Suite is
  29 tests total, all green.

## v0.1.0

Initial release: per-pixel median master + std/MAD tolerance map, phase
correlation + ECC alignment, dynamic thresholding, morphology, blob
analysis, Typer CLIs, and synthetic-data smoke tests.
