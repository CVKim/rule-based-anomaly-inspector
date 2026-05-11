# Changelog

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
