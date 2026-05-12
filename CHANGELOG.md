# Changelog

## v0.5.0

GPU-accelerated ridge filter, polygon GT support with pixel-level metrics,
agreement (geometric-mean) fusion, and a Markdown report generator.

### Added

- **Polygon GT support** (``GtBox.polygon`` + ``rasterise``)
  - ``load_labelme_json`` parses both ``rectangle`` and ``polygon`` shapes;
    polygons get an axis-aligned bbox derived from vertex extents AND
    keep the vertex list for true pixel-IoU.
  - New ``polygon_iou(gt, pred_bbox, h, w)`` helper, plus
    ``--polygon-iou`` flag on ``evaluate.py`` / ``tune.py`` to use it
    during matching.

- **Pixel-level metrics** (``PixelMetrics`` + ``pixel_metrics``)
  - IoU / Dice / pixel-recall / pixel-precision aggregated across all
    NG images, surfaced automatically by ``scripts/evaluate.py`` whenever
    polygon GT is present. Reported in ``report.json`` and as new
    ``pix_IoU/pix_Dice/pix_recall/pix_prec`` columns in ``summary.csv``.

- **Comparison panels draw polygon outlines**
  - When polygon GT is loaded the green GT outline is the actual polygon
    contour (not the loose axis-aligned bbox), so operators can
    immediately tell whether a "TP" really overlaps the crack.

- **Torch CUDA path for the multi-scale Frangi ridge filter**
  (``anomaly_inspector.gpu.torch_ridge_response``)
  - All Hessian convolutions, eigenvalue decomposition, vesselness
    aggregation, and per-scale max stay on the GPU. Numerically
    equivalent to the CPU path (cv2.Sobel ksize=3 second-order
    kernels matched explicitly, sigma**2 normalisation preserved).
  - **~25-50× speedup** on an RTX 3080 vs the OpenCV CPU path
    (12 MP image: 50s → 2-4s including the master-image pass).
  - Wired via ``ResidualConfig.use_gpu_ridge`` and ``--gpu-ridge``
    on the eval / sweep / tune scripts. Silent fallback to CPU when
    torch is missing or CUDA isn't available, so config files stay
    portable.

- **OpenCL UMat helper** (``anomaly_inspector.gpu.GpuContext``)
  - Thin layer over cv2.UMat for the rest of the residual stages
    (gradient, NCC, multiscale). Empirically slower than CPU on this
    workload because of host↔device transfer overhead, so off by
    default; exposed via ``ResidualConfig.use_opencl`` for users on
    different hardware to benchmark.

- **Agreement fusion** (``ResidualConfig.fused_op="agree"``)
  - Per-pixel weighted geometric mean (``exp(Σ w_i log(r_i))``) of
    constituent residuals. Bounded above by the arithmetic mean (the
    "mean" op), so where one mode is silent the agree map collapses
    toward zero — that's the FP-suppression property. Where multiple
    modes agree, the score is comparable to the arithmetic mean.

- **Markdown report generator** (``scripts/report_metrics.py``)
  - Reads one or more ``report.json`` files and emits a single
    ``report.md`` with run metadata, bbox- and image-level metric
    tables, pixel metrics, and per-image breakdowns. Use to diff runs
    or share results without spreadsheet handling.

- **Side-by-side comparison script** (``scripts/compare_eval.py``)
  - Console + ``compare.csv`` aligning Rect-GT vs Polygon-GT eval
    runs (same prediction set, two different GT label kinds). Useful
    when the labeller drops both rectangle and polygon variants.

### Changed

- ``GtBox.rasterise`` and ``pixel_metrics`` switched from numpy slicing
  (exclusive end) to ``cv2.rectangle`` / ``cv2.fillPoly`` (inclusive
  boundary) so a perfect polygon-vs-rectangle overlap reports IoU = 1
  instead of 1 - off-by-one.

### Real-data results (FOOSUNG side-view, 6 NG + 4 OK images)

Best balanced operating point and best-recall operating point are
genuinely different products — published both:

| run                       | mode    | F1    | Recall (bbox) | hardFP/img | Pix IoU | Pix recall |
|---------------------------|---------|-------|---------------|------------|---------|------------|
| 2400 wide (Polygon GT)    | absdiff | **0.261** | 0.261 | **2.12** | 0.179 | 0.242 |
| 2400 wide (Polygon GT)    | ridge   | 0.127 | 0.542         | 20.88      | 0.106 | 0.408 |
| Full-res GPU (v4)         | ridge   | 0.077 | 0.739         | 50.0       | 0.108 | 0.514 |
| Full-res GPU agree (v4)   | fused   | 0.047 | **0.783**     | 90.0       | 0.080 | **0.589** |
| Full-res GPU+autoignore (v5) | ridge | 0.084 | 0.739       | 45.5       | 0.114 | 0.510 |
| Full-res GPU+autoignore (v5) | fused | 0.050 | **0.783**   | 85.25      | 0.083 | 0.553 |

**Operator guidance** (also written into ``outputs/REPORT.md``):

- Default production setting — **absdiff at 2400-wide** with
  ``min_blob_area=60``, ``k_sigma=3.0``, ``base_tolerance=5.0``: best
  bbox F1 (0.261), best precision-per-cost, ~2 hard FP/image.
- "No-FN" regime — **agree fusion of (absdiff, ridge) at full-res with
  GPU**: catches **78% of GT cracks** (vs 26% for absdiff) and **59%
  of GT pixels**, but emits ~85-90 hard FPs/image — usable as a
  first-pass with downstream classifier or visual review.
- Image-level NG/OK accuracy is **6/6 NG correct** at every operating
  point but **2/4 OK images (#1, #11) are over-flagged** at every
  setting; that's a fundamental limit of comparing single rotations
  against a single reference and would require multi-rotation
  consensus to fix.

### Tests

- 18 new tests across ``test_evaluation.py`` (10) and
  ``test_ridge_fused.py`` (4 new for fused_op variants and GPU
  fallback). Suite total: **95 tests, all green**.

## v0.4.1

Polygon GT support driven by the FOOSUNG side-view dataset's
``2026-05-12_Polygon`` annotations sitting alongside the original
rectangles. Polygon GT areas are typically half the rectangle area for
thin/diagonal cracks (the rectangle bounding box overestimates by 2-5×
on diagonals), which lets the evaluator score predictions much more
honestly.

### Added

- **Polygon parsing in ``load_labelme_json``**
  - ``GtBox`` now carries an optional ``polygon`` field with the original
    vertex list. ``rasterise(h, w)`` fills the polygon (or rectangle) on
    a uint8 mask via ``cv2.fillPoly`` / ``cv2.rectangle`` so the boundary
    convention matches across both shape sources.

- **``polygon_iou(gt_box, pred_bbox, image_h, image_w)``**
  - True polygon IoU between a (possibly polygonal) GT and an
    axis-aligned predicted bbox via mask intersection. Pass
    ``--polygon-iou`` to ``evaluate.py`` / ``tune.py`` to use it during
    matching.

- **``PixelMetrics`` + ``pixel_metrics`` helper**
  - Pixel-level intersection / union / Dice / pixel-recall /
    pixel-precision aggregated over all NG images in a run, surfaced
    automatically by ``scripts/evaluate.py`` whenever the GT folder
    contains polygons. Reported in both ``report.json`` and
    ``summary.csv`` (extra columns ``pix_IoU``, ``pix_Dice``,
    ``pix_recall``, ``pix_prec``).

- **Polygon-aware comparison panels**
  - ``scripts/evaluate.py`` now draws the polygon contour in green
    instead of the loose axis-aligned bbox when polygons are present —
    much easier to see whether a "TP" prediction actually overlaps the
    crack vs just sitting in the same general area.

### Changed

- ``GtBox.rasterise`` and ``pixel_metrics`` use ``cv2.rectangle`` /
  ``cv2.fillPoly`` consistently so the boundary-pixel convention matches.
  Without this, a perfect polygon-vs-rectangle overlap would have
  reported IoU < 1 due to the off-by-one between numpy slicing
  (exclusive) and OpenCV fill (inclusive).

### Tests

- 8 new tests in ``test_evaluation.py`` covering polygon parsing,
  vertex-count guard, mixed rect+polygon shapes, ``GtBox.rasterise``,
  ``polygon_iou`` (vs ``bbox_iou`` for both axis-aligned and diagonal
  cases), and ``pixel_metrics`` perfect / no-overlap / mismatch.
  Suite total: **91 tests, all green**.

## v0.4.0

GT-driven evaluation harness, two new residual modes targeting cracks and
mode-agreement, an offline auto-tuner, and a major default-correction:
inspect at native resolution unless explicitly downsampled. Turning the
new pieces on is opt-in; v0.3 defaults are otherwise preserved.

### Added

- **Evaluation harness** (``anomaly_inspector.evaluation``)
  - ``load_labelme_json`` / ``load_gt_folder`` parser for labelme rectangle
    GT, with support for empty-shape-list = OK images.
  - ``evaluate_image`` performs greedy-IoU TP matching and buckets leftover
    predictions as ``hard_FP`` (counts toward precision) or ``soft_FP``
    (inside the part ROI, near a labelled GT — surfaced for review,
    excluded from precision since the user explicitly noted GT may be
    incomplete).
  - ``ModeReport`` aggregates per-image evaluations into bbox-level P/R/F1,
    image-level NG/OK confusion, and mode-complementarity matrix.
  - ``scripts/evaluate.py`` runs the full pipeline per mode and writes
    ``report.json``, ``summary.csv``, ``mode_complementarity.csv``, and
    per-image comparison panels (GT green, TP cyan, hard FP red, soft FP
    orange, FN yellow X).

- **Ridge filter residual** (``mode="ridge"``)
  - Multi-scale Frangi-style Hessian vesselness with sign-filtered
    polarity (``dark`` / ``bright`` / ``both``). Ideal for crack-like
    1D structures the absdiff/NCC modes underweight.
  - ``ridge_master_dilate`` neighbourhood-max on the master response
    before subtraction so legitimate machined edges that drift by a few
    sub-pixel slop pixels in the target still cancel cleanly. Cuts
    ridge FP/img by ~60% on the FOOSUNG dataset (153 → 63).

- **Score-level fusion** (``mode="fused"``)
  - Per-pixel weighted mean of robust-percentile-normalised residuals
    across ``fused_modes`` (default ``absdiff + ncc + ridge``). Rewards
    mode AGREEMENT instead of taking the loudest signal — mode-specific
    noise gets averaged down while real defects (which fire across
    multiple residual families) survive.

- **Auto-tuning** (``scripts/tune.py``)
  - Per-mode grid search over ``k_sigma`` × ``base_tolerance`` ×
    ``min_blob_area`` × ``morph_ksize`` × ``auto_ignore_percentile``.
    Caches the residual map per (mode, image) so only the cheap
    threshold/morph/blob/eval steps re-run per grid point — full sweep
    on the FOOSUNG dataset finishes in ~30 minutes at native resolution.
  - Outputs ``recommend.json`` (best F1 within the operator's hard-FP
    budget), ``pareto.csv`` (every grid point), and a ``pareto.png``
    scatter (recall vs hard-FP/img coloured by F1).

- **Crack-friendly default** ``min_blob_area`` lowered from 15 to 30 (still
  well below the smallest GT crack of ~68 px² at 1600-wide; raised from
  the v0.1 floor of 15 to suppress single-component noise without
  losing recall).

### Fixed

- The earlier sweep script's default ``--max-input-width 1600`` was
  silently zeroing out recall: the median GT crack is 14 px tall in the
  source image, which becomes ~5 px after the 2.56× downsample and is
  then smoothed away by the Gaussian blur. The eval and tune scripts now
  default to native resolution; the user-facing inference panel script
  retains its 1600 default for speed but is documented as a
  speed-vs-recall knob.

### Real-data impact (FOOSUNG side-view, 6 NG + 4 OK images, GT vs preds)

Bbox-level metrics — IoU≥0.05 against the labelled CRACK GT, hard FPs
only (soft FPs surfaced separately).

**Default-parameter baseline (k_sigma=3, base_tol=5, min_blob=80, full-res):**

| mode                           | Recall | Precision | F1   | hard_FP/img |
|--------------------------------|--------|-----------|------|-------------|
| absdiff                        | 0.458  | 0.083     | 0.141 | 12.1 |
| multiscale                     | 0.500  | 0.058     | 0.104 | 19.5 |
| ridge (with master_dilate)     | **0.792** | 0.029  | 0.056 | 63.4 |
| fused (absdiff+ncc+ridge)      | 0.000  | 0.000     | -    | 8.7 |

**After auto-tuning (per-mode best within FP budget = 10/image):**

| mode    | Recall | Precision | F1    | hard_FP/img | recommended params (k_σ, base, min_blob, auto_ig) |
|---------|--------|-----------|-------|-------------|----------------------------------------------------|
| absdiff | 0.417  | 0.196     | **0.267** | 4.1   | (3.5, 6.0, 150, off) |
| ridge   | **0.833** | 0.052  | 0.098 | 36.5        | (5.0, 6.0, 150, 99.5) |
| fused   | 0.042  | 0.017     | 0.024 | 5.9         | (5.0, 6.0, 150, 99.5) |

Image-level NG/OK classification: every mode catches 6/6 NG images at
default thresholds.

**Tuner findings — operator guidance:**

- **Balanced production setting → ``absdiff`` with the recommended
  params**: 1.9× F1 improvement over baseline (0.141 → 0.267) just by
  tightening ``min_blob_area`` to 150 (suppressing single-component
  noise) and ``k_sigma`` to 3.5. Best precision/recall trade.
- **No-FN regime → ``ridge``**: 83% recall is the highest of any
  rule-based mode but the hard-FP floor stays around 36/image. Use as
  a first-pass + downstream classifier or raise the FP budget.
- **``fused`` does NOT help for crack-style defects**: averaging
  normalised residuals dilutes ridge's strong crack signal with the
  weak absdiff/ncc responses, so most threshold combinations fire on
  shared edge artifacts instead of cracks. Fusion remains useful for
  defect families where multiple residual modes naturally agree
  (large scratches, smudges, contamination), which the auto-tuner
  output makes explicit so operators can pick the right tool.

### Tests

- 8 new tests in ``test_ridge_fused.py`` (Frangi response on synthetic
  lines, polarity filtering, master-ridge cancellation, fusion mode-
  agreement, meta round-trip, end-to-end pipeline). Suite total:
  **68 tests, all green**.

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
