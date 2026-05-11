# Rule-based Anomaly Inspector

A modular, OpenCV-only rule-based anomaly detection pipeline for industrial
vision inspection. Designed for the cold-start case where you have a stable
master image (or a small bag of known-good samples) and defects appear as
local deviations: scratches, dents, foreign material, missing components,
lighting glints, etc. **No training data required.**

The pipeline is seven stages, each independently swappable and configurable:

1. **Reference build** — stack ≥20–30 known-good images, compute the
   per-pixel **median** (master) and per-pixel **dispersion** (`std` or
   robust `MAD`) — these become the per-pixel tolerance budget.
2. **Photometric normalization** *(v0.2)* — flat-field divide /
   morphological top-hat / CLAHE to flatten low-frequency illumination drift
   *before* differencing. The same correction is applied to both master and
   target.
3. **Geometric alignment** — translation-only via phase correlation,
   refinement via ECC (translation + rotation), or *(v0.2)*
   log-polar phase correlation that recovers **rotation + scale + translation**
   in a single pass for rotary-stage cameras.
4. **Residual computation** *(new in v0.3)* — pluggable. Pick or fuse:
   `absdiff` (default — `|target - master|`), `multiscale` (3-level Gaussian
   pyramid, per-pixel max — recovers wide low-contrast blobs that vanish at
   full res), `ncc` (sliding-window normalized cross-correlation — robust to
   global brightness drift), or `gradient` (Sobel/Scharr magnitude diff —
   sensitive to edge-shape defects).
5. **Asymmetric dynamic thresholding** — per-pixel
   `base_tolerance + k_sigma * std`, with separate dark and bright budgets
   so a dent-sensitive line can keep dark tight while letting glints slide.
6. **Mask combination + morphology** — user `ignore_mask` (markers, text,
   barcodes) **OR**'d with an *(v0.2)* auto-derived high-variance
   mask, then open + close to denoise.
7. **Blob analysis + classification** *(v0.2)* — connected-component
   filtering by area, then each surviving blob is tagged as
   `scratch / spot / dent / smudge / unknown` from minAreaRect aspect ratio,
   `4πA/P²` circularity, convex-hull solidity, and the signed mean diff
   (polarity).

## Project layout

```
rule-based-anomaly-inspector/
├── src/anomaly_inspector/
│   ├── inspector.py            # DynamicToleranceInspector class
│   ├── reference.py            # ReferenceBuilder
│   ├── alignment.py            # phase / ECC / log-polar alignment
│   ├── photometric.py          # flat-field / top-hat / CLAHE
│   ├── classification.py       # shape descriptors + scratch/dent/spot rules
│   ├── visualization.py        # category-colored overlays + heatmap
│   ├── cli.py                  # Typer entrypoints
│   └── utils.py                # I/O, config, logging
├── scripts/
│   ├── build_reference.py      # CLI: build master + tolerance from a folder
│   └── inspect.py              # CLI: inspect images using a saved reference
├── configs/default.yaml        # default hyper-parameters
├── tests/                      # synthetic-data smoke tests (29 tests)
└── examples/example_usage.py   # minimal end-to-end demo
```

## Setup

Conda is recommended (GPU is not required — pure CPU OpenCV is enough):

```bash
conda env create -f environment.yml
conda activate anomaly-inspector
pip install -e .
```

Or with pip only:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
# 1) Build the reference from a folder of known-good images
python scripts/build_reference.py \
    --input  assets/sample_data/normal \
    --output reference/master.npz \
    --config configs/default.yaml

# 2) Inspect a single image (or a folder)
python scripts/inspect.py \
    --reference reference/master.npz \
    --input     assets/sample_data/test \
    --output    outputs/ \
    --config    configs/default.yaml
```

The `outputs/` folder will hold `<image>_result.png` debug panels
(master / aligned / diff heatmap / category-colored defects) and a
`summary.csv`:

```
filename,n_defects,shift_x,shift_y,rotation_deg,scale,categories
panel_001.bmp,2,0.412,-0.118,0.034,1.00012,dent/dark;scratch/dark
```

## Library API

```python
import cv2
from anomaly_inspector import (
    DynamicToleranceInspector, ReferenceBuilder, PhotometricCorrector,
)

# 1) Build reference with flat-field illumination correction
builder = ReferenceBuilder(
    blur_ksize=5, align=True, dispersion="mad",
    photometric=PhotometricCorrector(method="flat_field", sigma=31.0),
)
ref = builder.from_paths(["normals/img_01.bmp", "normals/img_02.bmp", ...])

# 2) Run inspection with rotation alignment, asymmetric thresholds,
#    auto-mask of edge halos, and shape classification
inspector = DynamicToleranceInspector(
    ref,
    align_method="logpolar+phase",
    k_sigma_dark=3.5, k_sigma_bright=6.0,        # tighten dark, loosen bright
    auto_ignore_percentile=99.0,                  # auto-mask high-variance pixels
)
result = inspector.inspect(cv2.imread("test.bmp", cv2.IMREAD_GRAYSCALE))

print(f"defects: {len(result.defects)}, "
      f"rotation: {result.rotation_deg:.2f}°, scale: {result.scale:.4f}")
for d in result.defects:
    print(f"  {d.category}/{d.polarity} bbox={d.bbox} area={d.area} "
          f"max_diff={d.max_diff:.0f}")
```

## Tuning guide

| Parameter                 | Default | Effect                                                                            |
|---------------------------|---------|-----------------------------------------------------------------------------------|
| `k_sigma`                 | 4.0     | ↓ catches subtler defects, ↑ only obvious ones. Symmetric default.                |
| `k_sigma_dark` / `_bright`| =k_sigma| Asymmetric: e.g. dark=3.5, bright=8.0 lets glints slide while keeping dents.      |
| `base_tolerance`          | 5.0     | 8-bit cushion for flat regions where `std≈0`.                                     |
| `min_blob_area`           | 15      | Reject specks below this pixel area.                                              |
| `align_method`            | `phase` | `none` / `phase` / `phase+ecc` / `logpolar` / `logpolar+phase`.                   |
| `photometric.method`      | `none`  | `flat_field` for ring-light drift, `top_hat_white` to isolate bright defects.     |
| `auto_ignore_percentile`  | `null`  | e.g. `99.0` to auto-suppress the noisiest 1% of pixels (typically edge halos).    |
| `classify_defects`        | `true`  | Tag blobs as scratch/spot/dent/smudge for triage. Disable for raw geometry only.  |
| `residual.mode`           | `absdiff`| `multiscale` for wide blobs, `ncc` for brightness drift, `gradient` for edge defects. |
| `residual.gradient_blend` | `0.0`   | >0 blends the gradient residual on top of the primary mode (best of both).        |

### Sweep all four residuals over a folder

```bash
python scripts/run_inference_panels.py \
    --normal "path/to/known_good" \
    --test   "path/to/test_images" \
    --output outputs/run01 \
    --modes  absdiff multiscale ncc gradient \
    --max-input-width 1600        # downsample huge inputs (set 0 to disable)
```

Produces `outputs/run01/<mode>/<stem>_panel.png` six-cell visualisations
(image | heatmap | mask pred | pred conf fg | pred conf bg | overlay) and
a per-mode `summary.csv` with detection counts, recovered alignment, and
per-defect categories.

### Choosing a photometric method

| Symptom                                              | Try                              |
|------------------------------------------------------|----------------------------------|
| Brightness drifts slowly across the panel            | `flat_field` with `sigma >> defect_size`           |
| You only care about bright defects on a dark surface | `top_hat_white` with `ksize >> defect_size`        |
| You only care about dark defects on a bright surface | `top_hat_black` with `ksize >> defect_size`        |
| Local *contrast* (not just brightness) drifts        | `clahe`, `clip_limit ~ 2.0–4.0`, `tile_grid ~ 8`   |

### Choosing an alignment method

| Stage characteristic                                  | `align_method`        |
|-------------------------------------------------------|-----------------------|
| Hard-mounted, sub-pixel translation only              | `phase`               |
| Mild in-plane rotation that ECC can polish            | `phase+ecc`           |
| Rotary stage, small but real rotation per shot        | `logpolar` or `logpolar+phase` |
| Already aligned mechanically; want zero overhead      | `none`                |

## Git branching model

- `main` — only validated, releasable code, tagged at each release.
- `dev` — integration branch, branched from `main`.
- `feature/*` — branched from `dev`, merged back to `dev` (no fast-forward),
  finally PR'd into `main` at release time.

This repo currently sits at **v0.3.0** on `main`. v0.3.0 added the pluggable
residual stage (multiscale / NCC / gradient) and the six-panel visualisation
+ inference-sweep script on top of v0.2.0's photometric, log-polar, and
classification stages, themselves built on the v0.1.0 core.

## Tests

```bash
python -m pytest -q
```

45 synthetic-data smoke tests covering the reference builder, all three
alignment modes, the photometric stage, the classifier, every residual
mode, and end-to-end defect detection under illumination drift and rotation.

## License

MIT
