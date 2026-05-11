# Rule-based Anomaly Inspector

A modular rule-based anomaly detection pipeline for industrial vision inspection,
built on top of OpenCV. It is intended for scenes where a stable master image
exists and defects appear as local deviations from the master (scratches, dents,
foreign material, missing components, etc.).

The pipeline follows four stages:

1. **Reference build** — stack multiple known-good images, compute the per-pixel
   median (master) and per-pixel standard deviation (tolerance map).
2. **Fine alignment** — phase-correlation based sub-pixel translation alignment
   of the target image against the master, with an optional ECC refinement.
3. **Dynamic thresholding** — per-pixel threshold `base_tolerance + k_sigma * std`
   so flat regions don't over-trigger and textured regions get a wider budget.
4. **Morphology + blob analysis** — open/close to denoise, connected components
   with area / aspect filters to emit defect candidates.

## Project layout

```
rule-based-anomaly-inspector/
├── src/anomaly_inspector/      # library code (importable package)
│   ├── inspector.py            # DynamicToleranceInspector class
│   ├── reference.py            # ReferenceBuilder
│   ├── alignment.py            # phaseCorrelate + ECC alignment
│   ├── visualization.py        # overlay rendering
│   └── utils.py                # I/O, config, logging helpers
├── scripts/
│   ├── build_reference.py      # CLI: build master+tolerance from a folder
│   └── inspect.py              # CLI: inspect images using a saved reference
├── configs/default.yaml        # default hyper-parameters
├── tests/test_inspector.py     # synthetic-data smoke tests
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

## Tuning tips

| Parameter         | Effect                                                                          |
|-------------------|---------------------------------------------------------------------------------|
| `k_sigma`         | ↓ over-detects fine defects, ↑ only catches obvious ones. Start at 4.0.         |
| `base_tolerance`  | Cushion for flat regions where `std≈0`. Start at 5 (8-bit gray).                |
| `min_blob_area`   | Reject specks below this pixel area.                                            |
| `align.method`    | `phase` (fast, translation only) or `phase+ecc` (slower, handles small warps).  |
| `ignore_mask`     | Bitmask of regions to exclude from inspection (markers, text, etc.).            |

## Git branching model

- `main` — only validated, releasable code.
- `dev` — integration branch, branched from `main`.
- `feature/*` — branched from `dev`, merged back to `dev`, finally fast-forwarded
  (or merged) into `main` after validation.

## License

MIT
