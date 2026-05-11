"""Rule-based anomaly inspector — public API."""

from .inspector import DynamicToleranceInspector, InspectionResult, DefectInfo
from .reference import ReferenceBuilder, Reference
from .alignment import align_translation, align_ecc, align_log_polar, estimate_rotation_scale
from .classification import auto_unreliable_mask, classify, shape_features
from .photometric import PhotometricCorrector, flat_field_divide, top_hat, clahe
from .residual import ResidualConfig, compute_residual
from .roi import RoiConfig, auto_part_roi
from .panel import make_panel
from .visualization import draw_defects, side_by_side

__all__ = [
    "DynamicToleranceInspector",
    "InspectionResult",
    "DefectInfo",
    "ReferenceBuilder",
    "Reference",
    "align_translation",
    "align_ecc",
    "align_log_polar",
    "estimate_rotation_scale",
    "PhotometricCorrector",
    "flat_field_divide",
    "top_hat",
    "clahe",
    "auto_unreliable_mask",
    "classify",
    "shape_features",
    "ResidualConfig",
    "compute_residual",
    "RoiConfig",
    "auto_part_roi",
    "make_panel",
    "draw_defects",
    "side_by_side",
]

__version__ = "0.3.1"
