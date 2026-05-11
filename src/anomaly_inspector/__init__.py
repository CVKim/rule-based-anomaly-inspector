"""Rule-based anomaly inspector — public API."""

from .inspector import DynamicToleranceInspector, InspectionResult, DefectInfo
from .reference import ReferenceBuilder, Reference
from .alignment import align_translation, align_ecc, align_log_polar, estimate_rotation_scale
from .classification import auto_unreliable_mask, classify, shape_features
from .photometric import PhotometricCorrector, flat_field_divide, top_hat, clahe
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
    "draw_defects",
    "side_by_side",
]

__version__ = "0.2.0"
