"""Rule-based anomaly inspector — public API."""

from .inspector import DynamicToleranceInspector, InspectionResult, DefectInfo
from .reference import ReferenceBuilder, Reference
from .alignment import align_translation, align_ecc
from .visualization import draw_defects, side_by_side

__all__ = [
    "DynamicToleranceInspector",
    "InspectionResult",
    "DefectInfo",
    "ReferenceBuilder",
    "Reference",
    "align_translation",
    "align_ecc",
    "draw_defects",
    "side_by_side",
]

__version__ = "0.1.0"
