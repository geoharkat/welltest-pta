"""Validation submodule — cross-validation and stability scores."""

from welltest_pta.validation.cross_validation import (
    cross_validate_detector,
    DetectorCVResult,
    bootstrap_score,
    parameter_sensitivity,
)

__all__ = [
    "cross_validate_detector",
    "DetectorCVResult",
    "bootstrap_score",
    "parameter_sensitivity",
]
