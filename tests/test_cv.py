"""Tests for the cross_validate_detector pipeline."""

import pytest

from welltest_pta import cross_validate_detector, DetectorCVResult
from welltest_pta.validation.cross_validation import (
    bootstrap_score,
    parameter_sensitivity,
)


def test_cv_returns_result(synth_df):
    res = cross_validate_detector(synth_df, n_bootstrap=3, print_report=False)
    assert isinstance(res, DetectorCVResult)
    assert 0.0 <= res.overall_score <= 100.0
    assert isinstance(res.grade, str)


def test_cv_grade_consistency(synth_df):
    """Grade label must be consistent with score thresholds."""
    res = cross_validate_detector(synth_df, n_bootstrap=3, print_report=False)
    s = res.overall_score
    if s >= 80:
        assert "ROBUST" in res.grade.upper()
    elif s >= 60:
        assert "REASONABLE" in res.grade.upper()
    elif s >= 40:
        assert "MARGINAL" in res.grade.upper()
    else:
        assert "UNSTABLE" in res.grade.upper()


def test_bootstrap_score_returns_finite(synth_df):
    s = bootstrap_score(synth_df, n_bootstrap=3)
    assert {"n_dd_mean", "n_dd_std", "n_bu_mean", "n_bu_std"}.issubset(s.keys())


def test_parameter_sensitivity_returns_dict(synth_df):
    s = parameter_sensitivity(synth_df, perturbation=0.20)
    expected_params = {
        "hampel_sigma", "spike_percentile",
        "min_pta_dp_psi", "tail_trim_dev_n_sigma",
    }
    assert expected_params.issubset(s.keys())
    for v in s.values():
        assert "delta_n_dd" in v
        assert "delta_n_bu" in v
