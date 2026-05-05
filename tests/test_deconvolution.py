"""Tests for the deconvolution module."""

import numpy as np
import pytest

from welltest_pta import deconvolve, DeconvolutionResult


def test_deconvolve_returns_result_object(fitted_wt):
    """deconvolve() returns a DeconvolutionResult with required arrays."""
    if len(fitted_wt.events) < 2:
        pytest.skip("need at least 2 events")
    res = deconvolve(fitted_wt.events, default_q=850, nu=1e-2,
                      n_response_nodes=30)
    assert isinstance(res, DeconvolutionResult)
    assert len(res.t) == 30
    assert len(res.pu) == 30
    assert len(res.dpu_dlnt) == 30
    assert np.all(np.isfinite(res.pu))


def test_deconvolve_pu_is_monotonic_nondecreasing(fitted_wt):
    """p_u is the integral of a non-negative function — must be non-decreasing."""
    if len(fitted_wt.events) < 2:
        pytest.skip("need at least 2 events")
    res = deconvolve(fitted_wt.events, default_q=850, nu=1e-2,
                      n_response_nodes=30)
    diffs = np.diff(res.pu)
    # By construction (integral of e^z), every diff must be ≥ 0
    assert np.all(diffs >= -1e-6)


def test_deconvolve_dpu_is_positive(fitted_wt):
    """dp_u/d(ln t) = e^z is positive by construction."""
    if len(fitted_wt.events) < 2:
        pytest.skip("need at least 2 events")
    res = deconvolve(fitted_wt.events, default_q=850, nu=1e-2,
                      n_response_nodes=30)
    assert np.all(res.dpu_dlnt > 0)


def test_deconvolve_export_csv(fitted_wt, tmp_path):
    if len(fitted_wt.events) < 2:
        pytest.skip("need at least 2 events")
    res = deconvolve(fitted_wt.events, default_q=850, nu=1e-2,
                      n_response_nodes=30)
    p = tmp_path / "decon.csv"
    res.export(p)
    assert p.exists()
    import pandas as pd
    out = pd.read_csv(p)
    assert "pu_psi_per_unit_q" in out.columns


def test_deconvolve_to_dataframe(fitted_wt):
    if len(fitted_wt.events) < 2:
        pytest.skip("need at least 2 events")
    res = deconvolve(fitted_wt.events, default_q=850, nu=1e-2,
                      n_response_nodes=30)
    df = res.to_dataframe()
    assert {"t_hr", "pu_psi_per_unit_q", "dpu_dlnt_psi_per_unit_q"}.issubset(df.columns)
    assert len(df) == 30


def test_deconvolve_fixed_p_initial(fitted_wt):
    """With fit_p_initial=False, p_initial should be honoured."""
    if len(fitted_wt.events) < 2:
        pytest.skip("need at least 2 events")
    p_i_fixed = 4500.0
    res = deconvolve(
        fitted_wt.events, default_q=850, nu=1e-2,
        n_response_nodes=30,
        fit_p_initial=False, p_initial=p_i_fixed,
    )
    assert abs(res.p_initial - p_i_fixed) < 1e-6
