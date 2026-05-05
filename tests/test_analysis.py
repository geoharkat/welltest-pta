"""Unit tests for analysis primitives."""

import numpy as np
import pytest

from welltest_pta import (
    bourdet_derivative,
    horner_extrapolation,
    mdh_extrapolation,
    identify_flow_regimes,
    reservoir_parameters,
)


# ── Bourdet ──────────────────────────────────────────────────────────────

def test_bourdet_radial_flow_gives_flat_derivative():
    """For Δp(t) = m·log10(t), Bourdet derivative must be roughly constant."""
    t = np.geomspace(0.001, 100, 500)
    m = 100.0  # psi/cycle
    dp = m * np.log10(t)  # IARF response
    dt_out, deriv_out = bourdet_derivative(t, dp, L=0.2)
    # m_natural-log = m / ln(10)
    expected = m / np.log(10)
    # Discard early/late edges where the derivative is noisier
    deriv_mid = deriv_out[len(deriv_out) // 4 : 3 * len(deriv_out) // 4]
    assert np.allclose(deriv_mid, expected, rtol=0.10)


def test_bourdet_rejects_nonpositive_t():
    t = np.array([0.0, 0.1, 0.2])
    dp = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        bourdet_derivative(t, dp)


# ── Horner ───────────────────────────────────────────────────────────────

def test_horner_flat_data_gives_p_star_close_to_p():
    """If p_ws is constant, P* should equal that constant."""
    t = np.linspace(0, 5, 200)
    p = np.full_like(t, 4500.0)
    res = horner_extrapolation(p, t, tp_hr=10.0)
    assert abs(res["p_star"] - 4500.0) < 1.0


def test_horner_returns_nan_for_invalid_tp():
    t = np.linspace(0, 5, 200)
    p = np.linspace(3000, 4500, 200)
    res = horner_extrapolation(p, t, tp_hr=-1)
    assert np.isnan(res["p_star"])


# ── MDH ──────────────────────────────────────────────────────────────────

def test_mdh_log_data_recovers_slope():
    """Δp(t) = m·log10(Δt) + p0 → MDH slope must equal m within 5 %."""
    dt = np.geomspace(1e-3, 10, 500)
    m_truth = 50.0
    p0 = 3000.0
    p = m_truth * np.log10(dt) + p0
    # Time vector must include t[0] = 0; we offset by the dt grid
    t = dt + 0.001  # arbitrary offset; only differences matter
    res = mdh_extrapolation(p, t)
    assert abs(res["slope_m"] - m_truth) / m_truth < 0.05


# ── Flow regimes ─────────────────────────────────────────────────────────

def test_flow_regime_identifies_iarf():
    """A flat (slope-0) derivative must be classified as IARF."""
    dt = np.geomspace(0.01, 100, 200)
    deriv = np.full_like(dt, 50.0)  # truly flat
    segs = identify_flow_regimes(dt, deriv, slope_tol=0.10)
    regimes = {s["regime"] for s in segs}
    assert "iarf" in regimes


def test_flow_regime_identifies_unit_slope():
    """slope-1 derivative → wellbore_storage OR boundary_closed."""
    dt = np.geomspace(0.01, 100, 200)
    deriv = 10.0 * dt  # slope = 1 in log-log
    segs = identify_flow_regimes(dt, deriv, slope_tol=0.10)
    regimes = {s["regime"] for s in segs}
    assert "wellbore_storage" in regimes or "boundary_closed" in regimes


# ── Reservoir parameters ─────────────────────────────────────────────────

def test_reservoir_parameters_kh_formula():
    """For known slope and inputs, kh = 162.6 q μ B / |m|."""
    p = reservoir_parameters(
        slope_m=-100.0, p_1hr=4400.0, p_wf_at_shut_in=3000.0,
        q=1000.0, mu=0.5, B=1.2,
        h=20.0, phi=0.15, ct=1e-5, rw=0.108,
    )
    expected_kh = 162.6 * 1000.0 * 0.5 * 1.2 / 100.0
    assert abs(p["kh"] - expected_kh) < 1e-3
    assert abs(p["k"] - expected_kh / 20.0) < 1e-3
    assert np.isfinite(p["skin"])


def test_reservoir_parameters_returns_nan_for_zero_slope():
    p = reservoir_parameters(
        slope_m=0.0, p_1hr=4400.0, p_wf_at_shut_in=3000.0,
        q=1000.0, mu=0.5, B=1.2, h=20.0, phi=0.15, ct=1e-5, rw=0.108,
    )
    assert np.isnan(p["kh"])
