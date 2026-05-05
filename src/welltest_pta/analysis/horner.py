"""
welltest_pta.analysis.horner
============================
Horner (1951) semi-log extrapolation for buildups.

The Horner plot of :math:`p_{ws}` vs :math:`\\log_{10}\\big[(t_p+\\Delta t)/\\Delta t\\big]`
is linear during infinite-acting radial flow, with slope

.. math::
    m = \\frac{162.6\\, q\\, \\mu\\, B}{k\\, h}\\quad\\text{(field units)}

Extrapolating the IARF straight line to :math:`(t_p+\\Delta t)/\\Delta t = 1`
(i.e. infinite shut-in) gives the false reservoir pressure :math:`P^*`.
For an infinite-acting reservoir :math:`P^* = \\bar{p}`; for bounded
systems use Matthews–Brons–Hazebroek (MBH) correction.
"""

from __future__ import annotations

import numpy as np


def horner_extrapolation(
    p: np.ndarray,
    t_hr: np.ndarray,
    tp_hr: float,
    fit_start_frac: float = 0.30,
    fit_end_frac: float = 0.85,
) -> dict[str, float]:
    r"""
    Linear regression of :math:`p_{ws}` vs Horner time on the IARF window.

    Parameters
    ----------
    p
        Shut-in pressure (smoothed) over the buildup, length :math:`n_b`.
    t_hr
        Elapsed time in hours (any reference; only differences matter).
    tp_hr
        Producing time before shut-in (hr). For a Horner plot this is
        the duration of the immediately preceding drawdown.
    fit_start_frac, fit_end_frac
        Fractional positions along the buildup that delimit the
        infinite-acting straight line. Defaults skip the early
        wellbore-storage transition (≤30 %) and the late boundary
        deviation (≥85 %).

    Returns
    -------
    dict with keys

    - ``p_star`` : extrapolated pressure at :math:`(t_p+\\Delta t)/\\Delta t = 1`
    - ``slope_m`` : Horner slope (psi / log-cycle), used for ``kh``
    - ``r2`` : coefficient of determination on the fit window
    - ``fit_start_idx``, ``fit_end_idx`` : indices of fit window
    """
    p = np.asarray(p, dtype=float)
    t_hr = np.asarray(t_hr, dtype=float)
    if tp_hr is None or np.isnan(tp_hr) or tp_hr <= 0:
        return {"p_star": np.nan, "slope_m": np.nan, "r2": np.nan,
                "fit_start_idx": -1, "fit_end_idx": -1}

    dt = t_hr - t_hr[0]
    dt = np.where(dt <= 0, 1e-6, dt)
    horner_x = np.log10((tp_hr + dt) / dt)

    n_seg = len(p)
    i0 = max(1, int(n_seg * fit_start_frac))
    i1 = max(i0 + 5, int(n_seg * fit_end_frac))
    if i1 - i0 < 5:
        return {"p_star": np.nan, "slope_m": np.nan, "r2": np.nan,
                "fit_start_idx": -1, "fit_end_idx": -1}

    x_fit, y_fit = horner_x[i0:i1], p[i0:i1]
    A = np.vstack([x_fit, np.ones_like(x_fit)]).T
    (m, p_star), *_ = np.linalg.lstsq(A, y_fit, rcond=None)
    y_pred = m * x_fit + p_star
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "p_star": float(p_star),
        "slope_m": float(m),
        "r2": float(r2),
        "fit_start_idx": int(i0),
        "fit_end_idx": int(i1),
    }


def horner_diagnostic_line(
    tp_hr: float, dt_min: float, dt_max: float, n: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Return (horner_x, dt_grid) for plotting the IARF line of slope ``m``."""
    dt = np.geomspace(max(dt_min, 1e-6), dt_max, n)
    return np.log10((tp_hr + dt) / dt), dt
