"""
welltest_pta.analysis.mdh
=========================
Miller–Dyes–Hutchinson (1950) semi-log buildup analysis.

The MDH plot is :math:`p_{ws}` vs :math:`\\log_{10}(\\Delta t)`. It is the
short-shut-in approximation of the Horner method, valid when
:math:`\\Delta t \\ll t_p` (i.e. when :math:`\\log\\big[(t_p+\\Delta t)/\\Delta t\\big]
\\approx \\log(t_p) - \\log(\\Delta t)`).

The MDH slope is **identical in magnitude** to the Horner slope and is used
for permeability and skin calculation. MDH is preferred when :math:`t_p` is
ill-defined (multiple drawdown periods) and the user wishes to avoid Horner-
time bias.
"""

from __future__ import annotations

import numpy as np


def mdh_extrapolation(
    p: np.ndarray,
    t_hr: np.ndarray,
    fit_start_frac: float = 0.30,
    fit_end_frac: float = 0.85,
) -> dict[str, float]:
    """
    Linear regression of :math:`p_{ws}` vs :math:`\\log_{10}(\\Delta t)` on
    the IARF window.

    Returns dict with ``slope_m`` (psi/cycle), ``intercept_p1hr``
    (extrapolated pressure at :math:`\\Delta t=1` hr — used for skin),
    and ``r2``.
    """
    p = np.asarray(p, dtype=float)
    t_hr = np.asarray(t_hr, dtype=float)

    dt = t_hr - t_hr[0]
    dt = np.where(dt <= 0, 1e-6, dt)
    log_dt = np.log10(dt)

    n_seg = len(p)
    i0 = max(1, int(n_seg * fit_start_frac))
    i1 = max(i0 + 5, int(n_seg * fit_end_frac))
    if i1 - i0 < 5:
        return {"slope_m": np.nan, "intercept_p1hr": np.nan, "r2": np.nan,
                "fit_start_idx": -1, "fit_end_idx": -1}

    x_fit, y_fit = log_dt[i0:i1], p[i0:i1]
    A = np.vstack([x_fit, np.ones_like(x_fit)]).T
    (m, b), *_ = np.linalg.lstsq(A, y_fit, rcond=None)
    y_pred = m * x_fit + b
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # p_1hr = m * log10(1) + b = b   (intercept at Δt = 1 hr)
    return {
        "slope_m": float(m),
        "intercept_p1hr": float(b),
        "r2": float(r2),
        "fit_start_idx": int(i0),
        "fit_end_idx": int(i1),
    }
