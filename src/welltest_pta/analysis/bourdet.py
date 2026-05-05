"""
welltest_pta.analysis.bourdet
=============================
Bourdet (1989) logarithmic pressure derivative for log–log diagnostic plots.

The derivative of pressure with respect to natural log of elapsed time is
the diagnostic backbone of modern PTA. Constant slopes in the log–log plot
reveal flow regimes:

- Unit slope (m=1) at early time → wellbore storage
- Zero slope (flat plateau)      → infinite-acting radial flow (IARF)
- Half slope (m=½)               → linear flow (channel / hydraulic fracture)
- Quarter slope (m=¼)            → bilinear flow (finite-conductivity fracture)
- Unit slope at late time        → closed boundary (pseudo-steady state)
- Slope of −½                    → constant-pressure boundary
"""

from __future__ import annotations

import numpy as np


def bourdet_derivative(
    dt: np.ndarray,
    dp: np.ndarray,
    L: float = 0.2,
    use_smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute the Bourdet logarithmic pressure derivative.

    Uses the three-point window formulation:

    .. math::
        \left.\frac{d\,\Delta p}{d\,\ln \Delta t}\right|_i \approx
        \frac{\Delta p_{j_+} - \Delta p_{j_-}}
             {\ln\Delta t_{j_+} - \ln\Delta t_{j_-}}

    where :math:`j_-` and :math:`j_+` are the nearest indices to :math:`i`
    with :math:`|\ln \Delta t_j - \ln \Delta t_i| \ge L`. The smoothing
    parameter :math:`L` (typically 0.05–0.4) controls noise rejection.

    Parameters
    ----------
    dt
        Elapsed shut-in time (or producing time) in hours, monotonically
        increasing, **starting strictly above zero** (use ``dt[0] = 1e-6``
        if needed).
    dp
        Pressure change :math:`\Delta p = p(t) - p(t=0)` (psi).
    L
        Logarithmic smoothing window (default 0.2). Larger = smoother.
    use_smooth
        If True (default) returns the absolute value (typical for log–log
        plots). Set False to keep the signed derivative.

    Returns
    -------
    dt_out, deriv_out
        Arrays of elapsed time and derivative values where the central-
        difference formula was evaluable. Length ≤ ``len(dt) - 2``.
    """
    dt = np.asarray(dt, dtype=float)
    dp = np.asarray(dp, dtype=float)
    if dt.shape != dp.shape:
        raise ValueError("dt and dp must have the same shape")
    if (dt <= 0).any():
        raise ValueError("dt must be strictly positive (use dt[0]=1e-6 if needed)")

    ln_dt = np.log(dt)
    n = len(dt)
    dt_out, deriv_out = [], []

    for i in range(1, n - 1):
        j_back = None
        for j in range(i - 1, -1, -1):
            if ln_dt[i] - ln_dt[j] >= L:
                j_back = j
                break
        j_fwd = None
        for j in range(i + 1, n):
            if ln_dt[j] - ln_dt[i] >= L:
                j_fwd = j
                break
        if j_back is None or j_fwd is None:
            continue

        ln_diff = ln_dt[j_fwd] - ln_dt[j_back]
        dt_diff = dt[j_fwd] - dt[j_back]
        if ln_diff == 0 or dt_diff == 0:
            continue

        # Bourdet "classic" form: deriv = dt_i * Δp / Δt — equivalent to dp/d(ln t)
        deriv = dt[i] * (dp[j_fwd] - dp[j_back]) / dt_diff
        dt_out.append(dt[i])
        deriv_out.append(abs(deriv) if use_smooth else deriv)

    return np.asarray(dt_out), np.asarray(deriv_out)
