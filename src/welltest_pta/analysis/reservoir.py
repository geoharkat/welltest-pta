r"""
welltest_pta.analysis.reservoir
===============================
Reservoir parameter calculation from PTA results — field oilfield units.

All formulae use **field oilfield units**:

==================  ====================  ====================
Quantity            Symbol                Units
==================  ====================  ====================
Flow rate           q                     STB/D (oil) or Mscf/D (gas)
Viscosity           µ                     cp
FVF                 B                     RB/STB (oil) or RB/Mscf
Permeability        k                     mD
Net pay             h                     ft
Slope (semilog)     m                     psi / log-cycle
Wellbore radius     rw                    ft
Total compress.     ct                    psi⁻¹
Porosity            φ                     fraction
Wellbore storage    C                     bbl/psi
==================  ====================  ====================

Standard radial-flow equations (oil):

.. math::
    k\,h = \frac{162.6\, q\, \mu\, B}{|m|}

.. math::
    k = \frac{k\,h}{h}

.. math::
    S = 1.151 \left[
        \frac{p_{1\text{hr}} - p_{wf}(\Delta t = 0)}{|m|}
        - \log_{10}\!\left(\frac{k}{\phi\, \mu\, c_t\, r_w^2}\right)
        + 3.23
    \right]

For drawdowns, replace :math:`p_{1\text{hr}} - p_{wf}(\Delta t = 0)` with
:math:`p_i - p_{1\text{hr}}` and switch sign convention. Wellbore storage
constant from a unit-slope early-time line:

.. math::
    C = \frac{q\, B}{24\, m_{\text{slope-1}}} \quad\text{[bbl/psi]}

where :math:`m_{\text{slope-1}}` is the early-time linear slope of
:math:`\Delta p` vs :math:`\Delta t` (psi/hr).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def reservoir_parameters(
    slope_m: float,
    p_1hr: float,
    p_wf_at_shut_in: float,
    q: float,
    mu: float,
    B: float,
    h: float,
    phi: float,
    ct: float,
    rw: float,
    early_storage_slope: Optional[float] = None,
) -> dict[str, float]:
    r"""
    Compute :math:`kh`, :math:`k`, :math:`S`, and (optionally) :math:`C`.

    Parameters
    ----------
    slope_m
        Semilog slope (psi/cycle) — magnitude doesn't matter, sign is
        handled internally.
    p_1hr
        Pressure at :math:`\Delta t = 1` hr on the IARF straight line
        (psi). For Horner: :math:`p_{1\text{hr}} = m \log_{10}(t_p+1)+P^*`.
        For MDH: just the intercept.
    p_wf_at_shut_in
        Flowing pressure at the instant of shut-in (psi). Skin is computed
        from :math:`(p_{1\text{hr}} - p_{wf})`.
    q
        Flow rate (STB/D for oil — convert gas to its own equation).
    mu
        Viscosity (cp).
    B
        Formation volume factor (RB/STB).
    h
        Net pay (ft).
    phi
        Porosity (fraction, 0–1).
    ct
        Total compressibility (1/psi).
    rw
        Wellbore radius (ft).
    early_storage_slope
        Optional — :math:`d\Delta p/dt` at very early time (psi/hr). If
        provided, wellbore storage :math:`C` (bbl/psi) is also returned.

    Returns
    -------
    dict with keys ``kh``, ``k``, ``skin``, ``C`` (None if not supplied),
    plus ``p_skin`` (additional pressure drop due to skin, psi).
    """
    if slope_m == 0 or not np.isfinite(slope_m):
        return {"kh": np.nan, "k": np.nan, "skin": np.nan, "C": None, "p_skin": np.nan}

    abs_m = abs(slope_m)
    kh = 162.6 * q * mu * B / abs_m
    k = kh / h if h > 0 else np.nan

    skin = np.nan
    if h > 0 and phi > 0 and mu > 0 and ct > 0 and rw > 0:
        skin = 1.151 * (
            (p_1hr - p_wf_at_shut_in) / abs_m
            - np.log10(k / (phi * mu * ct * rw ** 2))
            + 3.23
        )
    p_skin = 0.87 * abs_m * skin if np.isfinite(skin) else np.nan

    C: Optional[float] = None
    if early_storage_slope and early_storage_slope > 0:
        # Unit-slope WBS line:  Δp = (q·B / 24·C) · Δt   →   C = q·B / (24·slope)
        C = q * B / (24.0 * early_storage_slope)

    return {
        "kh": float(kh),
        "k": float(k),
        "skin": float(skin) if np.isfinite(skin) else np.nan,
        "C": float(C) if C is not None else None,
        "p_skin": float(p_skin) if np.isfinite(p_skin) else np.nan,
    }


def dimensionless_storage(C: float, h: float, phi: float, ct: float, rw: float) -> float:
    r"""
    Dimensionless wellbore storage :math:`C_D = 0.8936\, C / (\phi\, h\, c_t\, r_w^2)`
    """
    return 0.8936 * C / (phi * h * ct * rw ** 2)


def radius_of_investigation(
    k: float, t_hr: float, phi: float, mu: float, ct: float
) -> float:
    r"""
    Radius of investigation (Lee 1982): :math:`r_i = 0.0325 \sqrt{k\,t/(\phi\,\mu\,c_t)}`
    in feet (oilfield units).
    """
    if k <= 0 or t_hr <= 0 or phi <= 0 or mu <= 0 or ct <= 0:
        return np.nan
    return 0.0325 * np.sqrt(k * t_hr / (phi * mu * ct))
