"""
welltest_pta.analysis.flow_regimes
==================================
Automatic identification of flow regimes from the Bourdet derivative.

The log–log slope of the derivative is the diagnostic signature:

==============================  ==========  =========================================
Regime                          Slope       Physical interpretation
==============================  ==========  =========================================
Wellbore storage (WBS)          +1          Compressibility of fluid in tubing/annulus
Bilinear flow                   +¼          Finite-conductivity hydraulic fracture
Linear flow                     +½          Infinite-conductivity fracture / channel
Infinite-acting radial (IARF)    0          Cylindrical radial flow in homogeneous media
Spherical flow                  −½          Limited-entry / partially-penetrating well
Pseudo-steady (closed bndry)    +1          All boundaries reached
Constant-pressure boundary      Sharp drop  Aquifer support / gas cap
==============================  ==========  =========================================

This module fits piece-wise log–log slopes to the derivative curve and
reports the dominant regime in each segment, along with goodness-of-fit
and the time-window covered.
"""

from __future__ import annotations

import numpy as np

# Reference slopes for each regime
REGIME_SLOPES = {
    "wellbore_storage": 1.0,
    "bilinear": 0.25,
    "linear": 0.5,
    "iarf": 0.0,
    "spherical": -0.5,
    "boundary_closed": 1.0,
    "boundary_constant_p": -1.0,  # rough — actual signature is steep drop
}


def _moving_slope(log_x: np.ndarray, log_y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving slope of log_y vs log_x (least-squares per window)."""
    n = len(log_x)
    slopes = np.full(n, np.nan)
    half = window // 2
    for i in range(half, n - half):
        x_w = log_x[i - half: i + half + 1]
        y_w = log_y[i - half: i + half + 1]
        if len(x_w) < 3:
            continue
        m, _ = np.polyfit(x_w, y_w, 1)
        slopes[i] = m
    return slopes


def identify_flow_regimes(
    dt: np.ndarray,
    deriv: np.ndarray,
    window_decades: float = 0.4,
    slope_tol: float = 0.10,
    min_segment_pts: int = 3,
) -> list[dict]:
    """
    Segment the Bourdet derivative into flow-regime intervals.

    Parameters
    ----------
    dt
        Elapsed time array (hr) — the output of :func:`bourdet_derivative`.
    deriv
        Bourdet derivative magnitude (psi).
    window_decades
        Width of the moving-window slope estimator, in log decades. The
        window length in points is set so that it spans roughly this many
        decades on the log axis.
    slope_tol
        Tolerance around the reference slope to call a match (e.g. 0.10
        means slopes in [reference−0.10, reference+0.10]).
    min_segment_pts
        Minimum contiguous points to accept a regime segment.

    Returns
    -------
    list of dicts, each with keys
    ``regime``, ``slope_mean``, ``r2``, ``dt_start``, ``dt_end``, ``n_pts``.
    """
    dt = np.asarray(dt, dtype=float)
    deriv = np.asarray(deriv, dtype=float)
    if len(dt) < 5:
        return []
    mask = (dt > 0) & (deriv > 0) & np.isfinite(deriv)
    dt = dt[mask]
    deriv = deriv[mask]
    if len(dt) < 5:
        return []

    log_x = np.log10(dt)
    log_y = np.log10(deriv)

    # Pick window length to cover ~window_decades on x
    span = log_x[-1] - log_x[0]
    if span <= 0:
        return []
    pts_per_decade = len(log_x) / span
    window = max(5, int(window_decades * pts_per_decade))
    if window % 2 == 0:
        window += 1
    if window >= len(log_x):
        window = max(5, len(log_x) // 3)
        if window % 2 == 0:
            window += 1

    slopes = _moving_slope(log_x, log_y, window)

    # Classify each point by closest reference slope (within tolerance)
    classifications = np.full(len(slopes), "unknown", dtype=object)
    for i, s in enumerate(slopes):
        if not np.isfinite(s):
            continue
        # Pick the closest regime within tolerance
        best, best_diff = None, np.inf
        for name, ref in REGIME_SLOPES.items():
            d = abs(s - ref)
            if d < best_diff and d <= slope_tol:
                best, best_diff = name, d
        if best is None:
            classifications[i] = "transition"
        else:
            classifications[i] = best

    # Run-length encode + filter short segments
    segments: list[dict] = []
    if len(classifications) == 0:
        return segments

    cur, cur_start = classifications[0], 0
    for i in range(1, len(classifications)):
        if classifications[i] != cur:
            seg = _build_segment(
                cur, cur_start, i, dt, slopes, log_x, log_y, min_segment_pts
            )
            if seg is not None:
                segments.append(seg)
            cur, cur_start = classifications[i], i
    seg = _build_segment(
        cur, cur_start, len(classifications), dt, slopes, log_x, log_y, min_segment_pts
    )
    if seg is not None:
        segments.append(seg)

    return segments


def _build_segment(
    name: str,
    s: int,
    e: int,
    dt: np.ndarray,
    slopes: np.ndarray,
    log_x: np.ndarray,
    log_y: np.ndarray,
    min_segment_pts: int,
) -> dict | None:
    if e - s < min_segment_pts or name in ("unknown", "transition"):
        return None
    seg_slopes = slopes[s:e]
    seg_slopes = seg_slopes[np.isfinite(seg_slopes)]
    if len(seg_slopes) == 0:
        return None
    # R² of full log-log linear fit on the segment
    x_seg = log_x[s:e]
    y_seg = log_y[s:e]
    if len(x_seg) >= 3:
        m_lin, b_lin = np.polyfit(x_seg, y_seg, 1)
        y_pred = m_lin * x_seg + b_lin
        ss_res = float(np.sum((y_seg - y_pred) ** 2))
        ss_tot = float(np.sum((y_seg - y_seg.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        r2 = np.nan
    return {
        "regime": name,
        "slope_mean": float(np.mean(seg_slopes)),
        "r2": float(r2),
        "dt_start": float(dt[s]),
        "dt_end": float(dt[e - 1]),
        "n_pts": int(e - s),
    }
