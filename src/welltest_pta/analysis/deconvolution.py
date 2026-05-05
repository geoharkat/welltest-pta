r"""
welltest_pta.analysis.deconvolution
====================================
Multi-rate pressure-transient deconvolution.

The convolution problem for a variable-rate well test is

.. math::
    p_i - p_{wf}(t) = \int_0^t q(\tau)\, p_u'(t-\tau)\, d\tau

where :math:`p_u(t)` is the unit-rate constant-rate drawdown response and
:math:`p_u'(t) = dp_u/dt`. Deconvolution recovers :math:`p_u(t)` from the
measured :math:`p_{wf}(t)` and known rate history :math:`q(t)`. The result
is a **single equivalent buildup** that effectively merges all buildups
(and drawdowns) in the test, dramatically extending the radius of
investigation and revealing late-time boundaries that no individual
buildup is long enough to see.

Algorithm — von Schroeter–Hollaender–Gringarten (vSH04)
-------------------------------------------------------
Direct deconvolution is ill-posed (Hadamard-unstable to noise). vSH04
solve the regularised non-linear problem in the **encoded variable**

.. math::
    z(\sigma) \;=\; \ln\!\left[t\, \frac{dp_u}{dt}\right]
              \;=\; \ln\!\left[\frac{dp_u}{d\ln t}\right],
    \qquad \sigma = \ln t

so that :math:`p_u'(t) = e^{z(\ln t)}/t \ge 0` automatically (positivity
of the derivative enforced by construction). The objective is

.. math::
    J(z, p_i) \;=\; \|y - C(q,z) - p_i\|^2
                  \,+\, \nu \,\|D\,z\|^2

with :math:`D` a second-difference (curvature) operator and :math:`\nu`
the regularisation weight. The user specifies only :math:`\nu`
(typically :math:`10^{-3}\!-\!10^{-1}`); both :math:`z` and
:math:`p_i` are recovered.

References
----------
- von Schroeter, T., Hollaender, F., & Gringarten, A. C. (2004).
  *Deconvolution of well-test data as a nonlinear total least-squares
  problem.* SPE Journal **9** (4), 375–390.
- Levitan, M. M. (2005). *Practical application of pressure/rate
  deconvolution to analysis of real well tests.* SPE 84290.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeconvolutionResult:
    """Output of :func:`deconvolve`."""

    t: np.ndarray                    # response time grid (hr), log-spaced
    pu: np.ndarray                   # unit-rate cumulative response (psi per unit q)
    dpu_dlnt: np.ndarray             # log-derivative dp_u/d(ln t) (psi per unit q)
    z: np.ndarray                    # solved encoded variable
    p_initial: float                 # solved (or fixed) initial pressure (psi)
    nu: float                        # regularisation used
    converged: bool                  # solver convergence flag
    iterations: int                  # solver iterations
    residual_norm: float             # ||r||_2  (psi)
    fit_pressure: np.ndarray         # reconstructed p(t_obs) (psi)
    obs_pressure: np.ndarray         # observed p(t_obs) (psi)
    obs_time: np.ndarray             # observation time grid (hr from t=0)
    rate_history: pd.DataFrame = field(repr=False)  # input rate steps
    metadata: dict = field(default_factory=dict, repr=False)

    # ──────── methods ────────

    def to_dataframe(self) -> pd.DataFrame:
        """Long-form DataFrame of the recovered response."""
        return pd.DataFrame({
            "t_hr": self.t,
            "pu_psi_per_unit_q": self.pu,
            "dpu_dlnt_psi_per_unit_q": self.dpu_dlnt,
        })

    def export(self, path: str, format: str = "csv") -> None:
        """Save the response to CSV / Excel / JSON."""
        df = self.to_dataframe()
        path_lower = str(path).lower()
        if format == "csv" or path_lower.endswith(".csv"):
            df.to_csv(path, index=False)
        elif format == "excel" or path_lower.endswith((".xlsx", ".xls")):
            df.to_excel(path, index=False)
        elif format == "json" or path_lower.endswith(".json"):
            df.to_json(path, orient="records", date_format="iso")
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info("Deconvolution exported → %s", path)

    def plot(self, ax: "plt.Axes | None" = None, show_obs_fit: bool = False):
        """Log-log plot of the recovered :math:`p_u` and its derivative."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            fig = ax.figure
        ax.loglog(self.t, np.abs(self.pu), "o-", ms=3, lw=1.0,
                  color="#1f77b4", label=r"$p_u$")
        ax.loglog(self.t, np.abs(self.dpu_dlnt), "s-", ms=3, lw=1.0,
                  color="#d62728", label=r"$dp_u/d\ln t$")
        ax.set_xlabel(r"$\Delta t$ (hr)")
        ax.set_ylabel(r"Unit-rate response (psi per unit $q$)")
        ax.set_title("Deconvolved Unit-Rate Response (vSH04)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _build_log_grid(t_min: float, t_max: float, n: int) -> np.ndarray:
    """Geometric grid in time (log-spaced)."""
    if t_min <= 0 or t_max <= t_min:
        raise ValueError("t_min must be > 0 and t_max > t_min")
    return np.geomspace(t_min, t_max, n)


def _pu_from_z(z: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Given encoded variable :math:`z = \ln(dp_u/d\ln t)` on a log grid
    :math:`\sigma = \ln t`, return :math:`p_u(t)` and :math:`dp_u/d\ln t`.

    .. math::
        p_u(t_n) = \int_{-\infty}^{\sigma_n} e^{z(\sigma)}\, d\sigma
                 \approx \sum_{i \le n} e^{z_i}\, \Delta\sigma_i
    """
    dpdlnt = np.exp(z)
    dsigma = np.diff(sigma, prepend=sigma[0] - (sigma[1] - sigma[0]))
    pu = np.cumsum(dpdlnt * dsigma)
    return pu, dpdlnt


def _convolve_with_rates(
    t_obs: np.ndarray,
    rate_steps_t: np.ndarray,
    rate_steps_dq: np.ndarray,
    t_resp: np.ndarray,
    pu: np.ndarray,
) -> np.ndarray:
    r"""
    Compute :math:`\Delta p(t_j) = \sum_k \Delta q_k \cdot p_u(t_j - \tau_k)`
    using piecewise-linear interpolation on the log-spaced response grid.

    Parameters
    ----------
    t_obs
        Observation times (hr) from the start of the test (t=0).
    rate_steps_t
        Times at which rate changes occur (hr from t=0).
    rate_steps_dq
        Rate jumps :math:`\Delta q_k = q_k - q_{k-1}` at each step.
    t_resp, pu
        Unit-rate response on a log-spaced grid (must start at t_resp[0] > 0).
    """
    delta_p = np.zeros_like(t_obs, dtype=float)
    log_resp = np.log(t_resp)
    log_pu = np.log(np.maximum(pu, 1e-30))  # safe log
    for tau_k, dq_k in zip(rate_steps_t, rate_steps_dq):
        if dq_k == 0:
            continue
        dt_k = t_obs - tau_k
        active = dt_k > 0
        if not active.any():
            continue
        dt_active = dt_k[active]
        # Log-linear interpolation in time, but use linear pu directly
        log_dt = np.log(np.clip(dt_active, t_resp[0], t_resp[-1]))
        # numpy.interp handles outside-range as flat extrapolation
        pu_interp = np.interp(log_dt, log_resp, pu)
        delta_p[active] += dq_k * pu_interp
    return delta_p


def _second_difference_operator(n: int) -> np.ndarray:
    r"""Build the (n-2) × n centred second-difference matrix on a uniform grid."""
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


# ─────────────────────────────────────────────────────────────────────────────
# Rate-history helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rate_history_from_events(
    events,
    default_q: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build a rate-step DataFrame from a list of Event objects.

    Each event must have ``rate`` set (drawdowns get q, buildups get 0).
    If ``default_q`` is supplied, drawdowns without explicit rate use it.
    """
    rows = []
    last_q = 0.0
    for ev in events:
        q = ev.rate
        if q is None:
            if ev.event_type == "buildup":
                q = 0.0
            else:
                if default_q is None:
                    raise ValueError(
                        f"Event {ev.event_id} ({ev.event_type}) has no rate; "
                        f"set ev.rate or pass default_q."
                    )
                q = default_q
        dq = q - last_q
        if dq != 0.0:
            rows.append({"t_hr": float(ev.elapsed_start_hr), "q": float(q), "dq": float(dq)})
        last_q = q
    if not rows:
        # Fallback: single step at t=0 with q=0
        rows.append({"t_hr": 0.0, "q": 0.0, "dq": 0.0})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def deconvolve(
    events,
    rate_history: pd.DataFrame | dict | None = None,
    default_q: Optional[float] = None,
    nu: float = 1e-2,
    n_response_nodes: int = 60,
    t_response_min: float = 1e-3,
    t_response_max: Optional[float] = None,
    p_initial: Optional[float] = None,
    fit_p_initial: bool = True,
    max_iter: int = 200,
    verbose: bool = False,
) -> DeconvolutionResult:
    r"""
    Recover the unit-rate constant-rate response from multi-rate data.

    Implements the encoded vSH04 formulation. Suitable for merging two or
    more buildups/drawdowns into a single equivalent buildup whose log–log
    derivative is the diagnostic "master plot" of the entire test.

    Parameters
    ----------
    events
        Iterable of :class:`Event` (or ``EventCollection``). Both drawdowns
        and buildups are used. Each event must carry rate information
        (``ev.rate`` in STB/D — or set ``default_q``).
    rate_history
        Optional pre-built rate history. DataFrame with columns
        ``t_hr``, ``q`` (rate at each step). If ``None``, derived from
        events.
    default_q
        Fallback flow rate (STB/D) for drawdowns without an explicit
        ``ev.rate``. Buildups always get q=0.
    nu
        Regularisation weight for curvature of :math:`z`. Larger ⇒ smoother
        response (more bias). Smaller ⇒ noisier (more variance). Typical
        :math:`10^{-3}` (low noise) to :math:`10^{-1}` (high noise).
    n_response_nodes
        Number of log-spaced nodes on the recovered response grid.
    t_response_min, t_response_max
        Time range (hr) of the recovered response. ``t_response_max``
        defaults to the maximum observation time.
    p_initial
        Initial reservoir pressure (psi). If ``None`` and
        ``fit_p_initial=True``, it is solved as an additional unknown.
    fit_p_initial
        Whether to recover :math:`p_i` simultaneously with :math:`z`.
    max_iter
        Maximum solver iterations (Levenberg–Marquardt).
    verbose
        Print solver progress.

    Returns
    -------
    :class:`DeconvolutionResult`

    Notes
    -----
    For best results: include both a long buildup and the immediately
    preceding drawdown; clean buildup tails (V8.1 detector does this);
    use consistent rate units throughout (typically STB/D).
    """
    # ── 1. Assemble (t, p) observation arrays from all PTA events ──
    t_obs_list, p_obs_list = [], []
    for ev in events:
        df_ev = ev.data
        if df_ev is None or df_ev.empty:
            continue
        t_obs_list.append(df_ev["elapsed_hr"].to_numpy())
        p_obs_list.append(df_ev["p_smooth"].to_numpy())
    if not t_obs_list:
        raise ValueError("No event data available for deconvolution.")
    t_obs = np.concatenate(t_obs_list)
    p_obs = np.concatenate(p_obs_list)
    order = np.argsort(t_obs)
    t_obs = t_obs[order]
    p_obs = p_obs[order]

    # ── 2. Build rate history (step changes Δq_k at τ_k) ──
    if rate_history is None:
        rate_history = _rate_history_from_events(events, default_q=default_q)
    elif isinstance(rate_history, dict):
        rate_history = pd.DataFrame(rate_history)
    if "dq" not in rate_history.columns:
        # build dq from q
        q_arr = rate_history["q"].to_numpy()
        rate_history = rate_history.copy()
        rate_history["dq"] = np.concatenate([[q_arr[0]], np.diff(q_arr)])

    rate_t = rate_history["t_hr"].to_numpy()
    rate_dq = rate_history["dq"].to_numpy()

    # ── 3. Build log-spaced response grid ──
    t_max = t_response_max if t_response_max is not None else float(np.max(t_obs))
    t_resp = _build_log_grid(t_response_min, t_max, n_response_nodes)
    sigma = np.log(t_resp)

    # ── 4. Initial guess for z (constant — flat radial response) ──
    # Choose a level so that p_u at t_max is roughly p_obs - p_i normalized
    p_i_guess = float(p_initial) if p_initial is not None else float(np.percentile(p_obs, 95))
    typical_dp = max(p_i_guess - float(np.min(p_obs)), 50.0)
    typical_q = float(np.max(np.abs(rate_dq))) if (np.abs(rate_dq) > 0).any() else 1.0
    pu_guess = typical_dp / max(typical_q, 1e-6)
    # p_u(t_max) ≈ pu_guess  →  exp(z) * (sigma_max - sigma_min) ≈ pu_guess
    z0_const = np.log(max(pu_guess / (sigma[-1] - sigma[0] + 1e-9), 1e-6))
    z0 = np.full(n_response_nodes, z0_const)

    # ── 5. Build regularisation matrix ──
    D = _second_difference_operator(n_response_nodes)
    sqrt_nu = np.sqrt(nu)

    # ── 6. Residual function ──
    def residuals(params: np.ndarray) -> np.ndarray:
        if fit_p_initial and p_initial is None:
            z = params[:-1]
            p_i = params[-1]
        else:
            z = params
            p_i = p_i_guess if p_initial is None else float(p_initial)
        pu, _ = _pu_from_z(z, sigma)
        delta_p_pred = _convolve_with_rates(t_obs, rate_t, rate_dq, t_resp, pu)
        # In a buildup: p_obs grows, model: p_i + delta_p (delta_p < 0 if drawing down)
        # Convention: rates are positive when producing → delta_p contributions are NEGATIVE (drawdown)
        # We want residual = obs - model:
        p_model = p_i - delta_p_pred  # drawdown reduces pressure
        r_data = p_obs - p_model
        r_reg = sqrt_nu * D @ z
        return np.concatenate([r_data, r_reg])

    # ── 7. Initial parameter vector ──
    if fit_p_initial and p_initial is None:
        x0 = np.concatenate([z0, [p_i_guess]])
    else:
        x0 = z0.copy()

    # ── 8. Solve ──
    result = least_squares(
        residuals,
        x0=x0,
        method="lm",
        max_nfev=max_iter * (len(x0) + 1),
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        verbose=2 if verbose else 0,
    )

    # ── 9. Unpack solution ──
    if fit_p_initial and p_initial is None:
        z_sol = result.x[:-1]
        p_i_sol = float(result.x[-1])
    else:
        z_sol = result.x
        p_i_sol = p_i_guess if p_initial is None else float(p_initial)

    pu_sol, dpu_dlnt_sol = _pu_from_z(z_sol, sigma)
    delta_p_pred = _convolve_with_rates(t_obs, rate_t, rate_dq, t_resp, pu_sol)
    p_fit = p_i_sol - delta_p_pred
    res_norm = float(np.linalg.norm(p_obs - p_fit))

    return DeconvolutionResult(
        t=t_resp,
        pu=pu_sol,
        dpu_dlnt=dpu_dlnt_sol,
        z=z_sol,
        p_initial=p_i_sol,
        nu=nu,
        converged=bool(result.success),
        iterations=int(result.nfev),
        residual_norm=res_norm,
        fit_pressure=p_fit,
        obs_pressure=p_obs,
        obs_time=t_obs,
        rate_history=rate_history,
        metadata={
            "n_response_nodes": n_response_nodes,
            "t_response_min": t_response_min,
            "t_response_max": t_max,
            "fit_p_initial": fit_p_initial,
        },
    )
