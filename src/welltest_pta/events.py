r"""
welltest_pta.events
===================
First-class :class:`Event` objects + the :class:`EventCollection` container.

After event detection (or manual splitting) every drawdown / buildup is
exposed as a self-contained ``Event`` carrying its slice of the gauge
data plus all PTA methods relevant to that single event:

==================  ============================================================
Method              Returns
==================  ============================================================
``.print()``        Pretty-printed summary to stdout
``.summary()``      Dict of statistics (duration, ΔP, slopes, …)
``.export(...)``    CSV / Excel / JSON of this event's slice
``.plot()``         Pressure vs time (single-event view)
``.plot_loglog()``  Bourdet log–log diagnostic
``.plot_horner()``  Horner plot (buildups only)
``.plot_mdh()``     MDH semi-log plot (buildups only)
``.bourdet(...)``   Returns (dt, derivative) numpy arrays
``.horner()``       Dict from :func:`horner_extrapolation`
``.mdh()``          Dict from :func:`mdh_extrapolation`
``.flow_regimes()`` List of identified flow regimes
``.reservoir_params(...)``  k, kh, S, C from this event alone
==================  ============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from welltest_pta.analysis.bourdet import bourdet_derivative
from welltest_pta.analysis.flow_regimes import identify_flow_regimes
from welltest_pta.analysis.horner import horner_extrapolation
from welltest_pta.analysis.mdh import mdh_extrapolation
from welltest_pta.analysis.reservoir import reservoir_parameters

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


EVENT_COLOURS = {
    "buildup": "#2ca02c",
    "drawdown": "#d62728",
    "non_pta": "#cccccc",
}


# ─────────────────────────────────────────────────────────────────────────────
# Event
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """A single PTA event (drawdown or buildup) with its data slice and methods."""

    event_id: str                    # e.g. "DD-1", "BU-2"
    event_type: str                  # "drawdown" or "buildup"
    idx_start: int                   # row index in the parent annotated DataFrame
    idx_end: int                     # exclusive
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    elapsed_start_hr: float          # hours from t=0 of the test
    elapsed_end_hr: float
    data: pd.DataFrame = field(repr=False)  # slice of parent DataFrame for this event
    rate: Optional[float] = None     # flow rate during this event (STB/D)
    preceding_dd_dur_hr: Optional[float] = None
    parent_p_reservoir: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    # ──────── basic stats ────────

    @property
    def duration_hr(self) -> float:
        return float(self.elapsed_end_hr - self.elapsed_start_hr)

    @property
    def p_initial(self) -> float:
        return float(self.data["p_smooth"].iloc[0])

    @property
    def p_final(self) -> float:
        return float(self.data["p_smooth"].iloc[-1])

    @property
    def delta_p(self) -> float:
        return self.p_final - self.p_initial

    @property
    def rate_psi_per_hr(self) -> float:
        return self.delta_p / self.duration_hr if self.duration_hr > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        """Return a summary dictionary of this event's statistics."""
        return {
            "event_id": self.event_id,
            "type": self.event_type,
            "idx_start": self.idx_start,
            "idx_end": self.idx_end,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "duration_hr": round(self.duration_hr, 4),
            "p_initial": round(self.p_initial, 2),
            "p_final": round(self.p_final, 2),
            "delta_p": round(self.delta_p, 2),
            "rate_psi_hr": round(self.rate_psi_per_hr, 2),
            "p_max": round(float(self.data["p_smooth"].max()), 2),
            "p_min": round(float(self.data["p_smooth"].min()), 2),
            "preceding_dd_dur_hr": (
                round(self.preceding_dd_dur_hr, 4)
                if self.preceding_dd_dur_hr is not None else None
            ),
            "rate_stb_d": self.rate,
            "n_points": len(self.data),
        }

    def print(self) -> None:
        """Pretty-print the event summary."""
        s = self.summary()
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Event {s['event_id']}  ({s['type']})")
        print(sep)
        print(f"  Time range:    {s['t_start']}  →  {s['t_end']}")
        print(f"  Duration:      {s['duration_hr']:.4f} hr")
        print(f"  P_initial:     {s['p_initial']:.2f} psi")
        print(f"  P_final:       {s['p_final']:.2f} psi")
        print(f"  ΔP:            {s['delta_p']:+.2f} psi")
        print(f"  Slope:         {s['rate_psi_hr']:.2f} psi/hr")
        print(f"  Samples:       {s['n_points']}")
        if s.get("preceding_dd_dur_hr") is not None:
            print(f"  tp (prev DD):  {s['preceding_dd_dur_hr']:.4f} hr")
        if self.rate is not None:
            print(f"  Flow rate q:   {self.rate:.2f} STB/D")
        print(sep)

    # ──────── export ────────

    def export(self, path: str | Path, format: str = "auto") -> None:
        """Save this event's data slice to CSV / Excel / JSON."""
        path = Path(path)
        if format == "auto":
            ext = path.suffix.lower()
            if ext == ".csv":
                format = "csv"
            elif ext in (".xlsx", ".xls"):
                format = "excel"
            elif ext == ".json":
                format = "json"
            else:
                format = "csv"
        if format == "csv":
            self.data.to_csv(path, index=False)
        elif format == "excel":
            self.data.to_excel(path, index=False)
        elif format == "json":
            self.data.to_json(path, orient="records", date_format="iso")
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info("Event %s exported → %s", self.event_id, path)

    # ──────── analysis methods ────────

    def bourdet(self, L: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Return (Δt, Bourdet derivative) for this event."""
        seg_p = self.data["p_smooth"].to_numpy()
        seg_hr = self.data["elapsed_hr"].to_numpy()
        dt = seg_hr - seg_hr[0]
        dt[0] = 1e-6
        dp = seg_p - seg_p[0]
        return bourdet_derivative(dt, dp, L=L)

    def horner(
        self,
        fit_start_frac: float = 0.30,
        fit_end_frac: float = 0.85,
    ) -> dict[str, float]:
        """Horner P\\* extrapolation. Buildup only."""
        if self.event_type != "buildup":
            raise ValueError("Horner analysis applies to buildups only.")
        if self.preceding_dd_dur_hr is None or self.preceding_dd_dur_hr <= 0:
            raise ValueError("Cannot run Horner: preceding_dd_dur_hr unknown or non-positive.")
        return horner_extrapolation(
            self.data["p_smooth"].to_numpy(),
            self.data["elapsed_hr"].to_numpy(),
            tp_hr=self.preceding_dd_dur_hr,
            fit_start_frac=fit_start_frac,
            fit_end_frac=fit_end_frac,
        )

    def mdh(
        self,
        fit_start_frac: float = 0.30,
        fit_end_frac: float = 0.85,
    ) -> dict[str, float]:
        """MDH semi-log analysis. Buildup only."""
        if self.event_type != "buildup":
            raise ValueError("MDH analysis applies to buildups only.")
        return mdh_extrapolation(
            self.data["p_smooth"].to_numpy(),
            self.data["elapsed_hr"].to_numpy(),
            fit_start_frac=fit_start_frac,
            fit_end_frac=fit_end_frac,
        )

    def flow_regimes(
        self,
        L: float = 0.2,
        slope_tol: float = 0.10,
    ) -> list[dict[str, Any]]:
        """Auto-identify flow regimes from the Bourdet derivative."""
        dt, deriv = self.bourdet(L=L)
        return identify_flow_regimes(dt, deriv, slope_tol=slope_tol)

    def reservoir_params(
        self,
        q: float,
        mu: float,
        B: float,
        h: float,
        phi: float,
        ct: float,
        rw: float,
        method: str = "horner",
    ) -> dict[str, float]:
        """
        Compute :math:`k`, :math:`kh`, skin :math:`S` (and :math:`C` if
        early-storage slope detected) for this buildup.

        Parameters
        ----------
        q, mu, B, h, phi, ct, rw
            Fluid / reservoir / well properties (see
            :func:`welltest_pta.analysis.reservoir.reservoir_parameters`).
        method
            ``"horner"`` (default) or ``"mdh"``.
        """
        if self.event_type != "buildup":
            raise ValueError("Reservoir parameters from semilog analysis require a buildup.")
        if method == "horner":
            res = self.horner()
            slope = res["slope_m"]
            # p_1hr from Horner line:  p_1hr = m * log10((tp+1)/1) + p_star
            tp = self.preceding_dd_dur_hr or 0.0
            p_1hr = slope * np.log10((tp + 1.0) / 1.0) + res["p_star"]
        elif method == "mdh":
            res = self.mdh()
            slope = res["slope_m"]
            p_1hr = res["intercept_p1hr"]
        else:
            raise ValueError("method must be 'horner' or 'mdh'")

        # p_wf at instant of shut-in = pressure at start of buildup
        p_wf = self.p_initial

        # Try to estimate early-time storage slope (psi/hr) from first 5% of data
        early_slope = None
        try:
            n_early = max(5, int(len(self.data) * 0.05))
            t_early = self.data["elapsed_hr"].iloc[:n_early].to_numpy()
            p_early = self.data["p_smooth"].iloc[:n_early].to_numpy()
            dt_early = t_early - t_early[0]
            dp_early = p_early - p_early[0]
            if len(dt_early) >= 3 and dt_early[-1] > 0:
                A = np.vstack([dt_early, np.ones_like(dt_early)]).T
                m_early, _ = np.linalg.lstsq(A, dp_early, rcond=None)[0]
                if m_early > 0:
                    early_slope = float(m_early)
        except Exception:
            pass

        return reservoir_parameters(
            slope_m=slope,
            p_1hr=p_1hr,
            p_wf_at_shut_in=p_wf,
            q=q, mu=mu, B=B, h=h, phi=phi, ct=ct, rw=rw,
            early_storage_slope=early_slope,
        )

    # ──────── plotting ────────

    def plot(self, ax: "plt.Axes | None" = None, raw: bool = True):
        """Pressure vs time for this single event."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.figure
        if raw and "pressure" in self.data.columns:
            ax.plot(self.data["timestamp"], self.data["pressure"],
                    color="#b0b0b0", lw=0.4, alpha=0.6, label="Raw")
        ax.plot(self.data["timestamp"], self.data["p_smooth"],
                color=EVENT_COLOURS.get(self.event_type, "#1f77b4"),
                lw=1.2, label=f"{self.event_id} ({self.event_type})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Pressure (psi)")
        ax.set_title(f"{self.event_id} — {self.event_type}", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    def plot_loglog(self, ax: "plt.Axes | None" = None, L: float = 0.2):
        """Log–log diagnostic plot (|ΔP| and Bourdet derivative)."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            fig = ax.figure
        seg_p = self.data["p_smooth"].to_numpy()
        seg_hr = self.data["elapsed_hr"].to_numpy()
        dt = seg_hr - seg_hr[0]
        dt[0] = 1e-6
        dp = seg_p - seg_p[0]
        ax.loglog(dt, np.abs(dp), "o-", ms=2, lw=0.8,
                  color="#1f77b4", label=r"$|\Delta P|$")
        dt_d, deriv_d = bourdet_derivative(dt, dp, L=L)
        if len(dt_d) > 3:
            ax.loglog(dt_d, deriv_d, "s-", ms=2, lw=0.8,
                      color="#d62728", label="Bourdet derivative")
        ax.set_xlabel(r"$\Delta t$ (hr)")
        ax.set_ylabel(r"$|\Delta P|$, $dP/d(\ln t)$ (psi)")
        ax.set_title(f"Log–Log Diagnostic — {self.event_id}", fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right")
        return fig

    def plot_horner(self, ax: "plt.Axes | None" = None):
        """Horner plot (buildups only)."""
        if self.event_type != "buildup":
            raise ValueError("Horner plot requires a buildup.")
        if self.preceding_dd_dur_hr is None or self.preceding_dd_dur_hr <= 0:
            raise ValueError("preceding_dd_dur_hr is not set.")
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.figure
        seg_p = self.data["p_smooth"].to_numpy()
        seg_hr = self.data["elapsed_hr"].to_numpy()
        dt = seg_hr - seg_hr[0]
        dt = np.where(dt <= 0, 1e-6, dt)
        tp = self.preceding_dd_dur_hr
        horner_x = (tp + dt) / dt
        ax.semilogx(horner_x, seg_p, "o-", ms=2, lw=0.8, color="#1f77b4",
                    label=self.event_id)
        # IARF fit annotation
        try:
            res = self.horner()
            ax.axhline(res["p_star"], color="#006400", ls="--", lw=1.5,
                       label=fr"$P^*$ = {res['p_star']:.1f} psi  ($R^2$ = {res['r2']:.4f})")
        except Exception:
            pass
        ax.set_xlabel(r"Horner time  $(t_p + \Delta t)/\Delta t$")
        ax.set_ylabel(r"$p_{ws}$ (psi)")
        ax.invert_xaxis()
        ax.set_title(f"Horner Plot — {self.event_id} (tp = {tp:.2f} hr)",
                     fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        return fig

    def plot_mdh(self, ax: "plt.Axes | None" = None):
        """MDH semi-log plot (buildups only)."""
        if self.event_type != "buildup":
            raise ValueError("MDH plot requires a buildup.")
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.figure
        seg_p = self.data["p_smooth"].to_numpy()
        seg_hr = self.data["elapsed_hr"].to_numpy()
        dt = seg_hr - seg_hr[0]
        dt = np.where(dt <= 0, 1e-6, dt)
        ax.semilogx(dt, seg_p, "o-", ms=2, lw=0.8, color="#1f77b4",
                    label=self.event_id)
        try:
            res = self.mdh()
            ax.axhline(res["intercept_p1hr"], color="#006400", ls="--", lw=1.5,
                       label=fr"$p_{{1hr}}$ = {res['intercept_p1hr']:.1f} psi")
        except Exception:
            pass
        ax.set_xlabel(r"$\Delta t$ (hr)")
        ax.set_ylabel(r"$p_{ws}$ (psi)")
        ax.set_title(f"MDH Plot — {self.event_id}", fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        return fig

    def __repr__(self) -> str:
        return (
            f"Event(id={self.event_id!r}, type={self.event_type!r}, "
            f"duration_hr={self.duration_hr:.3f}, "
            f"ΔP={self.delta_p:+.1f} psi, n={len(self.data)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# EventCollection
# ─────────────────────────────────────────────────────────────────────────────

class EventCollection:
    """
    An ordered, indexable container of :class:`Event` objects.

    Behaves like a list with extra convenience methods:

    >>> wt.events           # EventCollection
    >>> wt.events[0]        # first event
    >>> wt.events["BU-2"]   # by event_id
    >>> for e in wt.events: ...
    >>> len(wt.events)
    """

    def __init__(self, events: Iterable[Event] = ()):
        self._events: list[Event] = list(events)

    # ── container protocol ──

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __getitem__(self, key) -> Event | "EventCollection":
        if isinstance(key, int):
            return self._events[key]
        if isinstance(key, slice):
            return EventCollection(self._events[key])
        if isinstance(key, str):
            for e in self._events:
                if e.event_id == key:
                    return e
            raise KeyError(f"No event with id {key!r}")
        raise TypeError(f"Bad key type: {type(key)}")

    def __repr__(self) -> str:
        n_dd = sum(1 for e in self._events if e.event_type == "drawdown")
        n_bu = sum(1 for e in self._events if e.event_type == "buildup")
        return f"EventCollection({len(self._events)} events: {n_dd} DD, {n_bu} BU)"

    # ── filters ──

    @property
    def drawdowns(self) -> "EventCollection":
        return EventCollection(e for e in self._events if e.event_type == "drawdown")

    @property
    def buildups(self) -> "EventCollection":
        return EventCollection(e for e in self._events if e.event_type == "buildup")

    @property
    def longest_buildup(self) -> Optional[Event]:
        bus = list(self.buildups)
        return max(bus, key=lambda e: e.duration_hr) if bus else None

    # ── tabular ──

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.summary() for e in self._events])

    def print(self) -> None:
        df = self.to_dataframe()
        if df.empty:
            print("(no events)")
            return
        cols = ["event_id", "type", "duration_hr", "p_initial", "p_final",
                "delta_p", "rate_psi_hr", "n_points"]
        print(df[cols].to_string(index=False, float_format="%.2f"))

    def export(self, path: str | Path, format: str = "auto") -> None:
        """Export the catalogue (one row per event)."""
        path = Path(path)
        df = self.to_dataframe()
        if format == "auto":
            ext = path.suffix.lower()
            if ext == ".csv":
                format = "csv"
            elif ext in (".xlsx", ".xls"):
                format = "excel"
            elif ext == ".json":
                format = "json"
            else:
                format = "csv"
        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "excel":
            df.to_excel(path, index=False)
        elif format == "json":
            df.to_json(path, orient="records", date_format="iso")
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.info("Event catalogue exported → %s", path)

    # ── factories ──

    @classmethod
    def from_annotated_dataframe(
        cls,
        df: pd.DataFrame,
        p_reservoir: Optional[float] = None,
    ) -> "EventCollection":
        """
        Build an :class:`EventCollection` from a DataFrame annotated by the
        :class:`EventDetector` (i.e. with an ``event`` column).
        """
        if "event" not in df.columns:
            raise ValueError("DataFrame missing 'event' column — run the detector first.")
        if "p_smooth" not in df.columns or "elapsed_hr" not in df.columns:
            raise ValueError("DataFrame missing 'p_smooth' / 'elapsed_hr' — run the detector.")

        labels = df["event"]
        groups, cur, start = [], labels.iloc[0], 0
        for i in range(1, len(labels)):
            if labels.iloc[i] != cur:
                groups.append((start, i, cur))
                cur, start = labels.iloc[i], i
        groups.append((start, len(labels), cur))

        events: list[Event] = []
        dd_n = bu_n = 0
        prev_dd_dur: Optional[float] = None

        for s, e, lbl in groups:
            if lbl not in ("drawdown", "buildup"):
                continue
            if lbl == "drawdown":
                dd_n += 1
                eid = f"DD-{dd_n}"
            else:
                bu_n += 1
                eid = f"BU-{bu_n}"
            sub = df.iloc[s:e].reset_index(drop=True).copy()
            ev = Event(
                event_id=eid,
                event_type=lbl,
                idx_start=int(s),
                idx_end=int(e),
                t_start=df["timestamp"].iloc[s],
                t_end=df["timestamp"].iloc[min(e - 1, len(df) - 1)],
                elapsed_start_hr=float(df["elapsed_hr"].iloc[s]),
                elapsed_end_hr=float(df["elapsed_hr"].iloc[min(e - 1, len(df) - 1)]),
                data=sub,
                preceding_dd_dur_hr=prev_dd_dur if lbl == "buildup" else None,
                parent_p_reservoir=p_reservoir,
            )
            events.append(ev)
            if lbl == "drawdown":
                prev_dd_dur = ev.duration_hr

        return cls(events)
