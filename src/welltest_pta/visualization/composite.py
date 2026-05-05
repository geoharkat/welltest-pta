r"""
welltest_pta.visualization.composite
====================================
Publication-quality multi-panel figures.

* :func:`plot_overview`        — single panel, p(t) with event shading
* :func:`plot_composite_report`— 4-panel composite:
   1. Pressure timeline with events + P_res lines
   2. Temperature overlay (if available)
   3. Log–log diagnostic of the longest buildup
   4. Buildup-pressure histogram + KDE mode
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from welltest_pta.welltest import WellTest


EVENT_COLOURS = {
    "buildup": "#2ca02c",
    "drawdown": "#d62728",
    "non_pta": "#cccccc",
}


def apply_publication_style() -> None:
    """Apply publication-quality matplotlib defaults (serif, 150 dpi)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _rle(labels: pd.Series) -> list[tuple[int, int, str]]:
    groups, cur, start = [], labels.iloc[0], 0
    for i in range(1, len(labels)):
        if labels.iloc[i] != cur:
            groups.append((start, i, cur))
            cur, start = labels.iloc[i], i
    groups.append((start, len(labels), cur))
    return groups


def _span_events(ax, df: pd.DataFrame) -> None:
    groups = _rle(df["event"])
    added: set[str] = set()
    for s, e, lbl in groups:
        c = EVENT_COLOURS.get(lbl, "#eeeeee")
        label = lbl if lbl not in added else None
        ax.axvspan(
            df["timestamp"].iloc[s],
            df["timestamp"].iloc[min(e - 1, len(df) - 1)],
            color=c, alpha=0.15, label=label,
        )
        if label:
            added.add(lbl)


def _kde_mode(values: np.ndarray) -> tuple[float, float, float]:
    """Return (kde_mode, hist_mode, std). Empty -> (nan, nan, nan)."""
    from scipy.stats import gaussian_kde
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return np.nan, np.nan, np.nan
    n_bins = max(50, int(np.sqrt(len(v))))
    counts, edges = np.histogram(v, bins=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    hist_mode = float(centres[np.argmax(counts)])
    try:
        kde = gaussian_kde(v, bw_method="silverman")
        x = np.linspace(v.min(), v.max(), 2000)
        kde_mode = float(x[np.argmax(kde(x))])
    except Exception:
        kde_mode = hist_mode
    return kde_mode, hist_mode, float(np.std(v))


def plot_overview(
    wt: "WellTest",
    ax=None,
    show_events: bool = True,
    show_p_res: bool = True,
):
    """Single-panel pressure-vs-time overview."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    apply_publication_style()
    if wt.df is None:
        raise RuntimeError("Run wt.detect() before plotting.")

    df = wt.df
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))
    else:
        fig = ax.figure

    if show_events and "event" in df.columns:
        _span_events(ax, df)

    if "pressure" in df.columns:
        ax.plot(df["timestamp"], df["pressure"], color="#b0b0b0",
                lw=0.4, alpha=0.6, label="Raw")
    ax.plot(df["timestamp"], df["p_smooth"], color="#1f77b4",
            lw=1.0, label="Smoothed")

    if show_p_res and wt.p_reservoir is not None:
        ax.axhline(wt.p_reservoir, color="#ff8c00", ls=":", lw=1.2,
                   label=fr"$P_{{res}}$ = {wt.p_reservoir:.1f} psi")

    # Annotate events
    for ev in wt.events:
        mid_t = ev.t_start + (ev.t_end - ev.t_start) / 2
        mid_p = (ev.p_initial + ev.p_final) / 2
        ax.annotate(
            ev.event_id, xy=(mid_t, mid_p),
            fontsize=7, fontweight="bold", color="k",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="gray", alpha=0.80, lw=0.5),
        )

    ax.set_ylabel("Pressure (psi)")
    ax.set_title(
        f"Well Test — {wt.metadata.get('filename', 'Overview')}",
        fontweight="bold",
    )
    ax.legend(loc="upper right", ncol=2, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d-%b"))
    return fig


def plot_composite_report(
    wt: "WellTest",
    out_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (14, 12),
):
    """
    Four-panel composite report:

    [0] Pressure timeline with event shading
    [1] Temperature (if available)
    [2] Log–log diagnostic of the longest buildup
    [3] Buildup-pressure histogram with KDE mode
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    from welltest_pta.analysis.bourdet import bourdet_derivative

    apply_publication_style()
    if wt.df is None:
        raise RuntimeError("Run wt.detect() before plotting.")
    df = wt.df

    has_temp = "T_smooth" in df.columns and df["T_smooth"].notna().any()
    n_rows = 4 if has_temp else 3
    ratios = [4, 1.2, 2, 2] if has_temp else [4, 2, 2]

    fig, axes = plt.subplots(
        n_rows, 1, figsize=figsize,
        gridspec_kw={"height_ratios": ratios, "hspace": 0.35},
    )

    idx = 0

    # ── Panel 0: Pressure timeline ──────────────────────────────────────
    ax = axes[idx]; idx += 1
    _span_events(ax, df)
    ax.plot(df["timestamp"], df["pressure"], color="#b0b0b0",
            lw=0.4, alpha=0.6, label="Raw")
    ax.plot(df["timestamp"], df["p_smooth"], color="#1f77b4",
            lw=1.0, label="Smoothed")

    bu_p = df.loc[df["event"] == "buildup", "p_smooth"].dropna()
    kde_mode, hist_mode, _ = _kde_mode(bu_p.to_numpy()) if len(bu_p) > 20 else (np.nan, np.nan, np.nan)
    if not np.isnan(kde_mode):
        ax.axhline(kde_mode, color="#006400", ls="--", lw=2.0,
                   label=fr"$P_{{res}}$ (KDE) = {kde_mode:.1f} psi")
    if wt.p_reservoir is not None:
        ax.axhline(wt.p_reservoir, color="#ff8c00", ls=":", lw=1.2,
                   label=fr"$P_{{res}}$ (pipeline) = {wt.p_reservoir:.1f} psi")
    for ev in wt.events:
        mid_t = ev.t_start + (ev.t_end - ev.t_start) / 2
        mid_p = (ev.p_initial + ev.p_final) / 2
        ax.annotate(
            ev.event_id, xy=(mid_t, mid_p),
            fontsize=7, fontweight="bold", color="k",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="gray", alpha=0.80, lw=0.5),
        )
    ax.set_ylabel("Pressure (psi)")
    ax.set_title("Well Test — Event Detection & Reservoir Pressure",
                 fontweight="bold")
    ax.legend(loc="upper right", ncol=2, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d-%b"))

    # ── Panel (optional): Temperature ───────────────────────────────────
    if has_temp:
        ax = axes[idx]; idx += 1
        ax.plot(df["timestamp"], df["T_smooth"], color="#e377c2", lw=0.9)
        ax.set_ylabel("Temperature")
        ax.set_title("Bottomhole Temperature", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # ── Panel: log-log diagnostic of longest BU ─────────────────────────
    ax = axes[idx]; idx += 1
    longest = wt.events.longest_buildup
    if longest is not None:
        seg_p = longest.data["p_smooth"].to_numpy()
        seg_hr = longest.data["elapsed_hr"].to_numpy()
        dt_bu = seg_hr - seg_hr[0]
        dt_bu[0] = 1e-6
        dp_bu = seg_p - seg_p[0]

        ax.loglog(dt_bu, np.abs(dp_bu), "o-", ms=2, lw=0.8,
                  color="#1f77b4", label=fr"$|\Delta P|$ — {longest.event_id}")
        dt_d, deriv_d = bourdet_derivative(dt_bu, dp_bu, L=0.2)
        if len(dt_d) > 3:
            ax.loglog(dt_d, deriv_d, "s-", ms=2, lw=0.8,
                      color="#d62728", label="Bourdet derivative")
        ax.set_xlabel(r"$\Delta t$ (hr)")
        ax.set_ylabel(r"$|\Delta P|$, $dP/d(\ln t)$ (psi)")
        tp_label = (f"  (tp = {longest.preceding_dd_dur_hr:.2f} hr)"
                    if longest.preceding_dd_dur_hr else "")
        ax.set_title(f"Log–Log Diagnostic — {longest.event_id}{tp_label}", fontsize=10)
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "No buildup detected", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Log–Log Diagnostic", fontsize=10)

    # ── Panel: BU pressure histogram + KDE ──────────────────────────────
    ax = axes[idx]
    if len(bu_p) > 20:
        n_bins = max(50, int(np.sqrt(len(bu_p))))
        ax.hist(bu_p, bins=n_bins, color="#2ca02c", alpha=0.55,
                edgecolor="#006400", lw=0.4, density=True, label="Histogram")
        try:
            kde = gaussian_kde(bu_p, bw_method="silverman")
            x_kde = np.linspace(bu_p.min(), bu_p.max(), 500)
            ax.plot(x_kde, kde(x_kde), color="#006400", lw=1.8, label="KDE")
        except Exception:
            pass
        if not np.isnan(kde_mode):
            ax.axvline(kde_mode, color="#006400", ls="--", lw=2.0,
                       label=f"Mode = {kde_mode:.1f}")
        ax.axvline(bu_p.mean(), color="#1f77b4", ls="-.", lw=1.2,
                   label=f"Mean = {bu_p.mean():.1f}")
        ax.axvline(bu_p.median(), color="#d62728", ls=":", lw=1.2,
                   label=f"Median = {bu_p.median():.1f}")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Pressure (psi)")
    ax.set_ylabel("Density")
    ax.set_title("Buildup Pressure Distribution", fontsize=10)

    fig.align_ylabels(axes)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig
