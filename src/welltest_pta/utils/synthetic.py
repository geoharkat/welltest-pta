r"""
welltest_pta.utils.synthetic
============================
Synthetic DST gauge data — useful for testing and tutorials.

The generator builds a multi-rate test consisting of an initial flow,
flow-and-shut-in cycles (FFSI), and a final extended buildup. Pressure
honours an exponential approach to a target value within each event,
plus optional Gaussian noise.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def generate_synthetic_dst(
    p_reservoir: float = 4500.0,
    n_samples: int = 18_000,
    sample_period_s: float = 5.0,
    start_time: Optional[str | datetime] = None,
    sequence: Optional[Sequence[tuple[str, float, float]]] = None,
    noise_psi: float = 1.0,
    seed: int = 42,
    include_temperature: bool = True,
) -> pd.DataFrame:
    r"""
    Build a synthetic DST gauge DataFrame.

    Parameters
    ----------
    p_reservoir
        Static reservoir pressure (psi). Buildups asymptote to this.
    n_samples
        Total number of samples to generate. The duration is
        ``n_samples × sample_period_s``.
    sample_period_s
        Sampling period in seconds (default 5 s — typical electronic gauge).
    start_time
        Test start; defaults to *now*.
    sequence
        List of ``(type, duration_hr, target_p)`` tuples that define the
        test programme. ``type`` is ``"DD"`` (drawdown) or ``"BU"``
        (buildup). The default mimics a typical multi-rate DST::

            [
                ("DD", 0.5,  3300),  # initial flow
                ("BU", 1.0,  4490),  # short BU
                ("DD", 1.0,  3000),  # main DD
                ("BU", 6.0,  4495),  # final BU
            ]

        plus padding non-PTA segments at start and end.
    noise_psi
        Standard deviation of additive Gaussian noise.
    seed
        RNG seed.
    include_temperature
        If True, also generate a smoothly varying ``temperature`` column.

    Returns
    -------
    DataFrame with columns ``timestamp``, ``pressure``, ``temperature``
    (optional), plus a ``true_event`` ground-truth label column.
    """
    rng = np.random.default_rng(seed)
    if sequence is None:
        sequence = [
            ("DD", 0.5, 3300.0),
            ("BU", 1.0, 4490.0),
            ("DD", 1.0, 3000.0),
            ("BU", 6.0, 4495.0),
        ]
    if start_time is None:
        start = datetime(2025, 1, 15, 8, 0, 0)
    else:
        start = pd.Timestamp(start_time).to_pydatetime()

    total_dur_hr = n_samples * sample_period_s / 3600.0

    # Build an event timeline
    events_dur = sum(d for _, d, _ in sequence)
    pad_total = max(total_dur_hr - events_dur, 0.10)
    pad_start = pad_total * 0.20
    pad_end = pad_total * 0.30
    intersegment = (pad_total - pad_start - pad_end) / max(len(sequence), 1)

    timeline: list[tuple[float, float, str, float]] = []
    cur_t = 0.0
    # Initial RIH segment
    timeline.append((cur_t, cur_t + pad_start, "non_pta", p_reservoir))
    cur_t += pad_start

    for i, (typ, dur, target) in enumerate(sequence):
        label = "drawdown" if typ.upper() == "DD" else "buildup"
        timeline.append((cur_t, cur_t + dur, label, target))
        cur_t += dur
        # add inter-segment short non_pta gap (except after last)
        if i < len(sequence) - 1:
            timeline.append((cur_t, cur_t + intersegment, "non_pta", target))
            cur_t += intersegment

    # POOH segment at the end
    timeline.append((cur_t, total_dur_hr, "non_pta", p_reservoir * 0.4))

    # Build the data
    times = np.arange(n_samples) * sample_period_s / 3600.0  # hr from start
    timestamps = pd.to_datetime(start) + pd.to_timedelta(times, unit="h")
    pressure = np.full(n_samples, p_reservoir, dtype=float)
    true_event = np.full(n_samples, "non_pta", dtype=object)

    last_p = p_reservoir
    for (t0, t1, lbl, target) in timeline:
        mask = (times >= t0) & (times < t1)
        if not mask.any():
            continue
        seg_t = times[mask] - t0
        dur = max(t1 - t0, 1e-6)
        # Exponential approach: p(t) = target + (p_start - target) * exp(-k * t / dur)
        if lbl == "buildup":
            k = 4.0  # faster approach in buildups
        elif lbl == "drawdown":
            k = 3.0
        else:
            k = 5.0  # fast for non-PTA transients (RIH/POOH spikes)
        decay = np.exp(-k * seg_t / dur)
        seg_p = target + (last_p - target) * decay
        pressure[mask] = seg_p
        true_event[mask] = lbl
        last_p = float(seg_p[-1])

    # Add noise
    pressure += rng.normal(0, noise_psi, size=pressure.shape)

    out = pd.DataFrame({
        "timestamp": timestamps,
        "pressure": pressure,
        "true_event": true_event,
    })

    if include_temperature:
        # Slowly varying temperature with cycle-following dips
        base_T = 180.0  # °F
        T = base_T + 5.0 * np.sin(2 * np.pi * times / 24.0)
        # Slight drop during drawdowns
        for (t0, t1, lbl, _) in timeline:
            if lbl != "drawdown":
                continue
            mask = (times >= t0) & (times < t1)
            T[mask] -= 1.5
        T += rng.normal(0, 0.05, size=T.shape)
        out["temperature"] = T

    return out
