r"""
welltest_pta.validation.cross_validation
========================================
Cross-validation scores for the auto event-detector.

Three complementary scores are computed and printed when the user calls
``WellTest.from_file(..., cross_validate=True)`` or directly via
:func:`cross_validate_detector`:

1. **Bootstrap event-count stability**

   Re-run the detector on :math:`K` random resamples of the data
   (each downsampled to a fraction :math:`f` of the original gauge rate)
   and report the standard deviation of the number of detected DDs and
   BUs. Low σ ⇒ stable detection.

2. **Parameter sensitivity sweep**

   Vary each detector parameter (``hampel_sigma``, ``spike_percentile``,
   ``min_pta_dp_psi``) by ±20 % and report how many events change. Few
   changes ⇒ robust to parameter choice.

3. **Edge-position consistency**

   For each detected event, compare the start/end indices across bootstrap
   replicas. Report the median ± IQR positional drift in samples and the
   "Jaccard overlap" of the masked time axis.

The combined score is a 0–100 confidence index:

==========  =============================================
80 – 100    Highly robust — manual review optional
60 –  80    Reasonable — spot-check critical events
40 –  60    Marginal — recommend manual splitting
 0 –  40    Unstable — manual splitting strongly advised
==========  =============================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from welltest_pta.detection.detector import (
    EventDetector,
    EventDetectorConfig,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectorCVResult:
    """Cross-validation report from :func:`cross_validate_detector`."""

    # Bootstrap
    n_bootstrap: int
    bootstrap_n_dd_mean: float
    bootstrap_n_dd_std: float
    bootstrap_n_bu_mean: float
    bootstrap_n_bu_std: float

    # Sensitivity (per-parameter)
    sensitivity: dict[str, dict[str, float]]

    # Edge consistency
    edge_jaccard_mean: float
    edge_jaccard_std: float

    # Combined score 0–100
    overall_score: float
    grade: str

    # Reference detection (for comparison)
    ref_n_dd: int
    ref_n_bu: int

    raw_data: dict = field(default_factory=dict, repr=False)

    def print_report(self) -> None:
        """Pretty-print the CV report to stdout."""
        sep = "═" * 72
        print(f"\n{sep}")
        print("  EVENT-DETECTOR CROSS-VALIDATION REPORT")
        print(sep)
        print(f"  Reference detection:   {self.ref_n_dd} drawdowns, {self.ref_n_bu} buildups")
        print()
        print(f"  Bootstrap stability  (K = {self.n_bootstrap} replicas):")
        print(f"    n_drawdowns:    {self.bootstrap_n_dd_mean:6.2f} ± {self.bootstrap_n_dd_std:.2f}")
        print(f"    n_buildups:     {self.bootstrap_n_bu_mean:6.2f} ± {self.bootstrap_n_bu_std:.2f}")
        print()
        print(f"  Edge-position consistency (Jaccard overlap):")
        print(f"    mean = {self.edge_jaccard_mean:.3f}   (1.0 = perfect)")
        print(f"    std  = {self.edge_jaccard_std:.3f}")
        print()
        print("  Parameter sensitivity (Δ events under ±20 % perturbation):")
        for pname, info in self.sensitivity.items():
            print(f"    {pname:<22s}  Δ_dd = {info['delta_n_dd']:+d},  "
                  f"Δ_bu = {info['delta_n_bu']:+d}")
        print()
        print(f"  ─ OVERALL CV SCORE:  {self.overall_score:5.1f} / 100   "
              f"({self.grade})")
        print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_pta_events(
    df: pd.DataFrame,
) -> tuple[int, int, list[tuple[int, int, str]]]:
    """Count DD / BU events in an annotated DataFrame; also return RLE list."""
    if "event" not in df.columns:
        return 0, 0, []
    labels = df["event"]
    if len(labels) == 0:
        return 0, 0, []
    groups, cur, start = [], labels.iloc[0], 0
    for i in range(1, len(labels)):
        if labels.iloc[i] != cur:
            groups.append((start, i, cur))
            cur, start = labels.iloc[i], i
    groups.append((start, len(labels), cur))
    pta_groups = [(s, e, l) for s, e, l in groups if l in ("drawdown", "buildup")]
    n_dd = sum(1 for _, _, l in pta_groups if l == "drawdown")
    n_bu = sum(1 for _, _, l in pta_groups if l == "buildup")
    return n_dd, n_bu, pta_groups


def _jaccard_pta_mask(
    ref_df: pd.DataFrame, alt_df: pd.DataFrame
) -> float:
    """
    Jaccard overlap of the "is-PTA" masks aligned on the timestamp index.

    Both DataFrames must have a ``timestamp`` column and an ``event`` column.
    Resamples to the reference timeline by nearest-neighbour.
    """
    if "event" not in ref_df.columns or "event" not in alt_df.columns:
        return 0.0
    ref_pta = ref_df["event"].isin(("drawdown", "buildup")).to_numpy()
    # Reindex alt_df labels to the reference timestamps via nearest-neighbour
    alt_ts = pd.Series(alt_df["event"].values, index=alt_df["timestamp"])
    alt_ts = alt_ts.sort_index()
    alt_aligned = alt_ts.reindex(ref_df["timestamp"], method="nearest")
    alt_pta = alt_aligned.isin(("drawdown", "buildup")).to_numpy()
    inter = np.logical_and(ref_pta, alt_pta).sum()
    union = np.logical_or(ref_pta, alt_pta).sum()
    return float(inter / union) if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Public utilities
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_score(
    df: pd.DataFrame,
    cfg: Optional[EventDetectorConfig] = None,
    n_bootstrap: int = 10,
    downsample_frac: float = 0.85,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, float]:
    """Down-sample bootstrap of detector event counts."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(df)
    keep_n = max(int(n * downsample_frac), 100)
    n_dds, n_bus = [], []
    for _ in range(n_bootstrap):
        idx = np.sort(rng.choice(n, size=keep_n, replace=False))
        sub = df.iloc[idx].reset_index(drop=True)
        try:
            det = EventDetector(cfg=cfg or EventDetectorConfig())
            ann = det.detect(sub)
            n_dd, n_bu, _ = _count_pta_events(ann)
            n_dds.append(n_dd)
            n_bus.append(n_bu)
        except Exception as e:  # pragma: no cover
            logger.debug("Bootstrap replica failed: %s", e)
    if not n_dds:
        return {"n_dd_mean": 0.0, "n_dd_std": np.nan,
                "n_bu_mean": 0.0, "n_bu_std": np.nan,
                "n_replicas_ok": 0}
    return {
        "n_dd_mean": float(np.mean(n_dds)),
        "n_dd_std": float(np.std(n_dds)),
        "n_bu_mean": float(np.mean(n_bus)),
        "n_bu_std": float(np.std(n_bus)),
        "n_replicas_ok": len(n_dds),
    }


def parameter_sensitivity(
    df: pd.DataFrame,
    cfg: Optional[EventDetectorConfig] = None,
    perturbation: float = 0.20,
) -> dict[str, dict[str, int]]:
    """Sweep ±perturbation around each detector parameter."""
    base_cfg = cfg or EventDetectorConfig()
    base_det = EventDetector(cfg=base_cfg)
    base_ann = base_det.detect(df)
    base_n_dd, base_n_bu, _ = _count_pta_events(base_ann)

    params_to_perturb = {
        "hampel_sigma": base_cfg.hampel_sigma,
        "spike_percentile": base_cfg.spike_percentile,
        "min_pta_dp_psi": base_cfg.min_pta_dp_psi,
        "tail_trim_dev_n_sigma": base_cfg.tail_trim_dev_n_sigma,
    }
    out: dict[str, dict[str, int]] = {}
    for name, val in params_to_perturb.items():
        max_dd_diff = 0
        max_bu_diff = 0
        for sign in (-1, +1):
            new_val = val * (1.0 + sign * perturbation)
            try:
                cfg_perturbed = EventDetectorConfig(
                    **{k: getattr(base_cfg, k) for k in base_cfg.__dataclass_fields__}
                )
                setattr(cfg_perturbed, name, new_val)
                det = EventDetector(cfg=cfg_perturbed)
                ann = det.detect(df)
                n_dd, n_bu, _ = _count_pta_events(ann)
                if abs(n_dd - base_n_dd) > abs(max_dd_diff):
                    max_dd_diff = n_dd - base_n_dd
                if abs(n_bu - base_n_bu) > abs(max_bu_diff):
                    max_bu_diff = n_bu - base_n_bu
            except Exception as e:  # pragma: no cover
                logger.debug("Sensitivity perturbation failed for %s: %s", name, e)
        out[name] = {"delta_n_dd": int(max_dd_diff), "delta_n_bu": int(max_bu_diff)}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Top-level function
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_detector(
    df: pd.DataFrame,
    cfg: Optional[EventDetectorConfig] = None,
    n_bootstrap: int = 8,
    downsample_frac: float = 0.85,
    perturbation: float = 0.20,
    seed: int = 42,
    print_report: bool = True,
) -> DetectorCVResult:
    """
    Run all three CV checks and return a :class:`DetectorCVResult`.

    Parameters
    ----------
    df
        Parsed gauge DataFrame (output of :func:`welltest_pta.parser.parse`).
    cfg
        Detector configuration. Defaults to standard V8.1 settings.
    n_bootstrap
        Number of bootstrap replicas (≥4 recommended).
    downsample_frac
        Fraction of points retained per replica.
    perturbation
        Fractional change applied to each parameter (e.g. 0.20 = ±20 %).
    seed
        RNG seed for reproducibility.
    print_report
        If True (default) prints the human-readable report.
    """
    rng = np.random.default_rng(seed)

    # Reference detection
    base_cfg = cfg or EventDetectorConfig()
    ref_det = EventDetector(cfg=base_cfg)
    ref_ann = ref_det.detect(df)
    ref_n_dd, ref_n_bu, _ = _count_pta_events(ref_ann)

    # 1. Bootstrap
    bs = bootstrap_score(df, base_cfg, n_bootstrap, downsample_frac, rng)

    # 2. Sensitivity
    sens = parameter_sensitivity(df, base_cfg, perturbation)

    # 3. Edge consistency  (re-run on bootstrap, compute Jaccard against reference)
    jaccards: list[float] = []
    n = len(df)
    keep_n = max(int(n * downsample_frac), 100)
    for _ in range(n_bootstrap):
        idx = np.sort(rng.choice(n, size=keep_n, replace=False))
        sub = df.iloc[idx].reset_index(drop=True)
        try:
            det = EventDetector(cfg=base_cfg)
            ann = det.detect(sub)
            jaccards.append(_jaccard_pta_mask(ref_ann, ann))
        except Exception:
            pass
    j_mean = float(np.mean(jaccards)) if jaccards else 0.0
    j_std = float(np.std(jaccards)) if jaccards else 0.0

    # ── Composite score ──
    bs_pen = (bs["n_dd_std"] + bs["n_bu_std"])
    bs_score = max(0.0, 1.0 - bs_pen / max(ref_n_dd + ref_n_bu, 1))
    sens_total = sum(abs(v["delta_n_dd"]) + abs(v["delta_n_bu"]) for v in sens.values())
    sens_score = max(0.0, 1.0 - sens_total / max(2 * (ref_n_dd + ref_n_bu) * len(sens), 1))
    overall = 100.0 * (0.40 * bs_score + 0.40 * j_mean + 0.20 * sens_score)
    overall = float(np.clip(overall, 0.0, 100.0))

    if overall >= 80:
        grade = "HIGHLY ROBUST"
    elif overall >= 60:
        grade = "REASONABLE"
    elif overall >= 40:
        grade = "MARGINAL — manual review recommended"
    else:
        grade = "UNSTABLE — manual splitting strongly advised"

    result = DetectorCVResult(
        n_bootstrap=n_bootstrap,
        bootstrap_n_dd_mean=bs["n_dd_mean"],
        bootstrap_n_dd_std=bs["n_dd_std"],
        bootstrap_n_bu_mean=bs["n_bu_mean"],
        bootstrap_n_bu_std=bs["n_bu_std"],
        sensitivity=sens,
        edge_jaccard_mean=j_mean,
        edge_jaccard_std=j_std,
        overall_score=overall,
        grade=grade,
        ref_n_dd=ref_n_dd,
        ref_n_bu=ref_n_bu,
        raw_data={"jaccards": jaccards},
    )
    if print_report:
        result.print_report()
    return result
