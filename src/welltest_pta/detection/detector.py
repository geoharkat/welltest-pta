"""
welltest_pta.detection.detector
================================
Automatic PTA event detection (drawdown / buildup / non-PTA).

Algorithm — V8.1 "Spike-Boundary + Tail-Trim"
---------------------------------------------
Phase 0  Hampel-filter despike  →  Savitzky–Golay smoothing  →  noise-floor σ̂
Phase 1  Reservoir-pressure plateau detection
Phase 2  RIH / POOH edge masking
Phase 3  Spike-boundary + turning-point detection (with validation)
Phase 4  Zone classification using net-ΔP signed logic
Phase 5  Pause absorption  →  same-type merge  →  edge trimming
         →  V8.1 post-plateau tail trim (H→I→J spike before POOH)

Adapted from the original ``WellTestEventDetector V8.1`` (Harkat 2025)
into a stateless ``detect_events()`` function plus the stateful
``EventDetector`` class for advanced use cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventDetectorConfig:
    """Configuration for the V8.1 event detector.

    All fields are public; pass them as keyword arguments at construction
    time (e.g. ``EventDetectorConfig(sg_window=31, hampel_sigma=4.0)``).
    """

    # ── Smoothing (Savitzky–Golay) ─────────────────────────────────────
    sg_window: int = 0           # 0 = auto
    sg_polyorder: int = 3
    sg_min_window: int = 15
    sg_max_window: int = 301

    # ── Robustness (Hampel filter) ─────────────────────────────────────
    hampel_window: int = 0       # 0 = auto
    hampel_sigma: float = 3.0

    # ── Reservoir pressure detection ───────────────────────────────────
    p_res_override: float = 0.0
    p_res_stable_pct: float = 20.0
    p_res_min_pts: int = 30

    # ── Spike-boundary detection ───────────────────────────────────────
    spike_percentile: float = 95.0
    spike_min_gap_pts: int = 0   # 0 = auto

    # ── Zone classification ────────────────────────────────────────────
    min_zone_pts: int = 10
    min_pta_dp_psi: float = 15.0
    min_pta_duration_hr: float = 0.10

    # ── Same-type merge ────────────────────────────────────────────────
    merge_gap_max_pts: int = 0   # 0 = auto

    # ── V8.1 tail trim (long-buildup post-plateau spike removal) ───────
    tail_trim_enabled: bool = True
    tail_trim_min_dur_hr: float = 4.0
    tail_trim_min_plateau_frac: float = 0.40
    tail_trim_dev_n_sigma: float = 8.0
    tail_trim_min_tail_dur_hr: float = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Stateless wrapper
# ─────────────────────────────────────────────────────────────────────────────

def detect_events(
    df: pd.DataFrame,
    cfg: Optional[EventDetectorConfig] = None,
) -> tuple[pd.DataFrame, "EventDetector"]:
    """
    Detect drawdown / buildup events on a parsed gauge DataFrame.

    Parameters
    ----------
    df
        Output of :func:`welltest_pta.parser.parse`. Must contain
        ``timestamp`` and ``pressure``.
    cfg
        Detector configuration. Default values cover most DST cases.

    Returns
    -------
    annotated_df
        Copy of ``df`` with extra columns ``p_smooth``, ``dpdt``,
        ``elapsed_hr``, ``event``, ``p_reservoir``.
    detector
        The fitted ``EventDetector`` (exposes ``_p_res``, ``_noise_floor``).
    """
    detector = EventDetector(cfg=cfg)
    annotated = detector.detect(df)
    return annotated, detector


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class EventDetector:
    """V8.1 robust event detector — see module docstring for the pipeline."""

    def __init__(self, cfg: Optional[EventDetectorConfig] = None) -> None:
        self.cfg = cfg or EventDetectorConfig()
        self._sg_window = 21
        self._p_res = 0.0
        self._spike_thr = 100.0
        self._noise_floor = 1.0  # estimated noise σ̂ (psi)

    # ── small helpers ────────────────────────────────────────────────────

    @staticmethod
    def _odd(w: int, mn: int = 5) -> int:
        w = max(w, mn)
        return w + 1 if w % 2 == 0 else w

    @staticmethod
    def _hampel_filter(
        series: np.ndarray, window: int, n_sigmas: float = 3.0
    ) -> np.ndarray:
        """Vectorised Hampel filter using pandas rolling median + MAD."""
        n = len(series)
        if n == 0:
            return series
        window = max(3, window)
        if window % 2 == 0:
            window += 1
        s = pd.Series(series)
        med = s.rolling(window, center=True, min_periods=1).median()
        mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
        std_mad = 1.4826 * mad
        outliers = (s - med).abs() > n_sigmas * std_mad
        out = s.where(~outliers, med).to_numpy()
        return out

    @staticmethod
    def _contiguous(mask: np.ndarray) -> list[tuple[int, int]]:
        regions, in_r, start = [], False, 0
        for i, m in enumerate(mask):
            if m and not in_r:
                start, in_r = i, True
            elif not m and in_r:
                regions.append((start, i))
                in_r = False
        if in_r:
            regions.append((start, len(mask)))
        return regions

    @staticmethod
    def _rle(labels: pd.Series) -> list[tuple[int, int, str]]:
        groups, cur, start = [], labels.iloc[0], 0
        for i in range(1, len(labels)):
            if labels.iloc[i] != cur:
                groups.append((start, i, cur))
                cur, start = labels.iloc[i], i
        groups.append((start, len(labels), cur))
        return groups

    # ── PHASE 0: smoothing & despike ────────────────────────────────────

    def _compute_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        out = df.copy()
        t0 = df["timestamp"].iloc[0]
        hours = (df["timestamp"] - t0).dt.total_seconds().values / 3600.0
        out["elapsed_hr"] = hours

        n = len(df)
        diffs = np.diff(hours)
        mdt = max(np.median(diffs) if len(diffs) > 0 else 1e-3, 1e-6)

        p_raw = df["pressure"].interpolate(limit_direction="both").values

        # Hampel despike
        if cfg.hampel_window > 0:
            hw = cfg.hampel_window
        else:
            hw = max(int(180 / (mdt * 3600)), int(n * 0.005), 5)
            hw = min(hw, 51)
        logger.info("  Despike window: %d pts (Hampel filter)", hw)
        p_clean = self._hampel_filter(p_raw, hw, cfg.hampel_sigma)

        # SG smoothing
        if cfg.sg_window > 0:
            self._sg_window = cfg.sg_window
        else:
            w = max(int(n * 0.005), max(int(3.0 / 60 / mdt), 15), cfg.sg_min_window)
            w = min(w, cfg.sg_max_window, n - 1)
            self._sg_window = self._odd(w)

        wf = self._odd(min(self._sg_window, n - 1), cfg.sg_polyorder + 2)
        out["p_smooth"] = savgol_filter(p_clean, wf, cfg.sg_polyorder)
        out["dpdt"] = savgol_filter(p_clean, wf, cfg.sg_polyorder, deriv=1) / mdt

        # Optional temperature smoothing
        if "temperature" in df.columns and df["temperature"].notna().sum() > wf:
            tv = df["temperature"].interpolate(limit_direction="both").values
            out["T_smooth"] = savgol_filter(tv, wf, cfg.sg_polyorder)
            out["dTdt"] = savgol_filter(tv, wf, cfg.sg_polyorder, deriv=1) / mdt
        else:
            out["dTdt"] = np.nan

        # Noise floor estimate (75th percentile of |residuals|)
        residuals = p_raw - out["p_smooth"]
        noise_est = np.percentile(np.abs(residuals), 75)
        self._noise_floor = max(noise_est, 0.5)
        logger.info("  SG window=%d, Noise Floor ≈ %.2f psi", wf, self._noise_floor)

        return out

    # ── PHASE 1: P_res ───────────────────────────────────────────────────

    def _detect_pres(self, df: pd.DataFrame) -> float:
        cfg = self.cfg
        if cfg.p_res_override > 0:
            return cfg.p_res_override

        p = df["p_smooth"].values
        dpdt = np.abs(df["dpdt"].values)
        dpf = dpdt[np.isfinite(dpdt)]
        if len(dpf) < 10:
            return float(np.nanmax(p))

        thr = max(np.percentile(dpf, cfg.p_res_stable_pct), 0.5)
        stable = dpdt < thr
        if stable.sum() < cfg.p_res_min_pts:
            return float(np.percentile(p, 95))

        regions = self._contiguous(stable)
        valid = [(s, e) for s, e in regions if (e - s) >= cfg.p_res_min_pts]
        if not valid:
            return float(np.percentile(p[stable], 95))

        best_m, best = -np.inf, None
        for s, e in valid:
            m = np.mean(p[s:e])
            if m > best_m:
                best_m, best = m, (s, e)
        logger.info("  P_res = %.1f psi (plateau [%d-%d])", best_m, best[0], best[1])
        return float(best_m)

    # ── PHASE 2: edge masking ────────────────────────────────────────────

    def _mask_edges(
        self, df: pd.DataFrame, P_res: float
    ) -> tuple[np.ndarray, int, int]:
        p = df["p_smooth"].values
        n = len(p)
        cfg = self.cfg
        approach = P_res * (0.70 if P_res < 100 else 0.85)
        confirm = max(cfg.p_res_min_pts // 3, 5)

        # PTA start
        pta_start = 0
        for i in range(n):
            if p[i] >= approach:
                if np.mean(p[i:min(i + confirm, n)]) >= approach * 0.95:
                    pta_start = max(0, i - confirm)
                    break

        # PTA end
        pta_end = n - 1
        last_at = -1
        for i in range(n - 1, -1, -1):
            if p[i] >= approach:
                last_at = i
                break
        if last_at > pta_start:
            ext = min(confirm * 3, n - 1 - last_at)
            pta_end = last_at + ext
            crash = P_res * 0.50
            for j in range(last_at, min(pta_end + 1, n)):
                if p[j] < crash:
                    pta_end = max(last_at, j - 1)
                    break
            pta_end = min(pta_end, n - 1)

        mask = np.zeros(n, dtype=bool)
        mask[pta_start:pta_end + 1] = True
        logger.info("  PTA window: [%d-%d], trimmed=%d pts", pta_start, pta_end, n - mask.sum())
        return mask, pta_start, pta_end

    # ── PHASE 3: boundaries ──────────────────────────────────────────────

    def _detect_spike_boundaries(
        self, df: pd.DataFrame, pta_start: int, pta_end: int
    ) -> list[int]:
        cfg = self.cfg
        dpdt = df["dpdt"].values
        p = df["p_smooth"].values
        n = len(df)

        # 1. dp/dt spikes
        pta_dpdt = np.abs(dpdt[pta_start:pta_end + 1])
        pta_finite = pta_dpdt[np.isfinite(pta_dpdt)]
        spike_bounds: list[int] = []
        if len(pta_finite) > 10:
            self._spike_thr = max(
                np.percentile(pta_finite, cfg.spike_percentile),
                5.0,
                self._noise_floor * 5.0,
            )
            abs_dpdt = np.abs(dpdt)
            is_spike = np.zeros(n, dtype=bool)
            is_spike[pta_start:pta_end + 1] = (
                abs_dpdt[pta_start:pta_end + 1] > self._spike_thr
            )
            min_gap = cfg.spike_min_gap_pts
            if min_gap == 0:
                min_gap = max(int(n * 0.005), 10)

            in_spike = False
            last_bnd = pta_start
            for i in range(pta_start, pta_end + 1):
                if is_spike[i] and not in_spike:
                    if i - last_bnd >= min_gap:
                        spike_bounds.append(i)
                        last_bnd = i
                    in_spike = True
                elif not is_spike[i] and in_spike:
                    in_spike = False
        logger.info("  Initial spike boundaries: %d", len(spike_bounds))

        # 2. Pressure turning points
        tp_bounds: list[int] = []
        pta_p = p[pta_start:pta_end + 1]
        pta_n = len(pta_p)
        if pta_n > 50:
            tp_win = self._odd(max(int(pta_n * 0.05), 15))
            tp_win = min(tp_win, pta_n - 1)
            try:
                p_heavy = savgol_filter(pta_p, tp_win, 2)
            except Exception:
                p_heavy = pta_p
            dp_heavy = np.diff(p_heavy)
            sign_changes = np.where(np.diff(np.sign(dp_heavy)))[0]
            p_range = np.ptp(pta_p)
            min_prominence = max(p_range * 0.03, self._noise_floor * 10.0, 10.0)
            for sc in sign_changes:
                tp_idx = sc + pta_start
                if tp_idx <= pta_start + 20 or tp_idx >= pta_end - 20:
                    continue
                local_p = p[tp_idx]
                window_size = max(50, int(pta_n * 0.05))
                left_p = p[max(0, tp_idx - window_size):tp_idx]
                right_p = p[tp_idx + 1:min(n, tp_idx + window_size)]
                if len(left_p) < 5 or len(right_p) < 5:
                    continue
                if (np.min(left_p) > local_p + min_prominence
                        and np.min(right_p) > local_p + min_prominence):
                    tp_bounds.append(tp_idx)
                elif (np.max(left_p) < local_p - min_prominence
                        and np.max(right_p) < local_p - min_prominence):
                    tp_bounds.append(tp_idx)

        # 3. Validate (sustained change across ±check_window)
        all_bounds = sorted(set(spike_bounds + tp_bounds))
        validated: list[int] = []
        check_window = max(int(0.01 * len(df)), 10)
        for b in all_bounds:
            pre_start = max(pta_start, b - check_window)
            post_end = min(pta_end, b + check_window)
            p_before = np.median(p[pre_start:b])
            p_after = np.median(p[b:post_end])
            if abs(p_before - p_after) > (self._noise_floor * 5.0):
                validated.append(b)
        logger.info("  Final boundaries: %d (validated)", len(validated))
        return validated

    # ── PHASE 4: zone classification ────────────────────────────────────

    def _classify_zones(
        self,
        df: pd.DataFrame,
        boundaries: list[int],
        pta_start: int,
        pta_end: int,
        valid_mask: np.ndarray,
    ) -> pd.Series:
        cfg = self.cfg
        p = df["p_smooth"].values
        hours = df["elapsed_hr"].values
        n = len(df)

        labels = pd.Series("non_pta", index=df.index, name="event")
        zone_edges = sorted(set([pta_start] + boundaries + [pta_end + 1]))
        zones = [(zs, ze) for zs, ze in zip(zone_edges[:-1], zone_edges[1:]) if ze - zs >= 2]

        logger.info("  %d zones to classify", len(zones))

        for zs, ze in zones:
            ze_safe = min(ze - 1, n - 1)
            p_start = np.median(p[zs:min(zs + 5, ze)])
            p_end = np.median(p[max(zs, ze - 5):ze])
            net_dp = p_end - p_start
            dur = hours[ze_safe] - hours[zs]
            abs_dp = abs(net_dp)

            if not valid_mask[zs]:
                label = "non_pta"
            elif (abs_dp < max(cfg.min_pta_dp_psi, self._noise_floor * 5.0)
                    or dur < cfg.min_pta_duration_hr):
                label = "pause"
            elif net_dp < 0:
                label = "drawdown"
            else:
                label = "buildup"
            labels.iloc[zs:ze] = label

        return labels

    # ── PHASE 5: cleanup ─────────────────────────────────────────────────

    def _absorb_pauses(self, labels: pd.Series, df: pd.DataFrame) -> pd.Series:
        refined = labels.copy()
        p = df["p_smooth"].values
        changed = True
        while changed:
            changed = False
            groups = self._rle(refined)
            for i, (s, e, l) in enumerate(groups):
                if l != "pause":
                    continue
                left = groups[i - 1][2] if i > 0 else "non_pta"
                right = groups[i + 1][2] if i < len(groups) - 1 else "non_pta"
                if left == right and left in ("drawdown", "buildup"):
                    refined.iloc[s:e] = left
                    changed = True
                    break
                elif left in ("drawdown", "buildup") and right in ("drawdown", "buildup"):
                    net_dp = p[min(e - 1, len(p) - 1)] - p[s]
                    refined.iloc[s:e] = "drawdown" if net_dp < 0 else "buildup"
                    changed = True
                    break
                elif left in ("drawdown", "buildup"):
                    refined.iloc[s:e] = left
                    changed = True
                    break
                elif right in ("drawdown", "buildup"):
                    refined.iloc[s:e] = right
                    changed = True
                    break
                else:
                    refined.iloc[s:e] = "non_pta"
                    changed = True
                    break
        return refined.replace("pause", "non_pta")

    def _merge_same_type(self, labels: pd.Series, df: pd.DataFrame) -> pd.Series:
        p = df["p_smooth"].values
        hours = df["elapsed_hr"].values
        changed = True
        while changed:
            changed = False
            groups = self._rle(labels)
            if len(groups) < 3:
                break
            for i in range(len(groups) - 1):
                s1, e1, l1 = groups[i]
                s2, e2, l2 = groups[i + 1]
                if l1 == l2 and l1 in ("drawdown", "buildup", "non_pta"):
                    labels.iloc[s1:e2] = l1
                    changed = True
                    break
            if changed:
                continue
            for i in range(len(groups) - 2):
                s1, e1, l1 = groups[i]
                s2, e2, l2 = groups[i + 1]
                s3, e3, l3 = groups[i + 2]
                if not (l1 in ("drawdown", "buildup") and l1 == l3 and l2 != l1):
                    continue
                es2 = min(e2 - 1, len(p) - 1)
                mid_dp = abs(p[es2] - p[s2])
                mid_dur = hours[es2] - hours[s2]
                total_dp = abs(p[min(e3 - 1, len(p) - 1)] - p[s1])
                if (mid_dp < max(total_dp * 0.2, self._noise_floor * 10.0)
                        and mid_dur < 0.5):
                    labels.iloc[s1:e3] = l1
                    changed = True
                    break
        return labels

    def _absorb_tiny_gaps(self, labels: pd.Series, df: pd.DataFrame) -> pd.Series:
        refined = labels.copy()
        groups = self._rle(labels)
        for i, (s, e, l) in enumerate(groups):
            if l != "non_pta" or i == 0 or i == len(groups) - 1:
                continue
            pts = e - s
            left, right = groups[i - 1][2], groups[i + 1][2]
            if left not in ("drawdown", "buildup") or right not in ("drawdown", "buildup"):
                continue
            thr = max(20, int(len(df) * 0.004))
            if pts > thr:
                continue
            if left == right:
                refined.iloc[s:e] = left
            elif pts < 10:
                refined.iloc[s:e] = left
        return refined

    def _trim_buildup_tails(self, labels: pd.Series, df: pd.DataFrame) -> pd.Series:
        """V8.1 — trim post-plateau tail spikes (H→I→J before POOH)."""
        cfg = self.cfg
        if not cfg.tail_trim_enabled:
            return labels

        refined = labels.copy()
        p = df["p_smooth"].values
        hours = df["elapsed_hr"].values
        n = len(p)
        plat_tol = max(self._noise_floor * 4.0, 3.0)
        tail_thr_psi = max(self._noise_floor * cfg.tail_trim_dev_n_sigma, 5.0)

        for s, e, l in self._rle(refined):
            if l != "buildup":
                continue
            zone_n = e - s
            es = min(e - 1, n - 1)
            zone_dur = hours[es] - hours[s]
            if zone_dur < cfg.tail_trim_min_dur_hr or zone_n < 200:
                continue

            zone_p = p[s:e]
            if np.ptp(zone_p) < plat_tol * 4.0:
                continue

            n_bins = max(60, int(np.sqrt(zone_n)))
            hist, edges = np.histogram(zone_p, bins=n_bins)
            mode_idx = int(np.argmax(hist))
            p_plateau = float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)

            on_plateau = np.abs(zone_p - p_plateau) < plat_tol
            late_half_cov = float(np.mean(on_plateau[zone_n // 2:]))
            if late_half_cov < 0.30:
                continue

            idx_on = np.where(on_plateau)[0]
            if idx_on.size == 0:
                continue
            last_pl_loc = int(idx_on.max())
            last_pl_global = s + last_pl_loc + 1
            plateau_frac = (last_pl_global - s) / zone_n
            if plateau_frac < cfg.tail_trim_min_plateau_frac:
                continue

            tail_n = e - last_pl_global
            tail_dur = (hours[es] - hours[last_pl_global]
                        if last_pl_global < e else 0.0)
            if tail_n < 30 or tail_dur < cfg.tail_trim_min_tail_dur_hr:
                continue

            tail_p = zone_p[last_pl_loc + 1:]
            tail_dev = float(np.max(np.abs(tail_p - p_plateau)))
            if tail_dev <= tail_thr_psi:
                continue

            logger.info(
                "  ✂  Tail trim BU [%d-%d]:  plateau=%.1f psi  tail_dev=%.1f psi  "
                "tail_dur=%.2f hr  late_cov=%d%%  plateau_frac=%d%%  "
                "→ relabel [%d-%d] as non_pta",
                s, e, p_plateau, tail_dev, tail_dur,
                int(late_half_cov * 100), int(plateau_frac * 100),
                last_pl_global, e,
            )
            refined.iloc[last_pl_global:e] = "non_pta"

        return refined

    def _validate(self, labels: pd.Series, df: pd.DataFrame) -> pd.Series:
        cfg = self.cfg
        refined = labels.copy()
        p = df["p_smooth"].values
        hours = df["elapsed_hr"].values
        for s, e, l in self._rle(refined):
            if l not in ("drawdown", "buildup"):
                continue
            es = min(e - 1, len(p) - 1)
            dur = hours[es] - hours[s]
            dp = p[es] - p[s]
            if (dur < cfg.min_pta_duration_hr
                    or abs(dp) < max(cfg.min_pta_dp_psi, self._noise_floor * 5.0)):
                refined.iloc[s:e] = "non_pta"
                continue
            if l == "drawdown" and dp > cfg.min_pta_dp_psi:
                refined.iloc[s:e] = "buildup"
            elif l == "buildup" and dp < -cfg.min_pta_dp_psi:
                refined.iloc[s:e] = "drawdown"
        return refined

    def _trim_edges(self, labels: pd.Series, df: pd.DataFrame) -> pd.Series:
        refined = labels.copy()
        p = df["p_smooth"].values
        changed = True
        while changed:
            changed = False
            groups = self._rle(refined)
            pta = [(i, s, e, l) for i, (s, e, l) in enumerate(groups)
                   if l in ("drawdown", "buildup")]
            if not pta:
                break
            li, ls, le, ll = pta[-1]
            if ll == "drawdown":
                after = all(groups[j][2] == "non_pta" for j in range(li + 1, len(groups)))
                if after or li == len(groups) - 1:
                    refined.iloc[ls:le] = "non_pta"
                    changed = True
                    continue
            fi, fs, fe, fl = pta[0]
            if fl == "buildup" and len(pta) >= 2 and pta[1][3] == "buildup":
                if p[fs] > self._p_res * 0.80:
                    refined.iloc[fs:fe] = "non_pta"
                    changed = True
        return refined

    # ── PUBLIC API ───────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full V8.1 detection pipeline and return annotated DataFrame."""
        required = {"timestamp", "pressure"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df_work = df.dropna(subset=["timestamp", "pressure"]).copy()
        df_work = df_work.sort_values("timestamp").reset_index(drop=True)
        n = len(df_work)
        if n < 20:
            raise ValueError(f"Too few data points ({n})")

        logger.info("─" * 60)
        logger.info("PTA Event Detection V8.1 on %d samples", n)
        logger.info("─" * 60)

        df_work = self._compute_derivatives(df_work)
        self._p_res = self._detect_pres(df_work)
        valid_mask, pta_start, pta_end = self._mask_edges(df_work, self._p_res)
        boundaries = self._detect_spike_boundaries(df_work, pta_start, pta_end)
        labels = self._classify_zones(df_work, boundaries, pta_start, pta_end, valid_mask)

        labels = self._absorb_pauses(labels, df_work)
        labels = self._merge_same_type(labels, df_work)
        labels = self._absorb_tiny_gaps(labels, df_work)
        labels = self._merge_same_type(labels, df_work)
        labels = self._validate(labels, df_work)
        labels = self._trim_edges(labels, df_work)
        labels = self._trim_buildup_tails(labels, df_work)
        labels = self._merge_same_type(labels, df_work)
        labels = self._absorb_tiny_gaps(labels, df_work)
        labels = self._merge_same_type(labels, df_work)

        df_work["event"] = labels
        df_work["p_reservoir"] = self._p_res

        # Summary
        groups = self._rle(labels)
        pta_g = [(s, e, l) for s, e, l in groups if l in ("drawdown", "buildup")]
        logger.info("Detected P_res = %.1f psi", self._p_res)
        logger.info("Noise floor = %.2f psi", self._noise_floor)
        logger.info("Found %d PTA periods (%d segments).", len(pta_g), len(groups))
        return df_work
