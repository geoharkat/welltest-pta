r"""
welltest_pta.welltest
=====================
Top-level :class:`WellTest` orchestrator — the user-facing entry point.

Workflow (matches the original V8.1 pipeline + analytics):

>>> from welltest_pta import WellTest
>>>
>>> # 1) Load + auto-detect (with optional CV scores)
>>> wt = WellTest.from_file("DST.txt", cross_validate=True)
>>>
>>> # 2) Inspect detected events
>>> wt.events.print()
>>> wt.events["BU-2"].print()
>>>
>>> # 3) Manual override (if CV score was low)
>>> wt.split_manual([
...     ("DD",  "2025-01-15 10:00", "2025-01-15 12:30"),
...     ("BU",  "2025-01-15 12:30", "2025-01-15 18:00"),
... ])
>>>
>>> # 4) Per-event analysis
>>> bu = wt.events["BU-2"]
>>> bu.plot_loglog()
>>> params = bu.reservoir_params(q=850, mu=0.45, B=1.18, h=18,
...                              phi=0.12, ct=1.2e-5, rw=0.108)
>>>
>>> # 5) Multi-event deconvolution
>>> from welltest_pta import deconvolve
>>> recon = deconvolve(wt.events, default_q=850)
>>> recon.plot()
>>>
>>> # 6) Composite report
>>> wt.plot_composite(out_path="report.pdf")
>>> wt.export_all("output_dir/")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from welltest_pta.detection.detector import (
    EventDetector,
    EventDetectorConfig,
)
from welltest_pta.events import Event, EventCollection
from welltest_pta.parser import WellTestParser
from welltest_pta.validation.cross_validation import (
    DetectorCVResult,
    cross_validate_detector,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WellTest
# ─────────────────────────────────────────────────────────────────────────────

class WellTest:
    """
    Top-level handle for one well-test interpretation.

    A ``WellTest`` keeps three things in lock-step:
      * ``self.df``       — the parsed + annotated gauge DataFrame
      * ``self.events``   — the :class:`EventCollection` derived from ``self.df``
      * ``self.metadata`` — file-level metadata + detector / CV results

    Re-running event detection (via :meth:`detect` or :meth:`split_manual`)
    rebuilds ``self.events`` and updates ``self.df["event"]`` in place.
    """

    # ──────── construction ────────

    def __init__(
        self,
        df_raw: pd.DataFrame,
        metadata: Optional[dict[str, Any]] = None,
    ):
        if "timestamp" not in df_raw.columns or "pressure" not in df_raw.columns:
            raise ValueError("Input DataFrame must contain 'timestamp' and 'pressure'.")
        self._df_raw = df_raw.copy()
        self.df: Optional[pd.DataFrame] = None      # set by detect()
        self.events = EventCollection()
        self.metadata: dict[str, Any] = metadata or {}
        self.detector: Optional[EventDetector] = None
        self.cv_result: Optional[DetectorCVResult] = None

    # ──────── factory: from file ────────

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        cfg: Optional[EventDetectorConfig] = None,
        cross_validate: bool = False,
        cv_n_bootstrap: int = 8,
        cv_print: bool = True,
        auto_detect: bool = True,
    ) -> "WellTest":
        """
        Parse an ASCII gauge file, run the V8.1 detector, optionally
        cross-validate, and return a populated :class:`WellTest`.
        """
        path = Path(filepath)
        logger.info("Parsing %s", path.name)
        parser = WellTestParser()
        df_raw = parser.parse(path)
        if df_raw.empty:
            raise RuntimeError(f"Parser returned empty DataFrame for {path}")
        logger.info("Parsed %d rows  (cols: %s)", len(df_raw), list(df_raw.columns))

        wt = cls(df_raw, metadata={
            "filepath": str(path),
            "filename": path.name,
            "parser_metadata": parser.metadata,
            "parser_mapping": parser.mapping,
            "loaded_at": datetime.now().isoformat(timespec="seconds"),
        })

        if auto_detect:
            wt.detect(cfg=cfg)

        if cross_validate:
            wt.cv_result = cross_validate_detector(
                wt._df_raw,
                cfg=cfg,
                n_bootstrap=cv_n_bootstrap,
                print_report=cv_print,
            )

        return wt

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        cfg: Optional[EventDetectorConfig] = None,
        auto_detect: bool = True,
    ) -> "WellTest":
        """Build from an already-parsed DataFrame."""
        wt = cls(df, metadata={"source": "dataframe",
                                "loaded_at": datetime.now().isoformat(timespec="seconds")})
        if auto_detect:
            wt.detect(cfg=cfg)
        return wt

    # ──────── detection ────────

    def detect(self, cfg: Optional[EventDetectorConfig] = None) -> "WellTest":
        """Run (or re-run) automatic V8.1 event detection."""
        self.detector = EventDetector(cfg=cfg or EventDetectorConfig())
        self.df = self.detector.detect(self._df_raw)
        self.events = EventCollection.from_annotated_dataframe(
            self.df, p_reservoir=self.detector._p_res
        )
        self.metadata["p_reservoir"] = self.detector._p_res
        self.metadata["noise_floor"] = self.detector._noise_floor
        return self

    def cross_validate(
        self,
        cfg: Optional[EventDetectorConfig] = None,
        n_bootstrap: int = 8,
        print_report: bool = True,
    ) -> DetectorCVResult:
        """Run the bootstrap + sensitivity + Jaccard CV on the current data."""
        self.cv_result = cross_validate_detector(
            self._df_raw,
            cfg=cfg,
            n_bootstrap=n_bootstrap,
            print_report=print_report,
        )
        return self.cv_result

    # ──────── manual splitting ────────

    def split_manual(
        self,
        spec: list[tuple[str, Any, Any]],
        keep_existing_classifications: bool = False,
    ) -> "WellTest":
        """
        Override the auto-detected events with a manual list.

        Parameters
        ----------
        spec
            List of ``(type, t_start, t_end)`` tuples where:

            * ``type`` is one of ``"DD"``, ``"BU"``, ``"drawdown"``, ``"buildup"``
            * ``t_start``, ``t_end`` are timestamps (``str`` or ``pd.Timestamp``)

            Example::

                wt.split_manual([
                    ("DD", "2025-01-15 10:00:00", "2025-01-15 12:30:00"),
                    ("BU", "2025-01-15 12:30:00", "2025-01-15 18:00:00"),
                ])
        keep_existing_classifications
            If True, gaps between manual events keep their auto labels.
            If False (default), they are reset to ``non_pta``.

        Notes
        -----
        Calling this method requires that :meth:`detect` has already been
        run at least once, since it relies on the ``p_smooth`` / ``elapsed_hr``
        columns built in Phase 0.
        """
        if self.df is None:
            raise RuntimeError("Run .detect() at least once before manual splitting.")

        type_map = {
            "DD": "drawdown",
            "BU": "buildup",
            "drawdown": "drawdown",
            "buildup": "buildup",
        }

        df = self.df
        new_labels = (
            df["event"].copy() if keep_existing_classifications
            else pd.Series("non_pta", index=df.index, name="event")
        )

        ts = df["timestamp"]
        for entry in spec:
            if len(entry) != 3:
                raise ValueError(f"Bad spec entry: {entry!r} — expected (type, t0, t1).")
            etype, t0, t1 = entry
            if etype not in type_map:
                raise ValueError(f"Unknown type {etype!r}; use DD/BU/drawdown/buildup.")
            t0 = pd.Timestamp(t0)
            t1 = pd.Timestamp(t1)
            if t1 <= t0:
                raise ValueError(f"t_end ({t1}) must be > t_start ({t0}).")
            mask = (ts >= t0) & (ts <= t1)
            if not mask.any():
                logger.warning("Manual split (%s, %s, %s) matched zero rows.", etype, t0, t1)
                continue
            new_labels.loc[mask] = type_map[etype]

        df["event"] = new_labels
        self.df = df
        # Rebuild EventCollection
        self.events = EventCollection.from_annotated_dataframe(
            self.df, p_reservoir=self.metadata.get("p_reservoir")
        )
        logger.info("Manual splitting applied: %s", self.events)
        return self

    # ──────── shortcuts ────────

    @property
    def drawdowns(self) -> EventCollection:
        return self.events.drawdowns

    @property
    def buildups(self) -> EventCollection:
        return self.events.buildups

    @property
    def p_reservoir(self) -> Optional[float]:
        return self.metadata.get("p_reservoir")

    # ──────── summary / printing ────────

    def summary(self) -> dict[str, Any]:
        """Return a dict with file-, detector-, and event-level info."""
        out = {
            "filename": self.metadata.get("filename"),
            "n_samples": len(self._df_raw),
            "p_reservoir_psi": self.metadata.get("p_reservoir"),
            "noise_floor_psi": self.metadata.get("noise_floor"),
            "n_events": len(self.events),
            "n_drawdowns": len(self.drawdowns),
            "n_buildups": len(self.buildups),
        }
        if self.cv_result is not None:
            out["cv_score"] = round(self.cv_result.overall_score, 1)
            out["cv_grade"] = self.cv_result.grade
        return out

    def print_summary(self) -> None:
        """Print the high-level summary + the event catalogue."""
        s = self.summary()
        sep = "═" * 72
        print(f"\n{sep}\n  WELL TEST SUMMARY\n{sep}")
        print(f"  File:           {s['filename']}")
        print(f"  Samples:        {s['n_samples']}")
        print(f"  P_reservoir:    {s['p_reservoir_psi']:.2f} psi" if s['p_reservoir_psi'] else "")
        print(f"  Noise floor:    {s['noise_floor_psi']:.2f} psi" if s['noise_floor_psi'] else "")
        print(f"  Events:         {s['n_events']}  ({s['n_drawdowns']} DD, {s['n_buildups']} BU)")
        if "cv_score" in s:
            print(f"  CV score:       {s['cv_score']:.1f} / 100  ({s['cv_grade']})")
        print(sep)
        self.events.print()
        print()

    # ──────── plotting ────────

    def plot_composite(
        self,
        out_path: Optional[str | Path] = None,
        figsize: tuple[float, float] = (14, 10),
    ):
        """Composite 4-panel report (uses welltest_pta.visualization.composite)."""
        from welltest_pta.visualization.composite import plot_composite_report
        return plot_composite_report(self, out_path=out_path, figsize=figsize)

    def plot_overview(
        self,
        ax=None,
        show_events: bool = True,
        show_p_res: bool = True,
    ):
        """Single-panel pressure-vs-time overview with event shading."""
        from welltest_pta.visualization.composite import plot_overview
        return plot_overview(self, ax=ax, show_events=show_events,
                             show_p_res=show_p_res)

    # ──────── export ────────

    def export_all(
        self,
        out_dir: str | Path,
        prefix: str = "welltest",
        per_event: bool = True,
        catalogue_format: str = "csv",
    ) -> dict[str, Path]:
        """
        Export everything to a directory:

        * ``{prefix}_full_data.csv``      — the full annotated DataFrame
        * ``{prefix}_catalogue.csv``      — one row per event
        * ``{prefix}_metadata.json``      — file metadata + detector info
        * ``{prefix}_events/{id}.csv``    — one CSV per event (if per_event)

        Returns a dict of ``label → Path``.
        """
        import json

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        # 1. Full annotated data
        full = out / f"{prefix}_full_data.csv"
        self.df.to_csv(full, index=False)
        paths["full_data"] = full

        # 2. Catalogue
        cat_ext = {"csv": "csv", "excel": "xlsx", "json": "json"}.get(catalogue_format, "csv")
        cat = out / f"{prefix}_catalogue.{cat_ext}"
        self.events.export(cat, format=catalogue_format)
        paths["catalogue"] = cat

        # 3. Metadata JSON
        meta_path = out / f"{prefix}_metadata.json"
        meta_serialisable = {}
        for k, v in self.metadata.items():
            try:
                json.dumps(v, default=str)
                meta_serialisable[k] = v
            except TypeError:
                meta_serialisable[k] = str(v)
        if self.cv_result is not None:
            meta_serialisable["cv_score"] = self.cv_result.overall_score
            meta_serialisable["cv_grade"] = self.cv_result.grade
        with open(meta_path, "w") as f:
            json.dump(meta_serialisable, f, indent=2, default=str)
        paths["metadata"] = meta_path

        # 4. Per-event CSVs
        if per_event:
            ev_dir = out / f"{prefix}_events"
            ev_dir.mkdir(exist_ok=True)
            for ev in self.events:
                p = ev_dir / f"{ev.event_id}.csv"
                ev.export(p, format="csv")
                paths[ev.event_id] = p

        logger.info("Exported all → %s", out)
        return paths

    def __repr__(self) -> str:
        return (
            f"WellTest(file={self.metadata.get('filename')!r}, "
            f"n={len(self._df_raw)}, events={len(self.events)})"
        )
