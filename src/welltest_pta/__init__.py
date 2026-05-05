"""
welltest-pta — Pressure Transient Analysis (PTA) and Drill-Stem Test (DST) toolkit
==================================================================================

A complete Python pipeline for petroleum well-test analysis, designed to handle
the full workflow from raw gauge ASCII files to publication-ready PTA results.

Workflow
--------
1.  ``parse(filepath)``         — robust ASCII reader (any delimiter / encoding)
2.  ``WellTest.from_file(...)`` — load + auto-detect events + cross-validate
3.  ``wt.events``               — iterable of ``Event`` objects (DD-1, BU-1, ...)
4.  ``wt.split_manual(...)``    — override automatic splits by entering t_start / t_end
5.  ``event.bourdet()``         — log-log diagnostic for a single event
6.  ``event.horner()``          — Horner extrapolation for a single buildup
7.  ``event.mdh()``             — Miller-Dyes-Hutchinson semilog method
8.  ``event.flow_regimes()``    — automatic flow-regime identification
9.  ``event.reservoir_params(fluid_props)`` — k, kh, S, C
10. ``deconvolve(events, ...)`` — von Schroeter encoded deconvolution
11. ``wt.export(...)``          — CSV / Excel / JSON / KAPPA-Saphir
12. ``wt.plot_composite(...)``  — publication-quality 4-panel figure

Example
-------
>>> from welltest_pta import WellTest
>>> wt = WellTest.from_file("TCQR818_WELL-6_DST-1_Tubing.txt")
>>> print(wt.summary())
>>> bu = wt.buildups[-1]              # last (longest) buildup
>>> bu.plot_loglog()
>>> params = bu.reservoir_params(q=850, mu=0.45, ct=1.2e-5, phi=0.12, h=18, rw=0.108)
>>> print(f"k = {params['k']:.2f} mD,  S = {params['skin']:.2f}")

Author
------
Ismail Harkat (geoharkat) — Senior Wellsite/Operations Geologist, Sonatrach
"""

from welltest_pta.__version__ import __version__
from welltest_pta.parser import parse, WellTestParser
from welltest_pta.detection import (
    EventDetector,
    EventDetectorConfig,
    detect_events,
)
from welltest_pta.events import Event, EventCollection
from welltest_pta.welltest import WellTest
from welltest_pta.analysis.bourdet import bourdet_derivative
from welltest_pta.analysis.horner import horner_extrapolation, horner_diagnostic_line
from welltest_pta.analysis.mdh import mdh_extrapolation
from welltest_pta.analysis.flow_regimes import identify_flow_regimes
from welltest_pta.analysis.reservoir import reservoir_parameters
from welltest_pta.analysis.deconvolution import deconvolve, DeconvolutionResult
from welltest_pta.validation.cross_validation import (
    cross_validate_detector,
    DetectorCVResult,
)

__all__ = [
    "__version__",
    # Parsing
    "parse",
    "WellTestParser",
    # Detection
    "EventDetector",
    "EventDetectorConfig",
    "detect_events",
    # Events
    "Event",
    "EventCollection",
    "WellTest",
    # Analysis
    "bourdet_derivative",
    "horner_extrapolation",
    "horner_diagnostic_line",
    "mdh_extrapolation",
    "identify_flow_regimes",
    "reservoir_parameters",
    "deconvolve",
    "DeconvolutionResult",
    # Validation
    "cross_validate_detector",
    "DetectorCVResult",
]
