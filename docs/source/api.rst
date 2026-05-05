API reference
=============

This page lists the full public API of :mod:`welltest_pta`, generated
from the source docstrings.

.. currentmodule:: welltest_pta


Top-level public API
--------------------

The following names are re-exported from the top-level package and each
has its own auto-generated reference page:

.. autosummary::
   :toctree: api/_autogen
   :nosignatures:

   WellTest
   Event
   EventCollection
   EventDetector
   EventDetectorConfig
   detect_events
   parse
   WellTestParser
   bourdet_derivative
   horner_extrapolation
   mdh_extrapolation
   identify_flow_regimes
   reservoir_parameters
   deconvolve
   DeconvolutionResult
   cross_validate_detector
   DetectorCVResult


Submodules
----------

Parser
^^^^^^

.. automodule:: welltest_pta.parser
   :members:
   :show-inheritance:
   :exclude-members: WellTestParser, parse


Detection
^^^^^^^^^

.. automodule:: welltest_pta.detection.detector
   :members:
   :show-inheritance:
   :exclude-members: EventDetector, EventDetectorConfig, detect_events


Events
^^^^^^

.. automodule:: welltest_pta.events
   :members:
   :show-inheritance:
   :exclude-members: Event, EventCollection


WellTest orchestrator
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: welltest_pta.welltest
   :members:
   :show-inheritance:
   :exclude-members: WellTest


Analysis primitives
^^^^^^^^^^^^^^^^^^^

.. rubric:: Bourdet derivative

.. automodule:: welltest_pta.analysis.bourdet
   :members:
   :exclude-members: bourdet_derivative

.. rubric:: Horner extrapolation

.. automodule:: welltest_pta.analysis.horner
   :members:
   :exclude-members: horner_extrapolation

.. rubric:: Miller–Dyes–Hutchinson (MDH)

.. automodule:: welltest_pta.analysis.mdh
   :members:
   :exclude-members: mdh_extrapolation

.. rubric:: Flow-regime identification

.. automodule:: welltest_pta.analysis.flow_regimes
   :members:
   :exclude-members: identify_flow_regimes

.. rubric:: Reservoir parameters

.. automodule:: welltest_pta.analysis.reservoir
   :members:
   :exclude-members: reservoir_parameters


Deconvolution
^^^^^^^^^^^^^

.. automodule:: welltest_pta.analysis.deconvolution
   :members:
   :show-inheritance:
   :exclude-members: deconvolve, DeconvolutionResult


Cross-validation
^^^^^^^^^^^^^^^^

.. automodule:: welltest_pta.validation.cross_validation
   :members:
   :show-inheritance:
   :exclude-members: cross_validate_detector, DetectorCVResult


Visualization
^^^^^^^^^^^^^

.. automodule:: welltest_pta.visualization.composite
   :members:


Utilities
^^^^^^^^^

.. automodule:: welltest_pta.utils.synthetic
   :members:


Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: welltest_pta.cli
   :members: main
