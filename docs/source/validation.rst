Cross-validation
================

The auto-detector is robust on standard DSTs but no algorithm is
infallible. The :func:`welltest_pta.cross_validate_detector` function
quantifies how trustworthy the auto-detection is on **this particular
dataset** with a 0–100 confidence index.


Three independent stability checks
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Check
     - Method
     - Weight
   * - Bootstrap event-count
     - :math:`K` random downsample replicas (default
       :math:`f = 0.85`); report :math:`\sigma` of detected
       :math:`n_{DD}` and :math:`n_{BU}`.
     - 0.40
   * - Jaccard edge overlap
     - Compare the binary "is-PTA" mask of each replica to the
       reference mask via :math:`J = |A \cap B| / |A \cup B|`.
     - 0.40
   * - Parameter sensitivity
     - Sweep :math:`\pm 20\%` around each of
       ``hampel_sigma``, ``spike_percentile``, ``min_pta_dp_psi``,
       ``tail_trim_dev_n_sigma``; report the worst-case event-count
       drift.
     - 0.20


Composite score
---------------

The three components combine linearly:

.. math::

   S \;=\; 100 \cdot \big(\,
     0.40 \cdot s_{\text{boot}}
     + 0.40 \cdot \overline{J}
     + 0.20 \cdot s_{\text{sens}}
   \,\big)

where :math:`s_{\text{boot}}, s_{\text{sens}} \in [0, 1]` are normalised
penalties (lower :math:`\sigma` and lower :math:`\Delta n` are better)
and :math:`\overline{J}` is the mean Jaccard overlap.

The score is mapped to a human-readable grade:

============  ==========================================================
Score range   Grade
============  ==========================================================
80 – 100      **HIGHLY ROBUST** — manual review optional
60 – 80       **REASONABLE** — spot-check critical events
40 – 60       **MARGINAL** — recommend manual splitting
0 – 40        **UNSTABLE** — manual splitting strongly advised
============  ==========================================================


Usage
-----

The simplest way is to enable CV when loading the test:

.. code-block:: python

   from welltest_pta import WellTest

   wt = WellTest.from_file("DST.txt", cross_validate=True, cv_n_bootstrap=8)

A pretty-printed report is emitted to stdout. The result is also
attached to ``wt.cv_result`` for programmatic access:

.. code-block:: python

   print(wt.cv_result.overall_score)   # e.g. 82.4
   print(wt.cv_result.grade)           # e.g. "HIGHLY ROBUST"

For finer control, call :func:`~welltest_pta.cross_validate_detector`
directly:

.. code-block:: python

   from welltest_pta import cross_validate_detector
   from welltest_pta.parser import parse

   df = parse("DST.txt")
   res = cross_validate_detector(
       df,
       n_bootstrap=12,                 # more replicas → tighter σ
       downsample_frac=0.80,
       perturbation=0.25,              # ±25% sensitivity sweep
       seed=42,
   )

Returned object: :class:`~welltest_pta.DetectorCVResult`.


Sample report
-------------

.. code-block:: text

   ════════════════════════════════════════════════════════════════════════
     EVENT-DETECTOR CROSS-VALIDATION REPORT
   ════════════════════════════════════════════════════════════════════════
     Reference detection:   2 drawdowns, 2 buildups

     Bootstrap stability  (K = 8 replicas):
       n_drawdowns:      2.00 ± 0.00
       n_buildups:       2.00 ± 0.00

     Edge-position consistency (Jaccard overlap):
       mean = 0.952   (1.0 = perfect)
       std  = 0.018

     Parameter sensitivity (Δ events under ±20 % perturbation):
       hampel_sigma             Δ_dd = +0,  Δ_bu = +0
       spike_percentile         Δ_dd = +0,  Δ_bu = +0
       min_pta_dp_psi           Δ_dd = +0,  Δ_bu = +0
       tail_trim_dev_n_sigma    Δ_dd = +0,  Δ_bu = +0

     ─ OVERALL CV SCORE:   86.2 / 100   (HIGHLY ROBUST)
   ════════════════════════════════════════════════════════════════════════


When the score is low
---------------------

If :math:`S < 60`:

1. Re-run with ``cv_print=True`` to see *which* component is failing.
2. **Bootstrap σ high** ⇒ the test is short or has few clean events;
   try increasing ``min_pta_duration_hr``.
3. **Jaccard low** ⇒ event boundaries are unstable across replicas;
   tighten ``hampel_sigma`` and ``spike_percentile``.
4. **Sensitivity high** ⇒ events are right at the edge of detection
   thresholds; raise ``min_pta_dp_psi``.
5. If problems persist, fall back to manual splitting via
   :meth:`~welltest_pta.WellTest.split_manual`.
