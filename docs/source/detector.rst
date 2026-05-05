Event detection — V8.1 algorithm
=================================

The :class:`~welltest_pta.EventDetector` (algorithm version **V8.1**)
implements a fully automatic, multi-stage pipeline that classifies every
sample of a gauge record as one of three labels:

============  =========================================
``drawdown``  Producing period (pressure dropping toward
              flowing bottom-hole)
``buildup``   Shut-in period (pressure recovering toward
              :math:`P_{\text{res}}`)
``non_pta``   Edge effects (RIH/POOH), pauses, transients
              not suitable for PTA
============  =========================================

The algorithm has been validated on Rhourde Nouss (Algeria) DSTs and
offshore Qatar North Field tests.


Pipeline overview
-----------------

.. list-table::
   :header-rows: 1
   :widths: 8 35 57

   * - Phase
     - Step
     - Purpose
   * - 0
     - Hampel-filter despike → Savitzky–Golay smoothing → noise floor :math:`\hat\sigma`
     - Suppress single-sample spikes; obtain smoothed pressure and
       its time derivative; estimate the per-test noise level.
   * - 1
     - Reservoir-pressure plateau detection
     - Identify the most stable, highest-pressure region and report
       it as :math:`P_{\text{res}}`.
   * - 2
     - RIH / POOH edge masking
     - Trim away gauge-going-into-hole and out-of-hole transients.
   * - 3
     - Spike-boundary + turning-point detection (validated on :math:`\pm 5\hat\sigma`)
     - Find candidate event boundaries from large derivative
       excursions; verify each by sustained pressure change.
   * - 4
     - Zone classification using net-:math:`\Delta P` signed logic
     - Label each zone DD / BU / pause according to the sign and
       magnitude of net pressure change.
   * - 5
     - Pause absorption → same-type merge → edge trimming
     - Clean up tiny gaps, merge adjacent same-type zones, drop
       leading/trailing artefacts.
   * - 5b
     - **V8.1**: Post-plateau tail trim (H→I→J spike before POOH)
     - Detect and remove the characteristic "spike before pull"
       artefact at the tail of long buildups.


Phase 0 — Smoothing & noise floor
---------------------------------

A vectorised **Hampel filter** (rolling median ± 3 ⋅ MAD by default)
removes single-sample outliers without smearing the signal:

.. math::

   \hat p_i =
   \begin{cases}
     \text{median}_{j\in W_i}(p_j) & \text{if } |p_i - m_i| > k\,\sigma_i \\
     p_i                            & \text{otherwise}
   \end{cases}

where :math:`m_i` is the local median, :math:`\sigma_i = 1.4826\,\text{MAD}_i`,
and :math:`k = 3` (configurable via ``hampel_sigma``).

The despiked series is then **Savitzky–Golay smoothed** with an
adaptive window (default :math:`\sim 0.5\%` of the series length, polynomial
order 3) to obtain :math:`p_{\text{smooth}}` and its first derivative
:math:`dp/dt`.

The **noise floor** is

.. math::

   \hat\sigma = \max\!\big(0.5,\ Q_{75}\!\left(|p_{\text{raw}} - p_{\text{smooth}}|\right)\big)

and is the unit of all subsequent thresholds.


Phase 1 — Reservoir pressure
----------------------------

The reservoir pressure :math:`P_{\text{res}}` is estimated as the mean
of the highest stable plateau in :math:`p_{\text{smooth}}`. A plateau is a
contiguous region of :math:`\geq 30` points (default ``p_res_min_pts``)
where :math:`|dp/dt|` is below the 20th percentile of all
:math:`|dp/dt|` values.

If no plateau is found the 95th percentile of :math:`p_{\text{smooth}}` is
used as a fallback.


Phase 2 — Edge masking
----------------------

Two pointers — ``pta_start`` and ``pta_end`` — bracket the portion of
the record where the pressure is "near" :math:`P_{\text{res}}`. The
threshold defaults to :math:`0.85\,P_{\text{res}}` (or :math:`0.70` for
low-pressure shallow tests). Anything outside this window is forced to
``non_pta``.


Phase 3 — Boundary detection
----------------------------

Candidate boundaries are unioned from two independent detectors:

1. **dp/dt spikes** — points where :math:`|dp/dt|` exceeds the 95th
   percentile (configurable via ``spike_percentile``).
2. **Pressure turning points** — sign changes of :math:`d^2p/dt^2` on a
   heavily smoothed copy of the signal, retained only if the prominence
   exceeds :math:`10\,\hat\sigma`.

Each candidate is then **validated**: the median pressure in
:math:`[\,b - W,\,b\,)` and :math:`[\,b,\,b + W\,)` must differ by
:math:`> 5\,\hat\sigma`. Otherwise the candidate is dropped as noise.


Phase 4 — Zone classification
-----------------------------

The validated boundaries partition ``[pta_start, pta_end]`` into zones.
For each zone the signed net pressure change

.. math::

   \Delta p_{\text{net}} = \text{median}(p_{\text{end}}) - \text{median}(p_{\text{start}})

is computed (over 5-point windows on each side). The zone is labelled:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Condition
     - Label
     - Notes
   * - :math:`|\Delta p_{\text{net}}| < \max(15,\,5\hat\sigma)` **or** duration < 6 min
     - ``pause``
     - Will be absorbed in Phase 5
   * - :math:`\Delta p_{\text{net}} < 0`
     - ``drawdown``
     -
   * - :math:`\Delta p_{\text{net}} > 0`
     - ``buildup``
     -


Phase 5 — Cleanup
-----------------

Three passes run in sequence:

1. **Pause absorption** — every ``pause`` zone is merged into its larger
   PTA neighbour (or split-merged if neighbours are different types).
2. **Same-type merge** — adjacent zones with identical labels are
   joined; tiny non-PTA gaps (:math:`< 20` pts) between same-type events
   are absorbed.
3. **Edge trimming** — leading buildups that start above
   :math:`0.80\,P_{\text{res}}` (RIH artefact) and trailing drawdowns
   followed only by ``non_pta`` (POOH artefact) are demoted to non-PTA.

The whole process iterates until the label vector is stable.


Phase 5b — Tail trim (V8.1)
---------------------------

Long buildups often exhibit a characteristic *late-time spike* just
before the gauge is pulled. The V8.1 tail-trim heuristic targets this
artefact:

For each buildup with duration :math:`\geq 4` h and
:math:`\geq 200` pts:

1. Estimate the dominant pressure level :math:`p_{\text{plateau}}` from the
   mode of a 60-bin histogram.
2. Compute the on-plateau coverage on the late half of the event:
   :math:`\text{cov}_{\text{late}} = \text{frac}(|p - p_{\text{plateau}}|
   < 4\hat\sigma)`. Skip if :math:`< 30\%`.
3. Find the last sample on plateau; require that this sample lies in
   the second :math:`40\%` of the event (otherwise it's not a tail-spike
   but a real recovery).
4. Compute the tail deviation
   :math:`\delta_{\text{tail}} = \max|p_{\text{tail}} - p_{\text{plateau}}|`.
5. If :math:`\delta_{\text{tail}} > 8\hat\sigma` and the tail duration
   :math:`> 0.30` h, **demote the tail to non-PTA**.

This recovers a clean buildup whose Bourdet derivative is no longer
contaminated by the pull-induced spike.


Configuration
-------------

All thresholds live in :class:`~welltest_pta.EventDetectorConfig`:

.. code-block:: python

   from welltest_pta import EventDetectorConfig, WellTest

   cfg = EventDetectorConfig(
       hampel_sigma=3.0,
       spike_percentile=95.0,
       min_pta_dp_psi=15.0,
       min_pta_duration_hr=0.10,
       tail_trim_enabled=True,
       tail_trim_min_dur_hr=4.0,
       tail_trim_dev_n_sigma=8.0,
   )
   wt = WellTest.from_file("DST.txt", cfg=cfg)

For the full list of fields see :class:`~welltest_pta.EventDetectorConfig`.


When to override the auto-detector
----------------------------------

V8.1 is robust on standard DSTs but can struggle with:

* Tests with **very short** drawdowns (:math:`< 6` min) — increase
  ``min_pta_duration_hr``.
* Multiphase wells where the plateau pressure drifts slowly — relax
  ``tail_trim_min_plateau_frac``.
* High-noise mechanical gauges — increase ``hampel_sigma`` to 4–5 and
  ``spike_percentile`` to 97–98.

Always check the cross-validation score (see :doc:`validation`) before
trusting auto-detection on a critical test. If the score drops below
60/100, fall back to manual splitting via
:meth:`~welltest_pta.WellTest.split_manual`.
