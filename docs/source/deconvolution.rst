Multi-event deconvolution
==========================

Multi-rate well-test deconvolution recovers a single equivalent
constant-rate response from any number of buildups and drawdowns,
dramatically extending the radius of investigation and revealing
late-time boundaries that no individual buildup is long enough to see.


Mathematical framework
----------------------

The convolution problem for a variable-rate well test is

.. math::

   p_i - p_{wf}(t) \;=\; \int_0^t q(\tau)\,p'_u(t-\tau)\,d\tau

where:

* :math:`p_i` is the initial reservoir pressure,
* :math:`p_{wf}(t)` is the measured flowing/shut-in pressure,
* :math:`q(\tau)` is the (piecewise-constant) rate history,
* :math:`p_u(t)` is the **unit-rate constant-rate response** — the
  unknown — and :math:`p'_u = dp_u/dt` is its derivative.

Direct deconvolution (recovering :math:`p_u` by deconvolving the rate
history) is famously ill-posed: tiny noise in :math:`p_{wf}` produces
unbounded errors in :math:`p_u`. Regularisation is mandatory.


The vSH04 encoded formulation
-----------------------------

We follow von Schroeter, Hollaender and Gringarten (2004), encoding the
unknown as

.. math::

   z(\sigma) \;=\; \ln\!\left[\,t\,\frac{dp_u}{dt}\,\right]
              \;=\; \ln\!\left[\,\frac{dp_u}{d\ln t}\,\right],
   \qquad \sigma = \ln t.

Two crucial properties follow:

1. **Positivity by construction.**
   :math:`p'_u(t) = e^{z(\ln t)}/t \geq 0`, so the recovered derivative
   cannot become negative — even when noise would otherwise drive it
   below zero.

2. **Logarithmic regularisation.**
   The flow-regime signatures (WBS, IARF, linear, bilinear, …) appear
   as straight lines on a log–log plot. Smoothness on the
   :math:`\sigma = \ln t` axis is the *physically meaningful* notion of
   smoothness for a Bourdet derivative.

The objective is

.. math::

   J(\mathbf{z}, p_i) \;=\;
   \underbrace{\| \mathbf{y} - C(q, \mathbf{z}) - p_i\,\mathbf{1} \|^2_2}_{\text{data fit}}
   \;+\;
   \nu \;
   \underbrace{\| D\,\mathbf{z}\|^2_2}_{\text{curvature}}

where:

* :math:`\mathbf{y}` is the observed pressure vector,
* :math:`C(q, \mathbf{z})` is the convolution operator applied to the
  rate history :math:`q` and the encoded response :math:`\mathbf{z}`,
* :math:`D` is a centred second-difference matrix on the log-spaced
  :math:`\sigma`-grid, so :math:`\| D\mathbf{z}\|^2 \approx
  \int |z''(\sigma)|^2 d\sigma`,
* :math:`\nu` is the user-set regularisation weight.

Both :math:`\mathbf{z}` *and* :math:`p_i` are recovered jointly (set
``fit_p_initial=False`` to fix :math:`p_i` instead).


Choosing :math:`\nu`
^^^^^^^^^^^^^^^^^^^^

The regularisation weight :math:`\nu` controls the bias–variance
trade-off:

================  ===========================================================
:math:`\nu`       Effect
================  ===========================================================
small (≤ 1e−3)    Low bias, high variance — derivative may oscillate.
                  Use for low-noise tests with abundant late-time data.
moderate (1e−2)   Default. Typical balance for clean DSTs.
large (≥ 1e−1)    High bias, low variance — derivative is very smooth.
                  Use for noisy mechanical-gauge data.
================  ===========================================================

A practical strategy: start at :math:`\nu = 10^{-2}`, then **L-curve**
or **GCV** to refine if needed (not yet implemented in this package —
contributions welcome).


API
---

The entry point is :func:`welltest_pta.deconvolve`:

.. code-block:: python

   from welltest_pta import deconvolve

   res = deconvolve(
       events=wt.events,            # iterable of Event
       default_q=850,               # STB/D for DDs without ev.rate
       nu=1e-2,                     # regularisation weight
       n_response_nodes=60,         # log-spaced grid points
       t_response_min=1e-3,         # smallest Δt of recovered response (hr)
       t_response_max=None,         # default: max observation time
       p_initial=None,              # solve jointly with z (default)
       fit_p_initial=True,
       max_iter=200,
       verbose=False,
   )

The return type is :class:`welltest_pta.DeconvolutionResult`, which
carries:

==================  =====================================================
Attribute            Meaning
==================  =====================================================
``t``               Recovered response grid (hr), log-spaced
``pu``              Unit-rate cumulative response (psi per unit :math:`q`)
``dpu_dlnt``        Bourdet log-derivative
``z``               Encoded variable :math:`z = \ln(dp_u/d\ln t)`
``p_initial``       Recovered (or fixed) :math:`p_i` (psi)
``fit_pressure``    Reconstructed :math:`p(t_{\text{obs}})`
``residual_norm``   :math:`\|\mathbf{y} - \mathbf{p}_{\text{fit}}\|_2`
==================  =====================================================

Convenience methods:

* :py:meth:`~welltest_pta.DeconvolutionResult.plot` — log–log of recovered
  :math:`p_u` and its derivative.
* :py:meth:`~welltest_pta.DeconvolutionResult.export` — write CSV / Excel
  / JSON.
* :py:meth:`~welltest_pta.DeconvolutionResult.to_dataframe` — long-form
  Pandas DataFrame.


Implementation notes
--------------------

* The non-linear least-squares problem is solved with **Levenberg–
  Marquardt** via :func:`scipy.optimize.least_squares` (``method="lm"``).
* Default residual tolerances are :math:`10^{-9}`.
* The convolution operator uses **piecewise-linear log-time
  interpolation** of :math:`p_u` on the response grid; this preserves
  the slopes of the various flow regimes when sampled.
* Initial guess for :math:`\mathbf{z}` is a constant level chosen so
  that :math:`p_u(t_{\max}) \approx \max p - \min p` at the dominant
  rate.


Best-practice checklist
-----------------------

For a successful deconvolution:

1. Include **both** the immediately-preceding drawdown(s) and the
   buildup(s) you care about. Drawdowns provide the late-time
   constraint that buildups alone cannot.
2. **Clean the buildup tails** before deconvolving (the V8.1 detector
   does this automatically, but for manually-split events check that
   :math:`p_{\text{end}}` is a true plateau, not a pull-spike).
3. Use **consistent rate units** throughout (typically STB/D for oil).
   The recovered :math:`p_u` will be in *psi per unit* :math:`q`.
4. If :math:`\nu = 10^{-2}` produces a wiggly derivative, increase by
   :math:`\times 10`. If it over-smooths a known flow regime,
   decrease by :math:`\times 10`.


References
----------

* von Schroeter, T., Hollaender, F., & Gringarten, A. C. (2004).
  *Deconvolution of well-test data as a nonlinear total least-squares
  problem.* SPE Journal **9** (4), 375–390.
* Levitan, M. M. (2005). *Practical application of pressure/rate
  deconvolution to analysis of real well tests.* SPE 84290.
* Gringarten, A. C. (2008). *From straight lines to deconvolution: the
  evolution of the state of the art in well test analysis.* SPE
  Reservoir Evaluation & Engineering **11** (1), 41–62.
