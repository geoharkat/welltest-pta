Quick start
===========

This page walks through the entire pipeline on a synthetic DST so it can
be reproduced without any external file.


1. Load gauge data
------------------

For a real test you would use :meth:`~welltest_pta.WellTest.from_file`:

.. code-block:: python

   from welltest_pta import WellTest

   wt = WellTest.from_file("DST_WELL-6.txt", cross_validate=True)

For this tutorial we generate a multi-rate synthetic test:

.. code-block:: python

   from welltest_pta import WellTest
   from welltest_pta.utils.synthetic import generate_synthetic_dst

   df = generate_synthetic_dst(
       n_samples=20_000,
       sample_period_s=4.0,
       sequence=[
           ("DD", 0.5, 3300.0),    # initial flow
           ("BU", 1.0, 4490.0),    # short BU
           ("DD", 1.0, 3000.0),    # main DD
           ("BU", 8.0, 4495.0),    # extended BU
       ],
   )
   wt = WellTest.from_dataframe(df)


2. Inspect the catalogue
------------------------

.. code-block:: python

   wt.print_summary()

.. code-block:: text

   ════════════════════════════════════════════════════════════════════════
     WELL TEST SUMMARY
   ════════════════════════════════════════════════════════════════════════
     Samples:        20 000
     P_reservoir:    4 495.10 psi
     Noise floor:    0.97 psi
     Events:         4  (2 DD, 2 BU)
   ════════════════════════════════════════════════════════════════════════
    event_id      type  duration_hr  p_initial  p_final  delta_p  rate_psi_hr  n_points
        DD-1  drawdown         0.50    4495.06  3300.40 -1194.66     -2389.32       450
        BU-1   buildup         1.00    3300.51  4488.10  1187.59     +1187.59       900
        DD-2  drawdown         1.00    4486.43  3010.55 -1475.88     -1475.88       900
        BU-2   buildup         8.00    3010.55  4495.30  1484.75      +185.59      7200


3. Per-event analysis
---------------------

Every event is a first-class :class:`~welltest_pta.Event` with
PTA-specific methods:

.. code-block:: python

   bu = wt.events["BU-2"]                          # by event_id
   bu.print()

   # Bourdet log–log diagnostic
   bu.plot_loglog()

   # Horner extrapolation
   h = bu.horner()
   print(f"P*    = {h['p_star']:.1f} psi")
   print(f"slope = {h['slope_m']:.2f} psi/cycle")
   print(f"R²    = {h['r2']:.4f}")

   # Reservoir parameters (oilfield units)
   params = bu.reservoir_params(
       q=850, mu=0.45, B=1.18,                     # rate, viscosity, FVF
       h=18, phi=0.12, ct=1.2e-5, rw=0.108,        # net pay, φ, c_t, r_w
       method="horner",
   )
   print(f"k    = {params['k']:.3f} mD")
   print(f"kh   = {params['kh']:.1f} mD·ft")
   print(f"skin = {params['skin']:+.3f}")


4. Manual splitting (when CV score is marginal)
-----------------------------------------------

If :func:`~welltest_pta.cross_validate_detector` returns a confidence below
60/100, override the auto-detector with explicit timestamps:

.. code-block:: python

   wt.split_manual([
       ("DD", "2025-01-15 10:00", "2025-01-15 12:30"),
       ("BU", "2025-01-15 12:30", "2025-01-15 18:00"),
       ("DD", "2025-01-15 18:00", "2025-01-15 20:00"),
       ("BU", "2025-01-15 20:00", "2025-01-16 04:00"),
   ])

The :class:`~welltest_pta.events.EventCollection` is rebuilt in place;
all per-event methods continue to work unchanged.


5. Multi-event deconvolution
----------------------------

Merge all DDs and BUs into one equivalent unit-rate response:

.. code-block:: python

   from welltest_pta import deconvolve

   recon = deconvolve(
       wt.events,
       default_q=850,                  # STB/D for any DD without ev.rate
       nu=1e-2,                        # regularisation
       n_response_nodes=60,
   )

   print(f"converged = {recon.converged}, ||r|| = {recon.residual_norm:.2f} psi")
   recon.plot()                        # log–log of merged response
   recon.export("decon.csv")

See :doc:`deconvolution` for algorithmic details.


6. Composite report + bulk export
---------------------------------

.. code-block:: python

   wt.plot_composite(out_path="report.pdf")

   wt.export_all(
       out_dir="./output",
       prefix="DST_WELL-6",
       per_event=True,                 # one CSV per event
   )


Next steps
----------

* :doc:`detector`        — V8.1 algorithm details
* :doc:`deconvolution`   — vSH04 encoded formulation
* :doc:`validation`      — cross-validation scoring
* :doc:`api`             — full API reference
* :doc:`cli`             — command-line tool
