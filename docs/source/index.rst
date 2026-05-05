.. welltest-pta documentation master file.

welltest-pta
============

**Pressure Transient Analysis (PTA) and Drill-Stem Test (DST) toolkit for Python.**

A complete, batteries-included pipeline for petroleum well-test interpretation:
robust ASCII parsing :math:`\rightarrow` automatic event detection
:math:`\rightarrow` per-event analytics (Bourdet derivative, Horner, MDH)
:math:`\rightarrow` flow-regime identification :math:`\rightarrow` reservoir
parameters (:math:`k`, :math:`kh`, skin, wellbore storage)
:math:`\rightarrow` multi-event deconvolution :math:`\rightarrow`
publication-quality plots.

.. image:: https://img.shields.io/pypi/v/welltest-pta.svg
   :target: https://pypi.org/project/welltest-pta/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/welltest-pta.svg
   :target: https://pypi.org/project/welltest-pta/
   :alt: Python versions

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://readthedocs.org/projects/welltest-pta/badge/?version=latest
   :target: https://welltest-pta.readthedocs.io/en/latest/
   :alt: Documentation

----

At a glance
-----------

.. code-block:: python

   from welltest_pta import WellTest, deconvolve

   # 1) Load + auto-detect events (with optional CV scoring)
   wt = WellTest.from_file("DST.txt", cross_validate=True)
   wt.print_summary()

   # 2) Per-event PTA analysis
   bu = wt.events.longest_buildup
   params = bu.reservoir_params(q=850, mu=0.45, B=1.18,
                                h=18, phi=0.12, ct=1.2e-5, rw=0.108)
   print(f"k = {params['k']:.2f} mD,  skin = {params['skin']:+.2f}")

   # 3) Multi-event deconvolution
   recon = deconvolve(wt.events, default_q=850)
   recon.plot()

   # 4) Bulk export
   wt.export_all("./output", per_event=True)

----

User guide
----------

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   cli

.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   detector
   deconvolution
   validation

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   examples
   contributing
   changelog


----

Citation
--------

If you use **welltest-pta** in research, please cite:

.. code-block:: bibtex

   @software{harkat2026welltestpta,
     author  = {Harkat, Ismail},
     title   = {welltest-pta: Pressure Transient Analysis Toolkit for Python},
     year    = {2026},
     url     = {https://github.com/geoharkat/welltest-pta},
     version = {0.1.0},
   }


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
