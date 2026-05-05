Examples
========

The ``examples/`` directory in the source tree contains four runnable
scripts. Each is self-contained and uses the synthetic-data generator
so it can run on any machine without external files.


``quick_start.py``
------------------

A three-line tour of the package: load synthetic data, auto-detect
events, run Bourdet / Horner / reservoir-parameter analysis on the
longest buildup.

.. literalinclude:: ../../examples/quick_start.py
   :language: python
   :linenos:


``manual_split.py``
-------------------

Demonstrates the workflow when the auto-detector's CV score is
marginal. After auto-detection, the user passes a list of
``(type, t_start, t_end)`` tuples to
:meth:`~welltest_pta.WellTest.split_manual`, which rebuilds the
:class:`~welltest_pta.events.EventCollection` from explicit timestamps.

.. literalinclude:: ../../examples/manual_split.py
   :language: python
   :linenos:


``deconvolution_demo.py``
-------------------------

Multi-event deconvolution on a 6-event synthetic test. The recovered
unit-rate response merges all DDs and BUs into one diagnostic plot.

.. literalinclude:: ../../examples/deconvolution_demo.py
   :language: python
   :linenos:


``full_workflow.py``
--------------------

End-to-end pipeline: load → cross-validate → per-event analytics →
reservoir parameters → deconvolution → composite report → bulk export.
The recommended starting point for a real analysis.

.. literalinclude:: ../../examples/full_workflow.py
   :language: python
   :linenos:


Running the examples
--------------------

After installing the package in development mode:

.. code-block:: bash

   pip install -e ".[dev]"
   python examples/quick_start.py
   python examples/manual_split.py
   python examples/deconvolution_demo.py
   python examples/full_workflow.py

For headless servers (no display) set the matplotlib backend:

.. code-block:: bash

   MPLBACKEND=Agg python examples/full_workflow.py
