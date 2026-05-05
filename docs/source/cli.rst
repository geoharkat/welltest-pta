Command-line interface
=======================

After installation a ``welltest-pta`` command is available on your
``$PATH``:

.. code-block:: bash

   $ welltest-pta --help
   usage: welltest-pta [-h] [--verbose] {analyze,detect,deconvolve,synthetic} ...

   Pressure Transient Analysis & DST toolkit
   (V8.1 detector + vSH04 deconvolution).

   ...

Four sub-commands are available.


``analyze`` — full pipeline
---------------------------

Parse :math:`\rightarrow` detect :math:`\rightarrow` cross-validate
:math:`\rightarrow` plot :math:`\rightarrow` export, all in one go.

.. code-block:: bash

   welltest-pta analyze DST_WELL-6.txt \
       --output ./results \
       --cv \
       --plot \
       --per-event

Options:

================  =========================================================
``--output``      Output directory (default ``welltest_pta_output``)
``--cv``          Run cross-validation and print the report
``--cv-n``        Number of bootstrap replicas (default 8)
``--plot``        Save composite PDF report
``--per-event``   Also write one CSV per detected event
================  =========================================================


``detect`` — print catalogue only
---------------------------------

Lightweight: no plotting, no export by default.

.. code-block:: bash

   welltest-pta detect DST_WELL-6.txt
   welltest-pta detect DST_WELL-6.txt --export catalogue.csv


``deconvolve`` — multi-event vSH04
----------------------------------

Run deconvolution on every DD/BU in the file, optionally save the
recovered response and a log–log diagnostic plot.

.. code-block:: bash

   welltest-pta deconvolve DST_WELL-6.txt \
       --q 850 \
       --nu 1e-2 \
       --n-nodes 60 \
       --export decon_response.csv \
       --plot decon_diagnostic.png

Options:

================  =========================================================
``--q``           Flow rate (STB/D) for any drawdown without ``ev.rate``
``--nu``          Regularisation weight (default 1e-2)
``--n-nodes``     Log-spaced response nodes (default 60)
``--export``      Save response data (CSV/Excel/JSON)
``--plot``        Save log-log diagnostic figure
================  =========================================================


``synthetic`` — generate test data
----------------------------------

Useful for tutorials, CI, or stress-testing the detector.

.. code-block:: bash

   welltest-pta synthetic --output synth.csv --n 18000 --dt 4 --seed 42


Global options
--------------

============  ===========================================================
``-v``, ``--verbose``   Verbose logging (DEBUG level)
``-h``, ``--help``      Show help for the current sub-command
============  ===========================================================


Exit codes
----------

============  ===========================================================
0             Success
1             Generic error (parse failure, invalid arguments, etc.)
============  ===========================================================
