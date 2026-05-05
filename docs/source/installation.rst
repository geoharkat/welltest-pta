Installation
============

Requirements
------------

* Python **3.9 or newer**
* NumPy ≥ 1.22
* Pandas ≥ 1.4
* SciPy ≥ 1.8
* Matplotlib ≥ 3.5

All dependencies are pure-Python wheels and install automatically with
``pip``.


From PyPI
---------

The release version:

.. code-block:: bash

   pip install welltest-pta

With Excel-export support (``openpyxl`` + ``xlsxwriter``):

.. code-block:: bash

   pip install "welltest-pta[excel]"

Everything (Excel + dev tooling + docs):

.. code-block:: bash

   pip install "welltest-pta[all,dev,docs]"


From source (development install)
---------------------------------

.. code-block:: bash

   git clone https://github.com/geoharkat/welltest-pta.git
   cd welltest-pta
   pip install -e ".[dev]"

Run the test suite:

.. code-block:: bash

   pytest -v

Build the docs locally:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/build/html


Verifying the installation
--------------------------

After installation a short Python session should look like this:

.. code-block:: python

   >>> import welltest_pta
   >>> welltest_pta.__version__
   '0.1.0'
   >>> from welltest_pta import WellTest
   >>> from welltest_pta.utils.synthetic import generate_synthetic_dst
   >>> wt = WellTest.from_dataframe(generate_synthetic_dst(n_samples=4000))
   >>> wt.events
   EventCollection(2 events: 1 DD, 1 BU)

The CLI tool should also be on your ``$PATH``:

.. code-block:: bash

   $ welltest-pta --help

If installation succeeded but ``welltest-pta`` isn't found, your
``pip`` ``Scripts/`` directory may not be on ``PATH`` — add it or use
``python -m welltest_pta.cli`` instead.


Optional features
-----------------

Some features depend on extras:

================================  =========================================
Feature                           Extra to install
================================  =========================================
Excel export (``.xlsx``)          ``[excel]``
Run the test suite                ``[dev]``
Build documentation locally       ``[docs]``
================================  =========================================
