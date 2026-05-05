Contributing
============

Contributions are welcome — bug reports, feature requests, documentation
improvements, and pull requests.


Reporting bugs
--------------

Please open an issue on
`GitHub <https://github.com/geoharkat/welltest-pta/issues>`_ with:

* The version of ``welltest-pta`` (``python -c "import welltest_pta; print(welltest_pta.__version__)"``).
* The Python version and OS.
* A **minimal reproducible example** — ideally something that runs on
  the synthetic data generator so anybody can reproduce locally.
* The exact traceback (if any).


Setting up a development environment
------------------------------------

.. code-block:: bash

   git clone https://github.com/geoharkat/welltest-pta.git
   cd welltest-pta
   python -m venv .venv
   source .venv/bin/activate              # Linux/macOS
   # .venv\\Scripts\\Activate.ps1          # Windows PowerShell
   pip install -e ".[dev,docs,excel]"


Running the test suite
----------------------

.. code-block:: bash

   pytest -v                              # all tests
   pytest -k bourdet                      # tests matching a pattern
   pytest --cov=welltest_pta --cov-report=html
   open htmlcov/index.html                # macOS — coverage report

Targets:

* All tests must pass on Python 3.9 → 3.12.
* Coverage should not decrease.
* New features need at least one test.


Linting and type checking
-------------------------

We use **ruff** for linting and **mypy** for type checking:

.. code-block:: bash

   ruff check src/ tests/ examples/
   ruff format --check src/ tests/
   mypy src/welltest_pta

CI will reject code that fails ``ruff check`` or ``ruff format --check``.


Building the documentation
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/build/html
   python -m http.server -d docs/build/html 8000

Then open http://localhost:8000.


Pull request workflow
---------------------

1. Fork the repository and create a branch off ``main``:

   .. code-block:: bash

      git checkout -b feature/my-thing

2. Make your changes. Keep commits small and well-scoped.

3. Add or update tests. Documentation updates are appreciated for any
   user-facing change.

4. Run the full validation suite locally:

   .. code-block:: bash

      ruff check src/ tests/ examples/
      ruff format --check src/ tests/
      mypy src/welltest_pta
      pytest -v

5. Push and open a pull request against ``main``. The CI workflow will
   run automatically.


Release process (maintainers)
-----------------------------

1. Bump ``__version__`` in ``src/welltest_pta/__version__.py``.
2. Add a section to ``CHANGELOG.md`` under ``[X.Y.Z] — YYYY-MM-DD``.
3. Commit and tag:

   .. code-block:: bash

      git commit -am "Release vX.Y.Z"
      git tag vX.Y.Z
      git push && git push --tags

4. The ``release.yml`` workflow takes over: builds wheel + sdist,
   publishes to PyPI via OIDC trusted publishing, and creates a GitHub
   release with the changelog section as release notes.


Coding style
------------

* **Docstrings** use NumPy style — see existing modules for examples.
* **Equations** in docstrings use raw-string form so backslashes are
  preserved, e.g. ``r"\\Delta p = ..."``.
* Public functions get **type hints**.
* Avoid abbreviations except where the petroleum domain demands them
  (DD, BU, IARF, MDH, FVF, etc.).
* Field oilfield units throughout (psi, ft, cp, RB/STB, mD, …) unless
  explicitly stated.


Code of conduct
---------------

Be kind, professional, and assume good intent. Personal attacks,
harassment, or off-topic political discussion are not welcome.
