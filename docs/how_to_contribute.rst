.. _contribute:

How to contribute
=================

.. toctree::
   :maxdepth: 1

You are invited to contribute to the L5Kit with your examples and improvements.
These are peer-reviewed by the development team to maintain quality and reproducibility.

**Note:** All commands below must be run from the :code:`l5kit` folder (the one with the :code:`setup.py`).

License
-------

You will be required to sign a contributor agreement license upon a pull request (PR).


Installing l5kit as a developer
-------------------------------

Run:

.. code-block:: console

    pip install -r requirements.txt

to install all developer dependencies along with l5kit in editable (-e) mode.

Code Sanity checks
------------------

Before starting a PR, it is highly recommended to also install the git pre-commit hooks, run:

.. code-block:: console

    pre-commit install

This will run all required code checks before each commit and it ensures your builds won't fail in CI.

If, on the other hand, you want to run individual checks, please refer to the instructions below.

Testing
-------

To run the tests, run:

.. code-block:: console

    ./run_tests.sh tests


Coverage report
---------------

To generate a test coverage report, run:

.. code-block:: console

    # Outputs to std out
    pytest --cov

    # Output to HTML files within the coverage_report_html folder
    pytest --cov --cov-report html


Type checking
-------------

To run type checking with mypy, run:

.. code-block:: console

    ./run_tests.sh types


Code style, linting and formatting
----------------------------------

We use `isort <https://github.com/timothycrosley/isort>`_ for import sorting, and `flake8 <https://flake8.pycqa.org/en/latest/>`_ to check for linting errors.

You can check against those by running:

.. code-block:: console

    ./run_tests.sh lint

Or, if you want to apply those formatters:

.. code-block:: console

    # Sort imports automatically.
    isort l5kit --apply --recursive

    # Check linting errors.
    flake8 l5kit


Our docstrings are in `Google docstring <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ format.

Creating a distribution
-----------------------

.. code-block:: console

    # Clean up from potential earlier runs
    rm -rf dist
    rm -rf build

    python3 setup.py sdist bdist_wheel


You can now find the distribution files (both tar.gz and wheel) in the :code:`dist` folder.

**We look forward to your contributions!**
