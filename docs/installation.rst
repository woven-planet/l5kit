.. _install:

Installation
============

Installing as a User
--------------------

Follow this workflow if:

* you're not interested in developing and/or contributing to L5Kit;
* you don't need any features from a specific branch or latest master and you're fine with the latest release;
 
1. Install the package from pypy (in your project venv)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: console
    
    pip install l5kit

You should now be able to import from L5Kit (e.g. :code:`from l5kit.data import ChunkedDataset` should work)

2. Run example
++++++++++++++

Examples are not shipped with the package, but you can download the zip release from:
`L5Kit Releases <https://github.com/lyft/l5kit/releases>`_

Please download the zip matching your installed version (you can run :code:`pip freeze | grep l5kit` to get the right version)
Unzip the files and grab the example folder in the root of the project.

.. code-block:: console

    jupyter notebook examples/visualisation/visualise_data.ipynb


Installing as a Developer
-------------------------

Follow this workflow if:

+ you want to test latest master or another branch;
+ you want to contribute to L5Kit;
+ you want to test the examples using a non-release version of the code;

1. Clone the repo
+++++++++++++++++

.. code-block:: console
    
    git clone https://github.com/lyft/l5kit.git
    cd l5kit/l5kit


Please note the double :code:`l5kit` in the path, as we need to :code:`cd` where :code:`setup.py` file is.

2. Install L5Kit
++++++++++++++++

2.1 Deterministic Build (Suggested)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We support deterministic build through `pipenv <https://pipenv-fork.readthedocs.io/en/latest/>`_.

Once you've installed pipenv (or made it available in your env) run:

.. code-block:: console

    pipenv sync --dev


This will install all dependencies (:code:`--dev` includes dev-packages too) from the lock file.

2.2 Latest Build
~~~~~~~~~~~~~~~~

If you don't care about determinist builds or you're having troubles with packages resolution (Windows, Python<3.7, etc..),
you can install directly from the :code:`setup.py` by running:

.. code-block:: console

    pip install -e ."[dev]"


If you run into trouble installing L5Kit on Windows, you may need to

* install Pytorch and torchvision manually first (select the correct version required by your system, i.e. GPU or CPU-only), then run L5Kit install (remove the packages `torch <https://github.com/lyft/l5kit/blob/59f36f348682aac5fc488c6d39dd58f8c27b1ec6/l5kit/setup.py#L23>`_ and `torchvision <https://github.com/lyft/l5kit/blob/59f36f348682aac5fc488c6d39dd58f8c27b1ec6/l5kit/setup.py#L24>`_ from :code:`setup.py`)
* install `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.

3. Generate L5Kit code html documentation (optional)
++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: console

    sphinx-apidoc --module-first --separate -o API/ l5kit/l5kit l5kit/l5kit/tests*
    sphinx-build . docs


4. Run example
++++++++++++++

.. code-block:: console

    jupyter notebook examples/visualisation/visualise_data.ipynb

