# Installation

## Installing as a User
Follow this workflow if:
 - you're not interested in developing and/or contributing to L5Kit;
 - you don't need any features from a specific branch or latest master and you're fine with the latest release;
 
### 1. Install the package from pypy (in your project venv)
```shell
pip install l5kit
```
You should now be able to import from L5Kit (e.g. `from l5kit.data import ChunkedDataset` should work)

### 2. Run example
Examples are not shipped with the package, but you can download the zip release from:
[L5Kit Releases](https://github.com/lyft/l5kit/releases)

Please download the zip matching your installed version (you can run `pip freeze | grep l5kit` to get the right version)
Unzip the files and grab the example folder in the root of the project.

```shell
jupyter notebook examples/visualisation/visualise_data.ipynb
```

## Installing as a Developer
Follow this workflow if:
 - you want to test latest master or another branch;
 - you want to contribute to L5Kit;
 - you want to test the examples using a non-release version of the code;

### 1. Clone the repo
```shell
git clone https://github.com/lyft/l5kit.git
cd l5kit/l5kit
```

Please note the double `l5kit` in the path, as we need to `cd` where `setup.py` file is.

### 3. Install L5Kit

#### 3.1 Deterministic Build (Suggested)
We support deterministic build through [pipenv](https://pipenv-fork.readthedocs.io/en/latest/).

Once you've installed pipenv (or made it available in your env) run: 
```shell
pipenv sync --dev
```
This will install all dependencies (`--dev` includes dev-packages too) from the lock file.

#### 3.1 Latest Build
If you don't care about determinist builds or you're having troubles with packages resolution (Windows, Python<3.7, etc..),
you can install directly from the `setup.py` by running:
```shell
pip install -e ."[dev]"
```

If you run into trouble installing L5Kit on Windows, you may need to
- install Pytorch and torchvision manually first (select the correct version required by your system, i.e. GPU or CPU-only), then run L5Kit install (remove the packages [torch](https://github.com/lyft/l5kit/blob/59f36f348682aac5fc488c6d39dd58f8c27b1ec6/l5kit/setup.py#L23) and [torchvision](https://github.com/lyft/l5kit/blob/59f36f348682aac5fc488c6d39dd58f8c27b1ec6/l5kit/setup.py#L24) from ```setup.py```)
- install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### 4. Generate L5Kit code html documentation (optional)
```shell
sphinx-apidoc --module-first --separate -o API/ l5kit/l5kit l5kit/l5kit/tests*
sphinx-build . docs
```

### 5. Run example
```shell
jupyter notebook examples/visualisation/visualise_data.ipynb
```
