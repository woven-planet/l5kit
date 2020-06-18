# avkit

**avkit** is a python package containing utilities for training machine learning-based planning and simulation systems.

## Installation
You can install the avkit package in development mode along with its dependencies using the following command:

```shell
pip install -r requirements.txt
```

Pypi distributions are not available yet.

## Testing
To run the tests, run:

```shell
pytest
```

### Coverage report

To generate a test coverage report, run:

```shell
# Outputs to std out
pytest --cov

# Output to HTML files in the coverage_report_html folder
pytest --cov --cov-report html
```

### Type checking

To run type checking with mypy, run:

```shell
mypy avkit
```

## Code style, linting and formatting

We use [Black](https://black.readthedocs.io/en/stable/) for automatic formatting, [isort](https://github.com/timothycrosley/isort) for import sorting, and [flake8](https://flake8.pycqa.org/en/latest/) to check for linting errors. These are the relevant commands:

```shell
# Sort imports automatically.
isort avkit --apply --recursive

# Apply formatting.
black avkit

# Check for linting errors that black wasn't able to fix.
flake8 avkit

```

Our docstrings are in [Google docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) format.

## Creating a distribution

```shell
# Clean up from potential earlier runs
rm -rf dist
rm -rf build

python3 setup.py sdist bdist_wheel
```

You can now find the distribution files in the `dist` folder.
