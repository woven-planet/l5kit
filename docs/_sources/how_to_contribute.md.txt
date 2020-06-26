How to contribute
==

You are invited to contribute to the L5Kit with your examples and improvements.
These are peer-reviewed by the development team to maintain quality and reproducibility.

**All commands below must be run from the `l5kit` folder (the one with the `setup.py`).**

## License
You will be required to sign a contributor agreement license upon a pull request (PR).


## Installing l5kit as a developer
Run:
```shell
pip install -r requirements.txt 
```
to install all developer dependencies along with l5kit in editable (-e) mode.

## Code Sanity checks
Before starting a PR, it is highly recommended to also install the git pre-commit hooks, run:
```shell
pre-commit install 
```
This will run all required code checks before each commit and it ensures your builds 
won't fail in CI.

If, on the other hand, you want to run individual checks, please refer to the instructions below.

## Testing
To run the tests, run:

```shell
./run_tests.sh tests
```

## Coverage report

To generate a test coverage report, run:

```shell
# Outputs to std out
pytest --cov

# Output to HTML files in the coverage_report_html folder
pytest --cov --cov-report html
```

## Type checking

To run type checking with mypy, run:

```shell
./run_tests.sh types
```

## Code style, linting and formatting

We use [Black](https://black.readthedocs.io/en/stable/) for automatic formatting, [isort](https://github.com/timothycrosley/isort) for import sorting, and [flake8](https://flake8.pycqa.org/en/latest/) to check for linting errors. These are the relevant commands:

You can check against those by running:

```shell
./run_tests.sh lint

```

Or, if you want to apply those formatters:

```shell
# Sort imports automatically.
isort l5kit --apply --recursive

# Apply formatting.
black l5kit

# Check for linting errors that black wasn't able to fix.
flake8 l5kit

```

Our docstrings are in [Google docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) format.

## Creating a distribution

```shell
# Clean up from potential earlier runs
rm -rf dist
rm -rf build

python3 setup.py sdist bdist_wheel
```

You can now find the distribution files (both tar.gz and wheel) in the `dist` folder.

**We look forward to your contributions!**
