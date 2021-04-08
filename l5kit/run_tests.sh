#!/bin/bash

# Exit on error. Append "|| true" if you expect an error.
set -o errexit
# Exit on error inside any functions or subshells.
set -o errtrace
# Do not allow use of undefined vars. Use ${VAR:-} to use an undefined VAR
set -o nounset

TEST_TYPE=${1:-"all"}

PYTHON_EXECUTABLE=$(which python3)
echo "Using python interpreter: ${PYTHON_EXECUTABLE}"

lint() {
    echo "linting.."
    ${PYTHON_EXECUTABLE} -m flake8 --config .flake8 --show-source --statistics --jobs auto l5kit
}

isort() {
  echo "import sorting.."
  ${PYTHON_EXECUTABLE} -m isort l5kit --check-only
}

types() {
    echo "Type checking.."
    ${PYTHON_EXECUTABLE} -m mypy --ignore-missing-imports l5kit
}

tests() {
    echo "Testing.."
    ${PYTHON_EXECUTABLE} -m pytest --cache-clear --cov l5kit
}

if [ "${TEST_TYPE}" = "lint" ] ; then
  lint
elif [ "${TEST_TYPE}" = "isort" ] ; then
  isort
elif [ "${TEST_TYPE}" = "types" ] ; then
  types
elif [ "${TEST_TYPE}" = "tests" ] ; then
  tests
elif [ "${TEST_TYPE}" = "all" ] ; then
  lint
  isort
  types
  tests
else
  echo "invalid test type - ${TEST_TYPE}"
  echo
  echo "test type must be one of (lint|types|tests|all)."
  exit 1
fi

echo "All good."
