#!/bin/bash

# Runs CI/CD tests locally before committing
black --check src/ cloud/functions/
flake8 src/ cloud/functions/
pytest tests/

# If any of the checks fail, exit with a non-zero status
if [ $? -ne 0 ]; then
    echo "CI/CD checks failed"
    exit 1
fi