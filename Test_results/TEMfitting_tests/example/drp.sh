#!/bin/bash

# Exit on error
set -e

# Define the Conda environment name and the Python script path
CONDA_ENV_NAME="drprobe"
PYTHON_SCRIPT_PATH="Test_results/TEMfitting_tests/example/DrProbe.py"

# Check if both arguments are provided
if [ -z "$CONDA_ENV_NAME" ] || [ -z "$PYTHON_SCRIPT_PATH" ]; then
  echo "Usage: $0 <conda_env_name> <python_script_path>"
  exit 1
fi

# Activate the Conda environment
# Use 'source' if running in a bash shell
if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
else
  eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV_NAME"

# Run the Python script
python "$PYTHON_SCRIPT_PATH"

