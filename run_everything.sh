#!/bin/bash
set -e

## Set up environment
# conda/mamba/micromamba create -n poly -c conda-forge python=3.13 -y
# conda/mamba/micromamba activate poly

## Install PyTorch and TorchVision
# pip install torch torchvision

# The 'if' block prevents set -e from exiting on failure
if ! command -v python >/dev/null 2>&1; then
    echo "Python must be on path"
    exit 1
fi

## 1. Ensure Python version is at least 3.13
if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)" >/dev/null 2>&1; then
    echo "Error: Python version must be at least 3.13"
    echo "Found: $(python --version)"
    exit 1
fi

## 2. Ensure Torch is installed and >= 2.3.0
if ! python -c "import torch; from pathlib import Path; exit(0 if torch.__version__ >= '2.3.0' else 1)" >/dev/null 2>&1; then
    echo "Error: Torch version must be at least 2.3.0"
    exit 1
fi

## Install dependencies
pip install --upgrade pip
python -m pip install -r requirements.txt

## Download datasets and train models and set up experiments
python setup_experiments.py

## Run experiments on the networks trained on synthetic data
python run_synthetic.py 0 235

## Compute diameter bounds for the corresponding polyhedral complexes
python diameter_bounds.py

## Run experiments on the networks trained with real data
python run_real.py

## Run experiments on networks evolving over the course of training
python run_prog.py

## Run the notebooks in the visualization folder to generate the plots from the paper
for notebook in notebooks/*.ipynb; do
    python -m jupyter nbconvert --to notebook --execute "$notebook" --inplace --log-level=INFO
done