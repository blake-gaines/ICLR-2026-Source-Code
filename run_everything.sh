#!/bin/bash
set -e

## Set up environment
# # # conda/mamba/micromamba create -n poly -c conda-forge python=3.13 -y
# # # conda/mamba/micromamba activate poly

## Ensure Python is on path
which python
if [ $? -ne 0 ]; then
    echo "Python must be on path"
    exit 1
fi

## Ensure python version is at least 3.13
python --version
if [ $(python --version | cut -d' ' -f2 | cut -d'.' -f1) -lt 3 ] || [ $(python --version | cut -d' ' -f2 | cut -d'.' -f2) -lt 13 ]; then
    echo "Python version must be at least 3.13"
    exit 1
fi         

## Ensure jupyter is installed
which jupyter
if [ $? -ne 0 ]; then
    echo "Jupyter must be on path"
    exit 1
fi

# ## Install dependencies
# pip install --upgrade pip
# python -m pip install -r requirements.txt

## Download datasets, train models, and set up experiments
python setup_experiments.py

## Run experiments on the networks trained on synthetic data
python run_synthetic.py

## Compute diameter bounds for the corresponding polyhedral complexes
python diameter_bounds.py

## Run experiments on the networks trained with real data
python run_real.py

## Run the notebooks in the visualization folder to generate the plots from the paper
for notebook in notebooks/*.ipynb; do
    python -m jupyter nbconvert --to notebook --execute $notebook --inplace --log-level=INFO
done