# If you get an execution policy error: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

$ErrorActionPreference = "Stop"

## Set up environment
# conda/mamba create -n poly -c conda-forge python=3.13 -y
# conda/mamba activate poly

## Install PyTorch and TorchVision
# pip install torch torchvision

# Check that Python is on path
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python must be on path"
    exit 1
}

## 1. Ensure Python version is at least 3.13
$pythonVersionCheck = python -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)" 2>&1
if ($LASTEXITCODE -ne 0) {
    $versionOutput = python --version 2>&1
    Write-Error "Error: Python version must be at least 3.13`nFound: $versionOutput"
    exit 1
}

## 2. Ensure Torch is installed and >= 2.3.0
$torchCheck = python -c "import torch; from pathlib import Path; exit(0 if torch.__version__ >= '2.3.0' else 1)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: Torch version must be at least 2.3.0"
    exit 1
}

## Install dependencies
python -m pip install --upgrade pip
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
$notebooks = Get-ChildItem -Path "notebooks\*.ipynb" -ErrorAction SilentlyContinue
if ($notebooks) {
    foreach ($notebook in $notebooks) {
        python -m jupyter nbconvert --to notebook --execute $notebook.FullName --inplace --log-level=INFO
    }
} else {
    Write-Warning "No .ipynb files found in notebooks\"
}
