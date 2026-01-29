Code used in the experiments of the ICLR 2026 ["Characterizing the Combinatorial Complexity of ReLU Networks"](https://openreview.net/forum?id=TgLW2DiRDG).

## Environment Setup 
1. Install Python 3.13
2. Install [PyTorch 2.9 and TorchVision 0.24](https://pytorch.org/get-started/previous-versions/#:~:text=v2%2E9%2E0)
3. Install the remaining dependencies with `pip install -r requirements.txt`

## Replication of Experimental Results
Once your environment is ready, `bash run_everything.sh` will automatically run all of the experiments from the paper. This includes downloading the datasets, training the models, calculating properties of their complexes, and generating all of the figures. [run_everything.sh](run_everything.sh) also includes a line-by-line breakdown of each step, in case you would rather split them up or make any changes. With <number> <model> processors and a <model> GPU, this script runs from start to finish in about <X> hours.

## Parallelization
The `python run_synthetic.py` script may take a long time when run on a single machine. However, the experiments easily can be parallelized across multiple nodes. To facilitate this, the script optionally takes two command line arguments that are passed to the "range()" function to select a subset of indices of experiments (0-239 by default) to run.  For example, `python run_synthetic.py 50 55` will run experiments 50, 51, 52, 53, and 54 sequentially. There is no interdependence between the runs. By default they will use however many CPUs are detected on the machine, but an optional third command line argument restricts the number of CPUs used by the script. For example, `python run_synthetic.py 0 240 32` will run all of the experiments from the paper sequentially using only 32 CPUs. These arguments simplify the process of submitting batch jobs on a cluster to run experiments in parallel. Note that some Gurobi licenses limit the number of concurrent sessions.

## Relucent
You might notice that this code is missing a lot of the math we use in our paper. You can find the rest of it in our open-source package [Relucent](https://pypi.org/project/relucent/). Check it out if you're interested in ReLU network geometry!