import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from experiments import ExperimentArray

from relucent import Complex

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

start = time.time()

train_models = True
exp_dir = "exp_0/"
log_dir = "results/shi_counts/"

os.makedirs(log_dir, exist_ok=True)
print("Logging Results To:", log_dir)

e = ExperimentArray.load("experiments/" + exp_dir)
print("Loaded Experiment Array From:", exp_dir)

if len(sys.argv) > 2:
    try:
        lower = int(sys.argv[1])
        upper = int(sys.argv[2])
        exp_range = range(lower, upper + 1)
        print(f"Running experiments {lower} (inclusive) to {upper} (exclusive).\n")
    except Exception:
        raise ValueError("Invalid Experiment Range Specified")
else:
    exp_range = range(len(e))
print("Range:", list(exp_range))

for i in exp_range:
    exp = e[f"blobs_mlp_{i}"]
    exp.get_model(device="cpu")
    print(f"\n ========== Current Runtime: {time.time() - start:.2f} seconds ==========")
    print(
        f"\n========== Running Experiment {i + 1} / {len(e)} (Dim {exp.dim}, Width {exp.width}, Hidden {exp.nhidden}) =========="
    )
    cplx = Complex(exp.model)
    search_info = cplx.bfs(nworkers=int(sys.argv[3]) if len(sys.argv) > 3 else None)
    result = {
        "Experiment": exp,
        "Accuracy": exp.test_accuracy(),
        "Layer Widths": exp.model.widths,
        "Trial": exp.trial,
        "Model": str(exp.model),
    } | search_info

    time2 = time.time()
    result["# Regions"] = len(cplx)
    print(result)
    result["Facet Counts"] = [len(p.shis) for p in cplx.index2poly]
    result["Wl2s"] = [p.Wl2 for p in cplx.index2poly]
    result["Interior Point Norm"] = [p.interior_point_norm for p in cplx.index2poly]
    result["Inradii"] = [p.inradius for p in cplx.index2poly]
    result["Volumes"] = [p.volume for p in cplx.index2poly]

    result["Dual Graph"] = cplx.get_dual_graph(relabel=True)
    result["Avg # Facets"] = np.mean(result["Facet Counts"])
    result["Trained"] = train_models

    print(f"\n{time.time() - time2:.2f} Seconds to Calculate Properties")

    with open(os.path.join(log_dir, f"shi_count_results_{i}.pkl"), "wb") as f:
        pickle.dump(result, f)

print("\n\n ========= Experiments Complete =========\n\n")

end = time.time()
print(f"Time Elapsed: {(end - start) // 3600} Hours, {(end - start) // 60} Minutes, {(end - start) % 60:.2f} Seconds")
