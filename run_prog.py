import numpy as np
import pandas as pd
import torch
from experiments import ExperimentArray
from relucent import Polyhedron, Complex
from tqdm.auto import tqdm

torch.random.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

experiments = {
    "MNIST": "experiments/mnist_fc_Progress",
    # "CH Reg": "experiments/california_housing_reg_Progress",
    # "CIFAR10": "experiments/cifar10_cnn_Progress",
}

## For MNIST and CIFAR10, we split the networks into a feature extractor and classifier and analyze the cplx of the classifier
split_layers = {
    "CIFAR10": "relu3",
    "MNIST": "relu0",
}

final_df = pd.DataFrame()

for name, path in experiments.items():
    print("\n ======= Running Experiment:", name, "======= \n")
    exps = ExperimentArray.load(path)

    for epoch, e in tqdm(exps.items(), total=len(exps), desc="Running Progress Experiments"):
        train_dataset = e.train_dataset

        device = "cpu"

        if np.prod(e.model.input_shape) > 10:
            embedder, model = e.get_model(split_layer=split_layers[name], device=device)
        else:
            embedder, model = None, e.get_model(device=device)

        cplx = Complex(model)

        bfs_limit = 1000000
        point_limit = 10000
        cplx.bfs(
            max_polys=bfs_limit,
            start=torch.ones(e.model.input_shape, device=device) * 2,
            get_volumes=False,
        )
        num_bfs_polys = len(cplx)

        data_points = [x[0].cpu() for i, x in enumerate(train_dataset) if i < point_limit]
        ys = [x[1].item() for i, x in enumerate(train_dataset) if i < point_limit]
        nunique = len(set(ys))
        if nunique == 2 and min(ys) == -1:
            ys = [0 if y == -1 else 1 for y in ys]

        points_df = pd.DataFrame()
        points_df["Label"] = ys
        points, logits = [], []
        for i, p in enumerate(data_points):
            try:
                point = p.to(device)
                if embedder is not None:
                    point = embedder(point)
                logits.append(model(point).detach().cpu().numpy())
                points.append(point.cpu())
            except Exception as e:
                print(i, e)
                logits.append(None)

        points_df["Logits"] = logits
        points_df = points_df.dropna()

        if nunique == 2:
            points_df["Prediction"] = points_df["Logits"].apply(lambda x: x[0][0] > 0)
        else:
            points_df["Prediction"] = [np.argmax(x) for x in points_df["Logits"]]
        points_df["Correct"] = points_df["Label"] == points_df["Prediction"]

        point_polys = cplx.parallel_add(points, nworkers=32)
        points_df["p"] = point_polys
        print(len(cplx))

        df = pd.DataFrame({"p": cplx.index2poly, "index": range(len(cplx.index2poly))}).set_index("index")

        is_poly = df["p"].map(lambda x: isinstance(x, Polyhedron))
        print("Number of Mistakes:", len(is_poly) - is_poly.sum())
        df = df[is_poly]

        df["Volume"] = df["p"].map(
            lambda x: x._volume if x._volume is not None and x._volume < float("inf") and x._volume > 0 else np.nan
        )
        df["Log Volume"] = np.log(df["Volume"])
        df["Finite"] = (
            df["Volume"].map(lambda x: x is not None and x < float("inf"))
            if any(df["Volume"].notna())
            else df["p"].map(lambda x: x._finite)
        )
        df["# Faces"] = df["p"].map(lambda x: len(x._shis) if x._shis is not None else np.nan)
        columns = df.columns

        poly_df = df.copy().loc[:num_bfs_polys]

        points_df["Label"] = points_df["Label"].astype(str)

        df = (
            df.join(
                points_df.groupby(["p"], sort=False).size().rename("# Points"),
                on="p",
                how="right",
            )
            .infer_objects(copy=False)
            .fillna(0)
        )

        df = df[df["# Points"] > 0]
        df["Category"] = "Data Points"

        poly_df["# Points"] = 0
        poly_df["Label"] = np.nan
        poly_df["Category"] = "BFS"

        df = pd.concat([df, poly_df], sort=False)
        df["Volume"] = df["Volume"].map(lambda x: x if x > 0 else np.nan)
        df["Inradius"] = df["p"].map(
            lambda x: x._inradius if hasattr(x, "_inradius") and x._inradius is not None else np.nan
        )
        df["Epoch"] = epoch

        final_df = pd.concat([final_df, df], sort=False)

        del df

final_df.to_pickle(f"results/data_polys_{name}_Progress.pkl")
