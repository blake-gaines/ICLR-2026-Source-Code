import os
import random
from copy import deepcopy
from itertools import product

import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from experiments import Experiment, ExperimentArray
from iclr_models import get_model


def train_real_exps():
    print("\n\n ========= Training Real Experiments =========\n\n")

    exps = [
        ("mnist", "mnist_fc", {"epochs": 50, "lr": 0.1, "use_scheduler": True}),
        ("california_housing_reg", "california_housing_reg", {"epochs": 60, "lr": 0.001}),
        ("cifar10", "cifar10_cnn", {"epochs": 30, "lr": 0.01, "use_scheduler": True, "verbose": 2, "batch_size": 4}),
    ]
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for dataset_name, model_name, train_kwargs in exps:
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        print(f"Running experiment - Dataset: {dataset_name}, Model: {model_name}")

        model = get_model(model_name)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        train_kwargs["verbose"] = train_kwargs.get("verbose", 1)
        exp = Experiment(dataset_name, model=model, train_kwargs=train_kwargs)

        print("Before Training:")
        if exp.test_dataset.task == "classification":
            print(f"  Train Accuracy: {exp.train_accuracy()}")
            print(f"  Test Accuracy: {exp.test_accuracy()}")
        else:
            print(f"  Test MSE: {exp.test_dataset.get_mse(exp.model)}")
            print(f"  Test R2: {exp.test_dataset.get_rsquared(exp.model)}")

        exp.train_model()

        print("After Training:")
        if exp.test_dataset.task == "classification":
            print(f"  Train Accuracy: {exp.train_accuracy()}")
            print(f"  Test Accuracy: {exp.test_accuracy()}")
            print("\nTrain Classification Report")
            print(
                classification_report(
                    exp.train_dataset.y, exp.train_dataset.get_predictions(exp.model), zero_division=np.nan
                )
            )
            print("\nTest Classification Report")
            print(
                classification_report(
                    exp.test_dataset.y, exp.test_dataset.get_predictions(exp.model), zero_division=np.nan
                )
            )
        else:
            print(f"  Test MSE: {exp.test_dataset.get_mse(exp.model)}")
            print(f"  Test R2: {exp.test_dataset.get_rsquared(exp.model)}")

        exp.save_model(f"models/{model_name}.pt")

        exp.save(f"experiments/{model_name}.pkl")

    Experiment("imagenette", "alexnet", {}).save("experiments/alexnet.pkl")


def train_synthetic_exps():
    print("\n\n ========= Training Synthetic Experiments =========\n\n")

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    ncenters = 3
    ntrials = 5
    base_widths = [4, 8, 16]
    nhiddens = range(1, 5)
    ds = range(2, 6)
    train_models = True
    exp_dir = "exp_0/"

    e = ExperimentArray()

    for i, (d, base_width, nhidden, trial) in enumerate(product(ds, base_widths, nhiddens, range(ntrials))):
        widths = [base_width] * nhidden
        e.add_exp(
            Experiment(
                dataset_name="blobs",
                dataset_kwargs={"d": d, "n": 100 * ncenters, "centers": ncenters, "random_state": trial},
                train_kwargs={"epochs": 20, "lr": 0.01},
                model=get_model("mlp", widths=[d] + widths + [ncenters]),
                width=base_width,
                nhidden=nhidden,
                trial=trial,
                dim=d,
            ),
            key=f"blobs_mlp_{i}",
        )

    if train_models:
        e.train_models()

    for exp in e:
        print(f"{exp} Accuracy: ", exp.test_accuracy())

    if exp_dir:
        os.makedirs("experiments/" + exp_dir, exist_ok=True)
        e.save_models("experiments/" + exp_dir)
        e.save("experiments/" + exp_dir)


def train_progress_exps(n_saves=20):
    print("\n\n ========= Training Progress Experiments =========\n\n")

    exps = [
        ("mnist", "mnist_fc", {"epochs": 1, "lr": 0.01, "use_scheduler": False, "verbose": 0}),
        ("california_housing_reg", "california_housing_reg", {"epochs": 1, "lr": 0.001, "verbose": 0}),
        ("cifar10", "cifar10_cnn", {"epochs": 1, "lr": 0.01, "use_scheduler": False, "verbose": 0, "batch_size": 4}),
    ]

    for dataset_name, model_name, train_kwargs in exps:
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)

        print(f"Running experiment - Dataset: {dataset_name}, Model: {model_name}")

        os.makedirs(f"models/{model_name}_Progress", exist_ok=True)

        model = get_model(model_name)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        train_kwargs["verbose"] = train_kwargs.get("verbose", 1)

        exp_array = ExperimentArray()
        exp = Experiment(dataset_name, model=deepcopy(model), train_kwargs=train_kwargs)
        exp_array.add_exp(exp, key=0)

        for i in tqdm(range(n_saves)):
            exp = Experiment(dataset_name, model=deepcopy(exp.model), train_kwargs=train_kwargs)

            exp.train_model()

            # exp.save_model(f"models/{model_name}_Progress/{i * train_kwargs['epochs']}.pt")
            exp_array.add_exp(exp, key=(i + 1) * train_kwargs["epochs"])

        if exp.test_dataset.task == "classification":
            print(f"  Train Accuracy: {exp.train_accuracy()}")
            print(f"  Test Accuracy: {exp.test_accuracy()}")
            print("\nTrain Classification Report")
            print(
                classification_report(
                    exp.train_dataset.y, exp.train_dataset.get_predictions(exp.model), zero_division=np.nan
                )
            )
            print("\nTest Classification Report")
            print(
                classification_report(
                    exp.test_dataset.y, exp.test_dataset.get_predictions(exp.model), zero_division=np.nan
                )
            )
        else:
            print(f"  Test MSE: {exp.test_dataset.get_mse(exp.model)}")
            print(f"  Test R2: {exp.test_dataset.get_rsquared(exp.model)}")

        exp_array.save_models(f"experiments/{model_name}_Progress3/")
        exp_array.save(f"experiments/{model_name}_Progress3")


if __name__ == "__main__":
    os.makedirs("experiments", exist_ok=True)
    train_synthetic_exps()
    train_real_exps()
    train_progress_exps()
