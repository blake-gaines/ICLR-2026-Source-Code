import os
import pickle
from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F
from iclr_datasets import get_dataset
from iclr_models import get_model
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from relucent import convert, split_sequential

default_args = {
    "epochs": 10,
    "batch_size": 64,
    "test_batch_size": 1000,
    "lr": 0.01,
    "gamma": 0.9,
    "weight_decay": 0,
}


class Experiment:
    def __init__(
        self,
        dataset_name,
        model_path=None,
        train_kwargs={},
        dataset_kwargs={},
        split_layer=None,
        model=None,
        embedder=None,
        **other_kwargs,
    ):
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.train_kwargs = train_kwargs
        self.dataset_kwargs = dataset_kwargs
        # if not model and not model_path:
        #     raise ValueError("Must provide either model or model_path")
        self._model = model
        self._embedder = embedder
        self.model_path = model_path
        self.split_layer = split_layer
        self.train_loss = None
        self.test_loss = None

        for k, v in other_kwargs.items():
            setattr(self, k, v)

    @cached_property
    def train_dataset(self):
        return get_dataset(self.dataset_name, train=True, **self.dataset_kwargs)

    @cached_property
    def test_dataset(self):
        return get_dataset(self.dataset_name, train=False, **self.dataset_kwargs)

    @property
    def model(self):
        if self._model is None:
            self.get_model(device="cpu")
        return self._model

    @property
    def embedder(self):
        if self._embedder is None:
            self.get_model(device="cpu")
        return self._embedder

    def get_model(self, split_layer=None, device="cpu", root="."):
        if self.model_path == "alexnet":
            self._model = get_model("alexnet", device=device)
        elif self.model_path is not None:
            self._model = torch.load(os.path.join(root, self.model_path), weights_only=False)
        if self._model is None and self.model_path is None:
            raise ValueError("Must provide either model or model_path")
        self._model.to(device)
        if split_layer is not None:
            self.split_layer = split_layer
            self._embedder, self._model = split_sequential(self._model, split_layer)
            self._embedder.eval()
        self._model.eval()
        if split_layer is not None:
            return self._embedder, self._model
        return self._model

    @cached_property
    def converted_model(self):
        return convert(self.model)

    def train_model(self):
        self.train_loss, self.test_loss = train_exp_model(
            self.model, self.train_dataset, self.test_dataset, **self.train_kwargs
        )

    def save_model(self, filename):
        self.model_path = filename
        self.model.to("cpu")
        torch.save(self.model, filename)

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))

    def load(filename):
        return pickle.load(open(filename, "rb"))

    def train_accuracy(self):
        return self.train_dataset.get_accuracy(self.model)

    def test_accuracy(self):
        return self.test_dataset.get_accuracy(self.model)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __reduce__(self):
        # if self.model_path is None:
        #     raise ValueError("Cannot pickle Experiment without model_path")
        if self._model is not None:
            self.__dict__["_model"] = None
        if self._embedder is not None:
            self.__dict__["_embedder"] = None
        # Remove cached properties to avoid pickling large datasets/models
        # These can be regenerated from dataset_name and dataset_kwargs
        for cached_prop in ["train_dataset", "test_dataset", "converted_model"]:
            if cached_prop in self.__dict__:
                del self.__dict__[cached_prop]
        return (
            Experiment,
            (
                self.dataset_name,
                self.model_path,
                self.train_kwargs,
                self.dataset_kwargs,
                self.split_layer,
            ),
            self.__dict__,
        )


class ExperimentArray:
    ## TODO: Phase out
    def __init__(self, exps=None, keys=None):
        self.exps = exps if exps is not None else dict()
        self.keys = keys if keys is not None else []

    def add_exp(self, exp, key):
        self.keys.append(key)
        self.exps[key] = exp

    def save_models(self, filename):
        os.makedirs(filename, exist_ok=True)
        for key, exp in self.exps.items():
            exp.save_model(os.path.join(filename, f"{key}.pt"))

    def save_exps(self, filename):
        os.makedirs(filename, exist_ok=True)
        for key, exp in self.exps.items():
            exp.save(os.path.join(filename, f"{key}.pkl"))

    def train_models(self, verbose=0):
        for key, exp in (pbar := tqdm(self.exps.items(), total=len(self))):
            exp.train_model()
            if verbose > 1:
                pbar.set_postfix_str(
                    f"Train Loss: {exp.train_loss:.2E} Test Loss: {exp.test_loss:.2E} Train Accuracy: {exp.train_accuracy():.2f} Test Accuracy: {exp.test_accuracy():.2f}"
                )
            elif verbose > 0:
                pbar.set_postfix_str(f"Train Loss: {exp.train_loss:.2E} Test Loss: {exp.test_loss:.2E}")

    def save(self, filename):
        self.save_exps(filename)

    def load(filename):
        exps = dict()
        keys = []
        for f in os.listdir(filename):
            if f.endswith(".pkl"):
                exp = Experiment.load(os.path.join(filename, f))
                exps[f[:-4]] = exp
                keys.append(f[:-4])
        return ExperimentArray(exps=exps, keys=keys)

    def __len__(self):
        return len(self.exps)

    def __iter__(self):
        return iter(self.exps.values())

    def __getitem__(self, key):
        return self.exps[key]

    def items(self):
        return self.exps.items()


def train(model, train_loader, criterion, optimizer, verbose=0):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, total=len(train_loader), disable=verbose < 2, leave=False, desc="Batches")
    for data, target in pbar:
        data, target = data.to(model.device, model.dtype), target.to(model.device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target.squeeze().float() if isinstance(criterion, nn.MSELoss) else target)
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
        optimizer.step()

    return train_loss / len(train_loader.dataset)


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model.device), target.to(model.device)
            output = model(data)
            test_loss += (
                criterion(output.squeeze(), target.squeeze().float() if isinstance(criterion, nn.MSELoss) else target)
                .sum()
                .item()
            )
    return test_loss / len(test_loader.dataset)


def train_exp_model(model, train_dataset, test_dataset, verbose=0, use_scheduler=False, **kwargs):
    args = default_args | kwargs  ## TODO: Just put default_args as default parameters

    train_loader = torch.utils.data.DataLoader(train_dataset, args["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, args["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = F.cross_entropy if train_dataset.task == "classification" else F.mse_loss

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args["lr"],
        weight_decay=args["weight_decay"],
    )

    scheduler = StepLR(optimizer, step_size=1, gamma=args["gamma"]) if use_scheduler else None

    pbar = tqdm(range(1, args["epochs"] + 1), desc="Training", leave=(verbose > 0), disable=(verbose == 0))
    for epoch in pbar:
        train_loss = train(model, train_loader, criterion, optimizer, verbose=verbose)
        test_loss = test(model, test_loader, criterion)
        if scheduler is not None:
            scheduler.step()
        pbar.set_postfix_str(
            f"Train Loss: {train_loss:.2E} Test Loss: {test_loss:.2E}"
            + (f"LR: {scheduler.get_last_lr()[0]:.2E}" if use_scheduler else "")
        )
    return train_loss, test_loss
