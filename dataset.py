# Imports
import numpy as np
import torch
import torch.nn
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, data, name=None, task="classification"):
        self.data = data
        self.y_shape = self[0][1].shape
        self.task = task
        self.name = name

    def __getitem__(self, index):
        x, y = self.data[index]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device)
        return x.to(self.device, self.dtype), y.to(self.device)

    def __len__(self):
        return len(self.data)

    @property
    def device(self):
        return self.data[0][0].device

    @property
    def dtype(self):
        return self.data[0][0].dtype

    @property
    def X(self):
        return torch.stack([x for x, y in self.data])

    @property
    def y(self):
        return np.stack([y.item() if isinstance(y, torch.Tensor) else y for x, y in self.data]).flatten()

    def filter_y(self, ys):
        ys = set(ys) if isinstance(ys, (list, tuple)) else {ys}
        for x, y in self.data:
            if int(y) in ys:
                yield x, y

    def get_predictions(self, model):
        return [model(torch.unsqueeze(x.to(model.device, model.dtype), 0)).argmax().item() for x, y in self.data]

    def get_accuracy(self, model):
        if self.task != "classification":
            raise ValueError(f"Cannot test accuracy for task: {self.task}")
        return sum(
            [model(torch.unsqueeze(x.to(model.device, model.dtype), 0)).argmax().item() == int(y) for x, y in self.data]
        ) / len(self)

    def get_auc(self, model, multi_class="ovr", average="micro"):
        if self.task != "classification":
            raise ValueError(f"Cannot test auc for task: {self.task}")
        return roc_auc_score(
            self.y,
            np.stack(
                [
                    model(torch.unsqueeze(x.to(model.device, model.dtype), 0))
                    .squeeze()
                    .softmax(0)
                    .detach()
                    .cpu()
                    .numpy()
                    for x, y in self.data
                ]
            ),
            multi_class=multi_class,
            average=average,
        )

    def get_rsquared(self, model):
        return r2_score(
            self.y,
            [model(torch.unsqueeze(x.to(model.device, model.dtype), 0)).item() for x, y in self.data],
        )

    def get_mse(self, model):
        return mean_squared_error(
            self.y,
            [model(torch.unsqueeze(x.to(model.device, model.dtype), 0)).item() for x, y in self.data],
        )
