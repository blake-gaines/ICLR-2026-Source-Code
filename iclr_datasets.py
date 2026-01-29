import numpy as np
import torch
from dataset import Dataset
from sklearn.datasets import fetch_california_housing, make_blobs
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


def get_dataset(name, train=True, **kwargs):
    if name == "mnist":
        return get_mnist_dataset(train)
    elif name == "imagenette":
        return get_imagenette_dataset(train)
    elif name == "xor":
        return get_xor_dataset(train)
    elif name == "circle":
        return get_circle_dataset(train)
    elif name == "california_housing":
        return get_california_housing(train)
    elif name == "california_housing_reg":
        return get_california_housing_regression(train)
    elif name == "blobs":
        return get_blobs_dataset(train, **kwargs)
    elif name == "cifar10":
        return get_cifar10_dataset(train, **kwargs)
    elif name == "cifar10_ood":
        return get_cifar10_ood_dataset(train, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_mnist_dataset(train=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    ## TODO: Just call Dataset right on the dataset
    return Dataset(
        [(x, torch.tensor(y)) for x, y in datasets.MNIST("./data", train=train, download=True, transform=transform)],
        name="mnist",
    )


def get_imagenette_dataset(train=True):
    split = "train" if train else "val"
    from torchvision import models

    weights = models.AlexNet_Weights.DEFAULT
    transform = weights.transforms()
    return Dataset(
        datasets.Imagenette("data/imagenette", split=split, transform=transform, download=False),
        name="imagenette",
    )


def get_cifar10_dataset(train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return Dataset(
        [
            (x, torch.tensor(y))
            for x, y in datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
        ],
        name="CIFAR10",
    )


def get_cifar10_ood_dataset(train=True, holdout_classes=[5, 6]):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    if train:
        return Dataset(
            [(x, y) for x, y in dataset if int(y) not in holdout_classes],
            name="CIFAR10_OOD",
        )
    else:
        return Dataset(dataset, name="CIFAR10_OOD")


def get_xor_dataset(train=True):
    data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = torch.tensor([[-1.0, 1.0, 1.0, -1.0]]).T
    dataset = torch.utils.data.TensorDataset(data, labels)
    return Dataset(dataset, name="xor", task="regression")


def get_california_housing(train=True):
    housing = fetch_california_housing(as_frame=True)
    housing.target = housing.target - housing.target.mean()
    housing.target = housing.target / housing.target.std()
    housing.data = housing.data - housing.data.mean(axis=0)
    housing.data = housing.data / housing.data.std(axis=0)
    housing = housing.data.merge(housing.target, left_index=True, right_index=True)
    train_df, test_df = train_test_split(housing, test_size=0.2, random_state=1)
    return Dataset(
        [
            (
                torch.tensor(tuple(row[housing.columns[:-1]]), dtype=torch.float32),
                torch.tensor(
                    [(row[housing.columns[-1]] < np.mean(housing[housing.columns[-1]])) * 2 - 1], dtype=torch.float32
                ),
            )
            for _, row in (train_df if train else test_df).iterrows()
        ],
        name="california_housing",
        task="regression",
    )


def get_california_housing_regression(train=True):
    housing = fetch_california_housing(as_frame=True)
    housing.target = housing.target - housing.target.mean()
    housing.target = housing.target / housing.target.std()
    housing.data = housing.data - housing.data.mean(axis=0)
    housing.data = housing.data / housing.data.std(axis=0)
    housing = housing.data.merge(housing.target, left_index=True, right_index=True)
    train_df, test_df = train_test_split(housing, test_size=0.2, random_state=1)
    print("Target Column:", housing.columns[-1])
    return Dataset(
        [
            (
                torch.tensor(tuple(row[housing.columns[:-1]]), dtype=torch.float32),
                torch.tensor(row[housing.columns[-1]], dtype=torch.float32),
            )
            for _, row in (train_df if train else test_df).iterrows()
        ],
        name="california_housing",
        task="regression",
    )


def get_circle_dataset(train=True, r1=0.5):
    ### Taken from https://doi.org/10.3389/frai.2023.1255192
    ### https://github.com/bsattelb/local-linearity-of-relu-neural-networks/tree/master
    r1sqr = r1**2
    r2sqr = 2 * r1sqr
    r3sqr = 3 * r1sqr

    train_x = []
    train_y = []

    test_x = []
    test_y = []

    while len(train_x) < 100000:
        sample = 2 * np.random.uniform(size=(1, 2)) - 1
        if np.sum(np.square(sample)) < r1sqr:
            train_x.append(sample)
            # train_y.append(np.array([[0, 1]]))
            train_y.append(np.array([-1.0]))
        elif r2sqr < np.sum(np.square(sample)) < r3sqr:
            train_x.append(sample)
            # train_y.append(np.array([[1, 0]]))
            train_y.append(np.array([1.0]))

    test_data = []

    for i in range(101):
        for j in range(101):
            test_data.append([2 * i / 100 - 1, 2 * j / 100 - 1])

    test_data = np.array(test_data)

    for elem in test_data:
        if np.sum(np.square(elem)) < r1sqr:
            test_x.append(elem)
            # test_y.append(np.array([[0, 1]]))
            test_y.append(np.array([-1.0]))
        elif r2sqr < np.sum(np.square(elem)) < r3sqr:
            test_x.append(elem)
            # test_y.append(np.array([[1, 0]]))
            test_y.append(np.array([1.0]))

    if train:
        tensor_x = torch.stack([torch.tensor(elem.astype(np.float32)) for elem in train_x]).squeeze()
        tensor_y = torch.stack([torch.tensor(elem.astype(np.float32)) for elem in train_y])
    else:
        tensor_x = torch.stack([torch.tensor(elem.astype(np.float32)) for elem in test_x]).squeeze()
        tensor_y = torch.stack([torch.tensor(elem.astype(np.float32)) for elem in test_y])

    return Dataset(torch.utils.data.TensorDataset(tensor_x, tensor_y), name="circle", task="regression")


def get_blobs_dataset(train=True, d=2, n=1000, centers=2, random_state=1, cluster_std=0.5):
    X, y = make_blobs(n_samples=n, n_features=d, centers=centers, random_state=random_state, cluster_std=cluster_std)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)
    dataset = torch.utils.data.TensorDataset(X, y)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=1)
    return Dataset(train_dataset if train else test_dataset, name="blobs", task="classification")
