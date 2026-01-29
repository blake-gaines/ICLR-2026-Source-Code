from collections import OrderedDict

import torch
import torch.nn as nn

from relucent import NN, get_mlp_model


def get_children(module):
    children = module.children()
    if not hasattr(children, "__iter__"):
        return [module]
    children = tuple(children)
    if len(children) == 0:
        return [module]
    if len(children) == 1:
        return get_children(children[0])
    flat_children = []
    for child in children:
        flat_children.extend(get_children(child))
    return flat_children


## TODO: Move


def get_model(name, **kwargs):
    if name == "mnist_cnn":
        return NN(
            layers=OrderedDict(
                [
                    ("conv1", nn.Conv2d(1, 8, 3, 1)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(8, 8, 3, 1)),
                    ("relu2", nn.ReLU()),
                    ("pool1", nn.AvgPool2d(2)),
                    ("conv3", nn.Conv2d(8, 8, 3, 1)),
                    ("relu3", nn.ReLU()),
                    ("pool2", nn.AvgPool2d(2)),
                    ("dropout1", nn.Dropout(0.25)),
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(200, 150)),
                    ("relu4", nn.ReLU()),
                    ("fc2", nn.Linear(150, 150)),
                    ("relu5", nn.ReLU()),
                    ("fc3", nn.Linear(150, 32)),
                    ("relu6", nn.ReLU()),
                    ("dropout2", nn.Dropout(0.5)),
                    ("fc4", nn.Linear(32, 10)),
                ]
            ),
            input_shape=(1, 28, 28),
        )
    elif name == "mnist_fc":
        return get_mlp_model([784, 5, 8, 8, 8, 10])
    elif name == "cifar10_cnn" or name == "cifar10_cnn_ood":
        return NN(
            layers=OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 6, 5)),
                    ("pool1", nn.MaxPool2d(2, 2)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(6, 16, 5)),
                    ("pool2", nn.MaxPool2d(2, 2)),
                    ("relu2", nn.ReLU()),
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(16 * 5 * 5, 10)),
                    ("relu3", nn.ReLU()),
                    ("fc4", nn.Linear(10, 64)),
                    ("relu4", nn.ReLU()),
                    ("fC3", nn.Linear(64, 64)),
                    ("relu5", nn.ReLU()),
                    ("fc3", nn.Linear(64, 10)),
                ]
            ),
            input_shape=(3, 32, 32),
        )
    elif name == "cifar10_cnn_deeper" or name == "cifar10_cnn_ood_deeper":
        return NN(
            layers=OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 6, 5)),
                    ("pool1", nn.MaxPool2d(2, 2)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(6, 16, 5)),
                    ("pool2", nn.MaxPool2d(2, 2)),
                    ("relu2", nn.ReLU()),
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(16 * 5 * 5, 5)),
                    ("relu3", nn.ReLU()),
                    ("fc2", nn.Linear(5, 64)),
                    ("relu4", nn.ReLU()),
                    ("fc3", nn.Linear(64, 64)),
                    ("relu5", nn.ReLU()),
                    ("fc4", nn.Linear(64, 64)),
                    ("relu6", nn.ReLU()),
                    ("fc5", nn.Linear(64, 64)),
                    ("relu7", nn.ReLU()),
                    ("fc6", nn.Linear(64, 10)),
                ]
            ),
            input_shape=(3, 32, 32),
        )
    elif name == "cifar10_cnn_ood_deeper_full":
        return NN(
            layers=OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 6, 5)),
                    ("pool1", nn.MaxPool2d(2, 2)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(6, 16, 5)),
                    ("pool2", nn.MaxPool2d(2, 2)),
                    ("relu2", nn.ReLU()),
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(16 * 5 * 5, 16)),
                    ("relu3", nn.ReLU()),
                    ("fc2", nn.Linear(16, 32)),
                    ("relu4", nn.ReLU()),
                    ("fc3", nn.Linear(32, 32)),
                    ("relu5", nn.ReLU()),
                    ("fc4", nn.Linear(32, 32)),
                    ("relu6", nn.ReLU()),
                    ("fc5", nn.Linear(32, 10)),
                ]
            ),
            input_shape=(3, 32, 32),
        )
    elif name == "cifar10_fc":
        return get_mlp_model([3 * 32 * 32, 64, 128, 128, 128, 10])
    elif name == "xor":
        return get_mlp_model([2, 2, 1])
    elif name == "circle":
        return get_mlp_model([2, 8, 8, 8, 1])
    elif name == "circle_shallow":
        return get_mlp_model([2, 16, 1])
    elif name == "alexnet":
        model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", verbose=False)
        layers = OrderedDict([(str(i), layer) for i, layer in enumerate(get_children(model))])
        layers = OrderedDict()
        i = 0
        for layer in get_children(model):
            layers[str(i)] = layer
            i += 1
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                layers[str(i)] = nn.Flatten()
                i += 1

        return NN(layers=layers, input_shape=(3, 224, 224))
    elif name == "california_housing":
        return get_mlp_model([8, 128, 128, 64, 1])
    elif name == "california_housing_shallow":
        return get_mlp_model([8, 128, 1])
    elif name == "california_housing_shallow_small":
        return get_mlp_model([8, 9, 1])
    elif name == "california_housing_reg":
        return get_mlp_model([8, 128, 1])
    elif name == "mlp":
        return get_mlp_model(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
