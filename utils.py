"""Small helper functions: seeding, run naming, optimizer factory."""
import os
import random

import numpy as np
import torch
import torch.optim as optim


def set_seed(seed):
    """Fix all random number generators so results can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # use only repeatable GPU algorithms
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Use the GPU if there is one, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def build_optimizer(name, params, lr, momentum, weight_decay):
    """Create one of the 6 optimizers used in the assignment."""
    name = name.lower()

    if name == "sgd":
        # plain SGD: no momentum
        return optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    elif name == "nesterov":
        # SGD with Nesterov momentum
        return optim.SGD(params, lr=lr, momentum=momentum, nesterov=True,
                         weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer: " + name)


def make_run_name(cfg):
    """Readable run name, e.g. act_relu_bs_128_opt_adam_lr_0.001_ep_20_wd_0.0005_bn_True"""
    activation = cfg["activation"]
    batch_size = cfg["batch_size"]
    optimizer = cfg["optimizer"]
    lr = cfg["lr"]
    epochs = cfg["epochs"]
    weight_decay = cfg["weight_decay"]
    batch_norm = cfg["batch_norm"]
    return ("act_" + str(activation) + "_bs_" + str(batch_size) + "_opt_" + str(optimizer)
            + "_lr_" + str(lr) + "_ep_" + str(epochs) + "_wd_" + str(weight_decay)
            + "_bn_" + str(batch_norm))

