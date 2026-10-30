"""Runs all controlled experiments in one go.

Idea: start from ONE baseline and change ONE thing at a time, so any change in
accuracy can be blamed on that one thing.

    python run_experiments.py                       # everything
    python run_experiments.py --groups activation   # only some groups
"""
import argparse

from train import run

# The baseline settings. Every experiment starts from these.
BASE = {
    "activation": "relu",
    "optimizer": "adam",
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 50,
    "batch_norm": True,
}

# The 5 activation functions
ACTIVATIONS = ["relu", "sigmoid", "tanh", "silu", "gelu"]

# The 6 optimizers, each with a learning rate
OPTIMIZERS = [
    ("sgd", 0.05),
    ("nesterov", 0.05),
    ("adam", 0.001),
    ("adagrad", 0.01),
    ("rmsprop", 0.0005),
    ("nadam", 0.001),
]

# ---- Group 1: vary only the activation (optimizer stays adam) ----
activation_runs = []
for activation in ACTIVATIONS:
    activation_runs.append({"activation": activation})

# ---- Group 2: vary only the optimizer (activation stays relu) ----
optimizer_runs = []
for optimizer, lr in OPTIMIZERS:
    optimizer_runs.append({"optimizer": optimizer, "lr": lr})

# ---- Group 3 (grid): EVERY optimizer with EVERY activation = 5 x 6 = 30 runs ----
grid_runs = []
for activation in ACTIVATIONS:
    for optimizer, lr in OPTIMIZERS:
        grid_runs.append({"activation": activation, "optimizer": optimizer, "lr": lr})

# ---- Group 4: vary only the batch size ----
batch_size_runs = [
    {"batch_size": 32},
    {"batch_size": 64},
    {"batch_size": 128},
    {"batch_size": 256},
    {"batch_size": 512},
]

# ---- Group 5: vary only the number of epochs ----
epochs_runs = [
    {"epochs": 30},
    {"epochs": 50},
    {"epochs": 60},
    {"epochs": 70},
]

# ---- Group 6: vary only the learning rate ----
lr_runs = [
    {"lr": 0.0001},
    {"lr": 0.0005},
    {"lr": 0.001},
    {"lr": 0.005},
    {"lr": 0.01},
]

# ---- Group 7: what happens without BatchNorm? ----
batch_norm_runs = [
    {"batch_norm": False},
]

GROUPS = {
    "activation": activation_runs,
    "optimizer": optimizer_runs,
    "grid": grid_runs,
    "batch_size": batch_size_runs,
    "epochs": epochs_runs,
    "lr": lr_runs,
    "batch_norm": batch_norm_runs,
}

GROUP_NAMES = ["activation", "optimizer", "grid", "batch_size", "epochs", "lr", "batch_norm"]


def get_val_acc(result):
    return result["best_val_acc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=GROUP_NAMES, choices=GROUP_NAMES)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    already_run = []  # settings that have been run already (to skip repeats)
    results = []

    for group_name in args.groups:
        for changes in GROUPS[group_name]:
            # Baseline settings + the changes of this experiment
            settings = BASE.copy()
            settings.update(changes)

            # Some settings appear in several groups; run each only once
            if settings in already_run:
                continue
            already_run.append(settings)

            result = run(settings, use_wandb=not args.no_wandb)
            results.append(result)

    # Print a summary, best validation accuracy first
    results.sort(key=get_val_acc, reverse=True)
    print("\n===== SUMMARY =====")
    for result in results:
        print(f"{result['best_val_acc']:6.2f}% val | {result['test_acc']:6.2f}% test | {result['name']}")


if __name__ == "__main__":
    main()
