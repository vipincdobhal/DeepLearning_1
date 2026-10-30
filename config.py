# This is the configration file for parameters used in the project.

DEFAULTS = dict(
    activation="relu",      # relu | sigmoid | tanh | silu | gelu | leaky_relu | selu
    optimizer="adam",       # sgd | nesterov | adam | adagrad | rmsprop | nadam
    lr=0.001,               # learning rate (step size of the optimizer)
    batch_size=128,         # images processed together per update
    epochs=50,              # full passes over the training set
    # ---- other settings used in the project name----
    momentum=0.9,           # only used by "nesterov" (plain "sgd" uses 0)
    weight_decay=0.0005,    # Weight Decay
    batch_norm=True,        # BatchNorm after every conv layer
    dropout=0.5,            # dropout before the last FC layer
    seed=42,                # random seed -> reproducible results
    val_size=5000,          # images taken from the 50k training set for validation
    num_workers=2,          # CPU workers for loading data
    data_dir="./data",
    save_dir="./checkpoints",
    results_dir="./results",
    project="VGG6_CIFAR10_VCD", # W&B project name for displaying the information
)
