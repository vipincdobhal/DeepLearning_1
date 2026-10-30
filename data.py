# Loads CIFAR-10, normalizes the images and applies data augmentation.
# The mean and standard deviation used for normalization are calculated here from the
# training images (These values are not hardcoded).

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def calculate_mean_std(images):
    """Average and spread of each colour channel (R, G, B).
    images: numpy array of shape (number of images, 32, 32, 3) with values 0 to 255.
    Returns two lists of 3 numbers: mean and std, with pixel values scaled to 0..1.
    """
    means = []
    stds = []
    for channel in range(3):  # 0 = red, 1 = green, 2 = blue
        values = images[:, :, :, channel].astype(np.float32) / 255.0
        means.append(float(values.mean()))
        stds.append(float(values.std()))
    return means, stds


def get_dataloaders(cfg):
    """Returns 3 loaders: train (45,000 images), validation (5,000) and test (10,000)."""
    folder = cfg["data_dir"]

    # 1. Split the training images into train and validation ----
    # (the raw images are loaded without any transform, only to count and measure them)
    raw_train = datasets.CIFAR10(folder, train=True, download=True)

    # The seed makes the split identical in every run.
    all_indices = np.random.RandomState(cfg["seed"]).permutation(len(raw_train))
    val_indices = all_indices[:cfg["val_size"]]
    train_indices = all_indices[cfg["val_size"]:]

    # 2. Calculate mean and std from the TRAINING images only ----
    mean, std = calculate_mean_std(raw_train.data[train_indices])
    print("Calculated mean:", [round(m, 4) for m in mean])
    print("Calculated std :", [round(s, 4) for s in std])

    # 3. Transforms for TRAINING images: augmentation + normalization ----
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # add 4 pixels of border, cut a random 32x32 piece
        transforms.RandomHorizontalFlip(p=0.5),  # mirror the image half of the time
        transforms.ToTensor(),  # picture -> numbers between 0 and 1
        transforms.Normalize(mean, std),  # (value - mean) / std
    ])

    # 4. Transforms for VALIDATION and TEST images: normalization only ----
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 5. Load the data again, this time with the transforms ----
    train_data = datasets.CIFAR10(folder, train=True, download=True, transform=train_transform)
    val_data = datasets.CIFAR10(folder, train=True, download=True, transform=eval_transform)
    test_data = datasets.CIFAR10(folder, train=False, download=True, transform=eval_transform)

    train_set = Subset(train_data, train_indices)
    val_set = Subset(val_data, val_indices)

    # Show how many images each part has
    print("Images -> train:", len(train_set), "| validation:", len(val_set),
          "| test:", len(test_data))

    # 6. Data loaders: hand the images to the model in batches ----
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])  # makes the shuffling repeatable
    use_gpu = torch.cuda.is_available()

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              generator=generator, num_workers=cfg["num_workers"],
                              pin_memory=use_gpu)
    val_loader = DataLoader(val_set, batch_size=500, shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=use_gpu)
    test_loader = DataLoader(test_data, batch_size=500, shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=use_gpu)

    return train_loader, val_loader, test_loader
