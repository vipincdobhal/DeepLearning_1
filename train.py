"""Training code. Run one experiment with:

    python train.py --activation relu --optimizer adam --lr 0.001 --batch_size 128 --epochs 20

Every hyper-parameter in config.DEFAULTS can be given as a --flag.
Also works as the program of a W&B sweep (see sweep.yaml).
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import wandb

from config import DEFAULTS
from data import get_dataloaders
from evaluate import evaluate
from model import build_model
from utils import build_optimizer, get_device, make_run_name, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One pass over the training data. Returns (average loss, accuracy in %)."""
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()               # 1. clear old gradients
        outputs = model(images)             # 2. forward pass: make predictions
        loss = criterion(outputs, labels)   # 3. how wrong are the predictions?
        loss.backward()                     # 4. backward pass: compute gradients
        optimizer.step()                    # 5. update the weights

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        count += images.size(0)

    return total_loss / count, 100.0 * correct / count


def run(overrides=None, use_wandb=True):
    """Train one configuration. 'overrides' is a dictionary of settings to change."""

    # ---- 1. Settings: defaults first, then the changes ----
    cfg = DEFAULTS.copy()
    if overrides is not None:
        cfg.update(overrides)

    # ---- 2. Start W&B (it also receives the settings) ----
    if use_wandb:
        mode = "online"
    else:
        mode = "disabled"
    wb = wandb.init(project=cfg["project"], config=cfg, mode=mode)
    cfg.update(dict(wb.config))  # in a sweep, W&B supplies the values
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["epochs"] = int(cfg["epochs"])

    name = make_run_name(cfg)
    wb.name = name
    wandb.define_metric("val_acc", summary="max")  # keep the BEST val_acc in the summary

    # ---- 3. Prepare everything ----
    set_seed(cfg["seed"])
    device = get_device()
    print("\n=== " + name + " | device: " + str(device) + " ===")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                                cfg["momentum"], cfg["weight_decay"])

    os.makedirs(cfg["save_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    checkpoint_path = os.path.join(cfg["save_dir"], name + ".pt")

    history = []
    best_val_acc = 0.0

    # ---- 4. Training loop ----
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        wandb.log(row, step=epoch)  # send the numbers to W&B

        print(f"epoch {epoch}/{cfg['epochs']} | train loss {train_loss:.4f} acc {train_acc:.2f}"
              f" | val loss {val_loss:.4f} acc {val_acc:.2f}")

        # Save the model whenever validation accuracy reaches a new best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_acc": val_acc}, checkpoint_path)

        # Stop if training blew up (loss became NaN or infinite)
        if not math.isfinite(train_loss):
            print("Loss is NaN or infinite, stopping this run early.")
            break

    # ---- 5. Final test accuracy, using the best saved model ----
    test_loss = float("nan")
    test_acc = float("nan")
    if os.path.exists(checkpoint_path):
        saved = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(saved["model_state"])
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    wb.summary["best_val_acc"] = best_val_acc
    wb.summary["test_acc"] = test_acc
    wb.summary["test_loss"] = test_loss
    print(f"Best val acc: {best_val_acc:.2f}% | Test top-1 acc: {test_acc:.2f}%")

    # ---- 6. Save the history to a file which can be used in future for plotting locally ----
    result = {"config": cfg, "history": history,
              "best_val_acc": best_val_acc, "test_acc": test_acc}
    history_path = os.path.join(cfg["results_dir"], name + "_history.json")
    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)

    wb.finish()
    return {"name": name, "best_val_acc": best_val_acc, "test_acc": test_acc}


def str2bool(text):
    """Turns the text 'True' / 'False' from the command line into a real True / False."""
    return text.lower() in ["true", "1", "yes"]


def parse_args():
    """Reads the --options typed after 'python train.py'. Missing ones use config.py."""
    p = argparse.ArgumentParser()
    p.add_argument("--activation", type=str, default=DEFAULTS["activation"])
    p.add_argument("--optimizer", type=str, default=DEFAULTS["optimizer"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--momentum", type=float, default=DEFAULTS["momentum"])
    p.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--batch_norm", type=str2bool, default=DEFAULTS["batch_norm"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--val_size", type=int, default=DEFAULTS["val_size"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--data_dir", type=str, default=DEFAULTS["data_dir"])
    p.add_argument("--save_dir", type=str, default=DEFAULTS["save_dir"])
    p.add_argument("--results_dir", type=str, default=DEFAULTS["results_dir"])
    p.add_argument("--project", type=str, default=DEFAULTS["project"])
    p.add_argument("--no_wandb", action="store_true", help="train without W&B logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings = vars(args)                   # turn the options into a dictionary
    no_wandb = settings.pop("no_wandb")     # this one is not a training setting
    run(settings, use_wandb=not no_wandb)
