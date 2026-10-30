"""Evaluation code for rechecking the saved model).
   Command to check the saved model named best_model.pt
   python evaluate.py --ckpt best_model.pt
"""
import argparse

import torch
import torch.nn as nn

from config import DEFAULTS
from data import get_dataloaders
from model import build_model
from utils import get_device


def evaluate(model, loader, criterion, device):
    """Measure the model on a set of images. Returns (average loss, accuracy in %)."""
    model.eval()  # evaluation mode: dropout off, BatchNorm uses stored averages
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():  # no learning here, so no gradients are needed
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            count += images.size(0)

    return total_loss / count, 100.0 * correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best_model.pt")
    parser.add_argument("--data_dir", default=DEFAULTS["data_dir"])
    args = parser.parse_args()

    device = get_device()

    # Load the saved file: it holds the weights and the settings used for training
    saved = torch.load(args.ckpt, map_location=device)
    cfg = DEFAULTS.copy()
    cfg.update(saved["config"])
    cfg["data_dir"] = args.data_dir

    # Rebuild the model and put the saved weights into it
    model = build_model(cfg).to(device)
    model.load_state_dict(saved["model_state"])

    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("Settings   :", saved["config"])
    print(f"Validation : loss {val_loss:.4f} | top-1 accuracy {val_acc:.2f}%")
    print(f"Test       : loss {test_loss:.4f} | top-1 accuracy {test_acc:.2f}%")


if __name__ == "__main__":
    main()
