"""VGG6 Implementation: 4 convolution layers + 2 fully-connected layers = 6 weight layers.

Input: 3x32x32 CIFAR-10 image.
  Block 1: Conv(3->64)   -> Conv(64->64)   -> MaxPool   (32x32 -> 16x16)
  Block 2: Conv(64->128) -> Conv(128->128) -> MaxPool   (16x16 -> 8x8)
  Classifier: Flatten(128*8*8 = 8192) -> FC(8192->512) -> Dropout -> FC(512->10)
"""
import torch.nn as nn


def get_activation(name):
    """Return the activation layer that matches the given name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "silu":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "selu":
        return nn.SELU()
    else:
        raise ValueError("Unknown activation: " + name)

def get_norm(use_batch_norm, channels):
    """BatchNorm layer if requested, otherwise a layer that does nothing."""
    if use_batch_norm:
        return nn.BatchNorm2d(channels)
    else:
        return nn.Identity()


class VGG6(nn.Module):
    def __init__(self, activation="relu", batch_norm=True, dropout=0.5, num_classes=10):
        super().__init__()

        # With BatchNorm the conv bias is not needed
        use_bias = not batch_norm

        # ----- convolution layers (3x3 filters, padding 1 keeps the size) -----
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=use_bias)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=use_bias)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=use_bias)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=use_bias)

        # ----- one BatchNorm per conv layer -----
        self.bn1 = get_norm(batch_norm, 64)
        self.bn2 = get_norm(batch_norm, 64)
        self.bn3 = get_norm(batch_norm, 128)
        self.bn4 = get_norm(batch_norm, 128)

        # ----- other layers -----
        self.act = get_activation(activation)  # used after every layer except the last
        self.pool = nn.MaxPool2d(2, 2)  # halves height and width
        self.dropout = nn.Dropout(dropout)

        # ----- fully-connected layers -----
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)  # 10 scores, one per class

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.pool(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act(x)

        x = self.pool(x)

        # Flatten: (batch, 128, 8, 8) -> (batch, 8192)
        x = x.flatten(1)

        # Decision part
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_model(cfg):
    """Create the model from the settings dictionary (used by train.py and evaluate.py)."""
    return VGG6(cfg["activation"], cfg["batch_norm"], cfg["dropout"])
