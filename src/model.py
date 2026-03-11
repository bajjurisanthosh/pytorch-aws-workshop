"""
PyTorch CNN model for CIFAR-10 image classification.

Architecture: Custom ConvNet with residual-style skip connections.
Input:  (B, 3, 32, 32)
Output: (B, 10) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResidualBlock(nn.Module):
    """Two ConvBlocks with an optional projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class CIFAR10Net(nn.Module):
    """
    Lightweight ResNet-style network for CIFAR-10.

    ~1.2M parameters — trains in minutes on a single GPU.

    Layers
    ------
    stem     : 3  → 64  channels, 32×32
    stage1   : 64 → 64  channels, 32×32
    stage2   : 64 → 128 channels, 16×16  (stride-2)
    stage3   : 128→ 256 channels, 8×8   (stride-2)
    head     : GlobalAvgPool → FC(10)
    """

    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.stem = ConvBlock(3, 64)

        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class indices (argmax of logits)."""
        return self.forward(x).argmax(dim=1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_classes: int = 10, dropout: float = 0.3) -> CIFAR10Net:
    """Factory function — returns a fresh CIFAR10Net."""
    return CIFAR10Net(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    model = build_model()
    print(f"CIFAR10Net  |  parameters: {model.num_parameters:,}")
    dummy = torch.randn(4, 3, 32, 32)
    logits = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {logits.shape}")
