"""
Data utilities for the PyTorch AWS Workshop.

Handles CIFAR-10 loading, augmentation, and S3 caching.
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

logger = logging.getLogger(__name__)

# CIFAR-10 channel statistics (pre-computed)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

DATA_DIR = Path(os.environ.get("SM_CHANNEL_TRAINING", "/tmp/data"))


def get_transforms(train: bool = True):
    """Return appropriate torchvision transform pipeline."""
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_datasets(data_dir: Path = DATA_DIR):
    """Download (or load cached) CIFAR-10 train/val splits."""
    data_dir.mkdir(parents=True, exist_ok=True)
    train_ds = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=get_transforms(train=True)
    )
    val_ds = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=get_transforms(train=False)
    )
    logger.info("CIFAR-10  train=%d  val=%d", len(train_ds), len(val_ds))
    return train_ds, val_ds


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: Path = DATA_DIR,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader)."""
    train_ds, val_ds = get_datasets(data_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def save_checkpoint(state: dict, path: str):
    """Save training checkpoint to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Checkpoint saved → %s", path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a checkpoint from disk."""
    ckpt = torch.load(path, map_location=device)
    logger.info("Checkpoint loaded ← %s", path)
    return ckpt


def upload_to_s3(local_path: str, bucket: str, s3_key: str):
    """Upload a local file to S3."""
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, s3_key)
    logger.info("Uploaded s3://%s/%s", bucket, s3_key)


def get_device() -> torch.device:
    """Return best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
