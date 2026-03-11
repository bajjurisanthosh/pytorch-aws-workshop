"""
Training script — runs locally or inside a SageMaker Training Job.

SageMaker injects hyperparameters via CLI args and environment variables:
  SM_CHANNEL_TRAINING  → path to training data
  SM_MODEL_DIR         → where to save the final model artifact
  SM_OUTPUT_DATA_DIR   → where to save additional outputs (metrics, plots)

Usage (local):
  python train.py --epochs 30 --batch-size 128 --lr 0.1

Usage (SageMaker): launched automatically by the PyTorch estimator.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from model import build_model
from utils import get_dataloaders, get_device, save_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 Training")

    # Hyperparameters
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=0.1)
    p.add_argument("--momentum",    type=float, default=0.9)
    p.add_argument("--weight-decay",type=float, default=5e-4)
    p.add_argument("--dropout",     type=float, default=0.3)

    # SageMaker environment (with local fallbacks)
    p.add_argument("--model-dir",   type=str, default=os.environ.get("SM_MODEL_DIR", "/tmp/model"))
    p.add_argument("--data-dir",    type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/tmp/data"))
    p.add_argument("--output-dir",  type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/tmp/output"))
    p.add_argument("--num-gpus",    type=int, default=int(os.environ.get("SM_NUM_GPUS", 0)))
    p.add_argument("--resume",      type=str, default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


# ──────────────────────────────────────────────
# Training & validation helpers
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = correct = total = 0

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

        if (step + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                "  epoch %02d  step %4d/%d  loss=%.4f  acc=%.2f%%  lr=%.5f",
                epoch, step + 1, len(loader),
                running_loss / total, 100.0 * correct / total, lr_now,
            )

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return running_loss / total, 100.0 * correct / total


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    device = get_device()
    logger.info("Device: %s", device)
    logger.info("Args: %s", vars(args))

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        data_dir=Path(args.data_dir),
    )

    # Model
    model = build_model(dropout=args.dropout).to(device)
    logger.info("Model parameters: %d", model.num_parameters)

    # Multi-GPU
    if args.num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info("Using DataParallel on %d GPUs", args.num_gpus)

    # Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start_epoch = 1
    best_ckpt_path = os.path.join(args.model_dir, "best_model.pth")

    if args.resume:
        from utils import load_checkpoint
        ckpt = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["val_acc"]
        logger.info("Resumed from epoch %d, val_acc=%.2f%%", ckpt["epoch"], best_val_acc)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %02d/%d  train_loss=%.4f  train_acc=%.2f%%  "
            "val_loss=%.4f  val_acc=%.2f%%  (%.1fs)",
            epoch, args.epochs, train_loss, train_acc, val_loss, val_acc, elapsed,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {
                "epoch": epoch,
                "model_state_dict": (
                    model.module.state_dict() if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }
            save_checkpoint(state, best_ckpt_path)
            logger.info("  *** New best val_acc=%.2f%% — checkpoint saved ***", val_acc)

    # Save training history
    history_path = os.path.join(args.output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved → %s", history_path)

    # Save final model in TorchScript format for SageMaker serving
    final_model_path = os.path.join(args.model_dir, "model.pth")
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(raw_model.state_dict(), final_model_path)
    logger.info("Final model saved → %s", final_model_path)

    logger.info("Training complete. Best val_acc=%.2f%%", best_val_acc)


if __name__ == "__main__":
    main()
