"""
Evaluation script — loads a checkpoint and reports per-class accuracy,
confusion matrix, and top-5 misclassified examples.

Usage:
  python evaluate.py --checkpoint /path/to/best_model.pth
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from model import CIFAR10Net, build_model
from utils import get_dataloaders, get_device, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 Evaluation")
    p.add_argument("--checkpoint",  type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--data-dir",    type=str, default="/tmp/data")
    p.add_argument("--output-dir",  type=str, default="/tmp/eval_output")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    """Run full pass over loader, return (all_preds, all_labels, all_probs)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels)
        all_probs.append(probs.cpu())

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_probs).numpy(),
    )


def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix — CIFAR-10")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix → {output_path}")


def plot_per_class_accuracy(report_dict, class_names, output_path):
    accs = [report_dict[c]["recall"] for c in class_names]
    colors = ["#2196F3" if a >= 0.9 else "#FF9800" if a >= 0.8 else "#F44336" for a in accs]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, [a * 100 for a in accs], color=colors)
    ax.axhline(y=90, color="green", linestyle="--", alpha=0.6, label="90% target")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class Accuracy — CIFAR-10")
    ax.legend()

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc*100:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Per-class accuracy → {output_path}")


def main():
    args = parse_args()
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model().to(device)
    ckpt = load_checkpoint(args.checkpoint, device)
    state_dict = ckpt.get("model_state_dict", ckpt)  # support both formats
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  "
          f"(val_acc={ckpt.get('val_acc', '?'):.2f}%)")

    # Data
    _, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=2,
        data_dir=Path(args.data_dir),
    )

    # Inference
    print("Running inference on validation set...")
    preds, labels, probs = run_inference(model, val_loader, device)

    # Overall accuracy
    overall_acc = (preds == labels).mean() * 100
    print(f"\nOverall accuracy: {overall_acc:.2f}%")

    # Classification report
    class_names = CIFAR10Net.CLASS_NAMES
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    report_text = classification_report(labels, preds, target_names=class_names)
    print("\nClassification Report:")
    print(report_text)

    # Save report as JSON
    report_path = output_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, str(output_dir / "confusion_matrix.png"))

    # Per-class accuracy bar chart
    plot_per_class_accuracy(report, class_names, str(output_dir / "per_class_accuracy.png"))

    # Top-5 errors per class
    errors_path = output_dir / "top_errors.json"
    errors = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = (labels == cls_idx) & (preds != cls_idx)
        if mask.sum() == 0:
            continue
        wrong_probs = probs[mask]
        wrong_preds = preds[mask]
        top = np.argsort(-wrong_probs.max(axis=1))[:5]
        errors[cls_name] = [
            {"predicted": class_names[wrong_preds[i]], "confidence": float(wrong_probs[i].max())}
            for i in top
        ]
    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=2)
    print(f"  Top errors → {errors_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
