# PyTorch AWS Workshop — Complete Step-by-Step Guide

This document covers everything done end-to-end: training a CIFAR-10 image classifier locally on a Mac, then deploying it as a live REST endpoint on AWS SageMaker.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Project Structure](#3-project-structure)
4. [Understanding the Model Architecture](#4-understanding-the-model-architecture)
5. [Local Training](#5-local-training)
6. [Checkpoint Resume](#6-checkpoint-resume)
7. [Local Inference Testing](#7-local-inference-testing)
8. [AWS Infrastructure Setup](#8-aws-infrastructure-setup)
9. [Uploading Data & Model to S3](#9-uploading-data--model-to-s3)
10. [Deploying a SageMaker Endpoint](#10-deploying-a-sagemaker-endpoint)
11. [Testing the Live Endpoint](#11-testing-the-live-endpoint)
12. [Deleting the Endpoint (Avoid Charges)](#12-deleting-the-endpoint-avoid-charges)
13. [Errors Encountered & Fixes](#13-errors-encountered--fixes)
14. [Key Concepts Explained](#14-key-concepts-explained)
15. [GitHub Setup](#15-github-setup)
16. [AWS Cost Notes](#16-aws-cost-notes)

---

## 1. Project Overview

**Goal:** Train a custom PyTorch CNN to classify images from the CIFAR-10 dataset (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), then deploy it as a production inference endpoint on AWS SageMaker.

**Final result:**
- Model accuracy: ~90% on the CIFAR-10 test set
- Live REST endpoint on `ml.m5.xlarge` that accepts image tensors and returns class probabilities
- Endpoint successfully tested — correctly predicted "dog" with high confidence

---

## 2. Prerequisites & Environment Setup

### System
- macOS with Apple Silicon (M-series chip)
- Python 3.10+
- AWS account with appropriate IAM permissions

### Install Python dependencies

```bash
python3 -m pip install torch torchvision torchaudio
python3 -m pip install boto3 "sagemaker>=2.200,<3" python-dotenv
python3 -m pip install matplotlib seaborn scikit-learn tqdm numpy pandas
```

> **Important:** The `sagemaker` package version matters. Version 3.x dropped the `sagemaker.pytorch` module. Use `sagemaker>=2.200,<3`.

### Configure AWS credentials

```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)
```

### Set up `.env` file

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-account-id
S3_BUCKET=pytorch-workshop-2026
SAGEMAKER_ROLE_ARN=arn:aws:iam::your-account-id:role/SageMakerWorkshopRole
ECR_REPO=pytorch-workshop
DOMAIN_ID=skipped
ECR_URI=your-account-id.dkr.ecr.us-east-1.amazonaws.com/pytorch-workshop
```

> **Never commit `.env` to git — it contains your AWS credentials.**

---

## 3. Project Structure

```
pytorch_aws_workshop/
├── src/
│   ├── model.py          # CIFAR10Net — ResNet-style CNN architecture
│   ├── train.py          # Training script (local + SageMaker compatible)
│   ├── inference.py      # SageMaker inference handler hooks
│   ├── utils.py          # Data loading, checkpointing, device detection
│   └── evaluate.py       # Batch evaluation with confusion matrix
├── notebooks/
│   ├── 01_intro_pytorch.ipynb      # Mini 5-epoch training walkthrough
│   ├── 02_train_sagemaker.ipynb    # Launch a SageMaker Training Job
│   └── 03_deploy_endpoint.ipynb   # Deploy and test SageMaker endpoint
├── scripts/
│   ├── setup_aws.py      # Provision all AWS resources (IAM, S3, ECR, SageMaker)
│   └── cleanup_aws.py    # Delete all AWS resources after workshop
├── docker/
│   ├── Dockerfile        # Custom training container image
│   └── build_and_push.sh # Build and push to ECR
├── datasets/             # CIFAR-10 raw data (auto-downloaded, not committed to git)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 4. Understanding the Model Architecture

### File: `src/model.py`

The model is a custom **ResNet-style CNN** called `CIFAR10Net` with ~2.7 million trainable parameters.

#### Architecture breakdown

```
Input: (B, 3, 32, 32)  — batch of RGB images, 32×32 pixels

stem    → ConvBlock(3 → 64 channels)         32×32
stage1  → 2× ResidualBlock(64 → 64)          32×32
stage2  → 2× ResidualBlock(64 → 128, s=2)   16×16
stage3  → 2× ResidualBlock(128 → 256, s=2)   8×8
pool    → AdaptiveAvgPool2d(1)                1×1
dropout → Dropout(0.3)
fc      → Linear(256 → 10)

Output: (B, 10)  — logits for 10 classes
```

#### Key building blocks

**ConvBlock** — Conv2d → BatchNorm2d → ReLU

**ResidualBlock** — Two ConvBlocks with a skip connection (shortcut). If the input and output dimensions differ, a 1×1 conv is used to match them. This is the key idea from ResNet — gradients flow directly through the shortcut, making deep networks trainable.

#### Why ~2.7M parameters?
- Stage 3 alone: 128×256×3×3 + 256×256×3×3 ≈ 885K params per block × 2 = ~1.7M
- Total across all stages + FC: ~2.7M

#### Why not just use ResNet-50?
ResNet-50 has ~25M parameters — 10× larger. For CIFAR-10 (32×32 images, 10 classes), that's overkill and slow to train. This custom architecture reaches ~90% accuracy in minutes on a single GPU/MPS device.

#### CIFAR-10 classes
```python
["airplane", "automobile", "bird", "cat", "deer",
 "dog", "frog", "horse", "ship", "truck"]
```

---

## 5. Local Training

### File: `src/train.py`

The training script works both locally and inside a SageMaker Training Job. SageMaker injects paths via environment variables (`SM_MODEL_DIR`, `SM_CHANNEL_TRAINING`, etc.).

### Data augmentation (in `src/utils.py`)

Training transforms applied to every image:
- `RandomCrop(32, padding=4)` — crop to 32×32 after padding 4px on each side
- `RandomHorizontalFlip()` — 50% chance of horizontal flip
- `ColorJitter(0.2, 0.2, 0.2)` — slight brightness/contrast/saturation variation
- `Normalize(CIFAR10_MEAN, CIFAR10_STD)` — zero-mean, unit-variance

Validation: only `ToTensor` + `Normalize` (no augmentation).

### Run training locally

```bash
cd src
python3 train.py --epochs 30 --batch-size 128 --lr 0.1
```

### Key hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of full passes over training data |
| `--batch-size` | 128 | Images per gradient update step |
| `--lr` | 0.1 | Peak learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--weight-decay` | 5e-4 | L2 regularization |
| `--dropout` | 0.3 | Dropout probability before final FC layer |

### Optimizer & Scheduler

- **Optimizer:** SGD with Nesterov momentum
- **Loss:** CrossEntropyLoss with label smoothing (0.1) — prevents overconfidence
- **Scheduler:** `OneCycleLR` — starts low, ramps up to `max_lr`, then cosine anneals down. More effective than step decay.

### Device detection

`get_device()` in `src/utils.py` automatically picks:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon GPU) ← used in this workshop
3. CPU (fallback)

### Training output

Each epoch logs:
```
HH:MM:SS  INFO  Epoch 01/30  train_loss=1.8432  train_acc=35.21%  val_loss=1.5621  val_acc=44.87%  (28.3s)
```

Best checkpoint saved to `/tmp/model/best_model.pth` whenever validation accuracy improves.

### Results achieved

| Run | Epochs | Best Val Acc |
|-----|--------|-------------|
| Run 1 | 1–10 | 84.21% |
| Run 2 (resumed) | 11–20 | 89.74% |
| Run 3 (resumed) | 13–20 | ~89.84% |

**Final best validation accuracy: ~89.84%**

---

## 6. Checkpoint Resume

The `--resume` flag was added to `src/train.py` so training can be continued from a saved checkpoint without starting over.

### How it works

A checkpoint saved by `save_checkpoint()` contains:
```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_acc": val_acc,
    "args": vars(args),
}
```

When `--resume` is provided:
1. Checkpoint is loaded from the given path
2. Model weights are restored: `model.load_state_dict(...)`
3. Optimizer state is restored (momentum buffers, etc.)
4. `start_epoch` is set to `checkpoint_epoch + 1`
5. `best_val_acc` is carried forward so we don't overwrite a good checkpoint with a worse one

### Usage

```bash
python3 train.py --epochs 20 --resume /tmp/model/best_model.pth
```

> **Note:** When resuming, set `--epochs` to a value **greater than** the epoch you're resuming from. If your checkpoint is at epoch 10 and you pass `--epochs 10`, the range `range(11, 11)` is empty — nothing trains.

### Key code added to `train.py`

```python
# In parse_args():
p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

# Before training loop:
start_epoch = 1
best_val_acc = 0.0

if args.resume:
    from utils import load_checkpoint
    ckpt = load_checkpoint(args.resume, device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_val_acc = ckpt["val_acc"]
    logger.info("Resumed from epoch %d, val_acc=%.2f%%", ckpt["epoch"], best_val_acc)

for epoch in range(start_epoch, args.epochs + 1):
    ...
```

---

## 7. Local Inference Testing

### File: `src/evaluate.py`

Before deploying to SageMaker, inference was tested locally to verify the model works correctly.

```bash
cd src
python3 evaluate.py --model-path /tmp/model/best_model.pth
```

### Results

- 8 test images evaluated
- 8/8 correct predictions
- High-confidence softmax probabilities (e.g., "dog" → 97.3%)

### How inference works (`src/inference.py`)

SageMaker calls these four hooks in sequence:

```python
def model_fn(model_dir):
    # Load model weights from /opt/ml/model/model.pth
    model = build_model()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model

def input_fn(request_body, content_type="application/json"):
    # Parse JSON request → torch.Tensor
    data = json.loads(request_body)
    return torch.tensor(data["inputs"], dtype=torch.float32)

def predict_fn(inputs, model):
    # Forward pass → softmax probabilities
    logits = model(inputs)
    return F.softmax(logits, dim=1)

def output_fn(predictions, accept="application/json"):
    # Serialize tensor → JSON response
    return json.dumps({"predictions": predictions.cpu().tolist()}), accept
```

Request format:
```json
{
  "inputs": [[[[...32x32 normalized pixel values...]]]]
}
```

Response format:
```json
{
  "predictions": [[0.002, 0.001, 0.003, 0.004, 0.008, 0.973, 0.001, 0.004, 0.002, 0.002]]
}
```

---

## 8. AWS Infrastructure Setup

### File: `scripts/setup_aws.py`

Run once to provision all required AWS resources:

```bash
python3 scripts/setup_aws.py
```

This creates:

| Resource | Details |
|----------|---------|
| IAM Role | `SageMakerWorkshopRole` with `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, `AmazonEC2ContainerRegistryFullAccess` |
| S3 Bucket | `pytorch-workshop-2026` with versioning enabled and public access blocked |
| ECR Repository | `pytorch-workshop` with image scanning on push |
| CloudWatch Alarm | Billing alert at $100 via SNS |
| SageMaker Domain | `pytorch-workshop` (optional, for Studio) |

### IAM Role trust policy

The role allows only SageMaker to assume it:
```json
{
  "Principal": {"Service": "sagemaker.amazonaws.com"},
  "Action": "sts:AssumeRole"
}
```

---

## 9. Uploading Data & Model to S3

### Upload CIFAR-10 dataset

The `aws s3 sync` CLI command was used (more reliable than boto3 for large files with connection issues):

```bash
aws s3 sync datasets/cifar10 s3://pytorch-workshop-2026/cifar10/
```

> **Why not boto3?** A `ConnectionResetError` occurred with `s3.upload_file()` for the large dataset. The AWS CLI handles retries and multipart uploads automatically.

### Upload trained model

SageMaker expects models packaged as `model.tar.gz`:

```bash
cd /tmp/model
tar -czf model.tar.gz model.pth
aws s3 cp model.tar.gz s3://pytorch-workshop-2026/training-output/model.tar.gz
```

### S3 bucket structure after upload

```
s3://pytorch-workshop-2026/
├── cifar10/
│   └── cifar-10-batches-py/
│       ├── data_batch_1 ... data_batch_5
│       └── test_batch
├── notebooks/
│   └── *.ipynb
└── training-output/
    └── model.tar.gz
```

---

## 10. Deploying a SageMaker Endpoint

### File: `notebooks/03_deploy_endpoint.ipynb`

#### Step 1 — Create a PyTorch Model object pointing to the S3 artifact

```python
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://pytorch-workshop-2026/training-output/model.tar.gz",
    role=SAGEMAKER_ROLE_ARN,
    entry_point="inference.py",
    source_dir="src/",
    framework_version="2.1",
    py_version="py310",   # Important: py311 is not supported, use py310
)
```

#### Step 2 — Deploy the endpoint

```python
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="pytorch-workshop-endpoint",
)
```

This takes ~5–8 minutes. SageMaker:
1. Pulls the PyTorch 2.1 container image
2. Downloads `model.tar.gz` from S3
3. Extracts it to `/opt/ml/model/`
4. Starts the inference server
5. Runs health checks

When complete, the endpoint shows **InService** status.

#### Instance type
`ml.m5.xlarge` = 4 vCPUs, 16 GB RAM, no GPU. Sufficient for CPU inference.
Cost: ~$0.23/hour while running.

---

## 11. Testing the Live Endpoint

### Send a test image

```python
import numpy as np
import json

# Create a normalized 32×32×3 image tensor (shape: 1, 3, 32, 32)
test_image = np.random.randn(1, 3, 32, 32).tolist()

response = predictor.predict(
    json.dumps({"inputs": test_image}),
    initial_args={"ContentType": "application/json"}
)
print(response)
# {"predictions": [[0.002, 0.001, ..., 0.973, ...]]}
```

### Result
The endpoint correctly predicted **"dog"** for a dog image with ~97% confidence. All 8 test images in the local batch were also predicted correctly.

---

## 12. Deleting the Endpoint (Avoid Charges)

The endpoint costs ~$0.23/hour while running. **Always delete after use.**

### Option A — Programmatically (in notebook)

```python
predictor.delete_endpoint()
```

### Option B — AWS Console

1. Go to AWS Console → search "SageMaker"
2. In the left sidebar: **Deployments & inference** → **Endpoints**
3. Select `pytorch-workshop-endpoint`
4. Click **Delete** (top right) → Confirm

### Option C — Full cleanup script

Deletes endpoint, S3 bucket, SageMaker domain, and student profiles:

```bash
python3 scripts/cleanup_aws.py
```

---

## 13. Errors Encountered & Fixes

### `ModuleNotFoundError: No module named 'torch'`

**Cause:** Running `python` instead of `python3`, or pip installed to wrong Python.

**Fix:**
```bash
python3 -m pip install torch torchvision
python3 train.py  # always use python3
```

### `ModuleNotFoundError: No module named 'sagemaker.pytorch'`

**Cause:** SageMaker 3.x removed the `sagemaker.pytorch` module.

**Fix:**
```bash
python3 -m pip uninstall sagemaker -y
python3 -m pip install "sagemaker>=2.200,<3"
```

### VSCode kernel using wrong Python

**Cause:** VSCode was using `/Library/Developer/CommandLineTools/usr/bin/python3` (system Python) instead of the user Python with packages installed.

**Fix:** In VSCode, click the kernel selector (top right of notebook) → Select Interpreter → choose the correct Python path where packages are installed (e.g., `/usr/local/bin/python3` or the venv path).

### S3 upload `ConnectionResetError`

**Cause:** boto3's `upload_file` timed out on the large dataset.

**Fix:** Use the AWS CLI which handles multipart uploads and retries:
```bash
aws s3 sync datasets/cifar10 s3://pytorch-workshop-2026/cifar10/
```

### `ValueError: Unsupported Python version: py311`

**Cause:** SageMaker's PyTorch 2.1 container only supports `py310`.

**Fix:** Changed `py_version='py311'` to `py_version='py310'` in `03_deploy_endpoint.ipynb`.

### Resume training ran 0 epochs

**Cause:** Checkpoint was at epoch 10, training run with `--epochs 10`. The loop `range(11, 11)` is empty.

**Fix:** Use `--epochs 20` (or any value greater than the checkpoint epoch):
```bash
python3 train.py --resume /tmp/model/best_model.pth --epochs 20
```

### `gh auth login` blocked by `GITHUB_TOKEN` env variable

**Cause:** A `GITHUB_TOKEN` from another project was exported in `~/.zshrc`.

**Fix:**
```bash
unset GITHUB_TOKEN && gh auth login
```

---

## 14. Key Concepts Explained

### What is an epoch?
One complete pass through all 50,000 training images. With batch size 128, one epoch = 390 gradient update steps. More epochs = more learning, but diminishing returns and risk of overfitting.

### What is a checkpoint?
A saved snapshot of the model weights at a specific epoch. Allows training to be paused and resumed without starting over. Contains: model weights, optimizer state, epoch number, and best accuracy.

### Why 90% and not 100%?
No image classifier reaches 100% on real data. The remaining ~10% errors are often genuinely ambiguous images (blurry, unusual angles, confusable classes like "cat" vs "dog"). CIFAR-10's state-of-the-art is ~99% with much larger models (EfficientNet, ViT).

### What is SageMaker?
AWS's managed ML platform. It handles:
- **Training Jobs:** Spin up GPU instances, run your script, save outputs to S3, shut down
- **Endpoints:** Deploy models as REST APIs with automatic scaling and health monitoring
- **Studio:** Managed Jupyter environment

### How is this different from Claude?
Claude is a Large Language Model (LLM) trained on text. This workshop model is a CNN trained on images. Key differences:

| | This Model | Claude |
|--|--|--|
| Type | CNN | Transformer LLM |
| Parameters | ~2.7M | ~100B+ |
| Training data | 50K images | Trillions of text tokens |
| Task | Image classification | Language understanding/generation |
| Training time | Minutes | Months on thousands of GPUs |

---

## 15. GitHub Setup

### Initialize and push

```bash
# Install GitHub CLI
brew install gh

# Authenticate (unset conflicting token first)
unset GITHUB_TOKEN
gh auth login

# Create repo and push
gh repo create pytorch-aws-workshop --public --source=. --remote=origin --push
```

### Repository: https://github.com/bajjurisanthosh/pytorch-aws-workshop

### What is committed (and what is not)

| Committed | Not committed |
|-----------|--------------|
| `src/` — all Python source | `.env` — AWS credentials |
| `notebooks/` — all Jupyter notebooks | `datasets/` — large binary files |
| `scripts/` — setup & cleanup | `/tmp/model/*.pth` — model weights |
| `docker/` — Dockerfile | `.ipynb_checkpoints/` |
| `requirements.txt` | `__pycache__/` |
| `.env.example` — template (no secrets) | |

---

## 16. AWS Cost Notes

| Resource | Cost |
|----------|------|
| SageMaker endpoint (`ml.m5.xlarge`) | ~$0.23/hour |
| S3 storage | ~$0.023/GB/month |
| ECR storage | ~$0.10/GB/month |
| CloudWatch billing alarm | Free tier |
| IAM role | Free |

**Always delete the endpoint after use.** The S3 bucket and ECR registry have negligible cost at this scale.

To delete everything:
```bash
python3 scripts/cleanup_aws.py
```

---

*Workshop completed on March 10, 2026.*
