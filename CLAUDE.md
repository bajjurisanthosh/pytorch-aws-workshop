# PyTorch AWS Workshop

## Project Overview

CIFAR-10 image classification workshop using PyTorch + AWS SageMaker. Trains a custom ResNet-style CNN (~2.7M params) locally or on SageMaker, deploys as an endpoint, and provides a Streamlit UI with a Claude chatbot.

## Architecture

```
src/
  model.py       — CIFAR10Net (ResNet-style CNN, ~2.7M params, 10 classes)
  train.py       — Training script (local or SageMaker Training Job)
  inference.py   — SageMaker inference handler (model_fn/input_fn/predict_fn/output_fn)
  evaluate.py    — Evaluation utilities
  utils.py       — DataLoaders, device detection, checkpoint helpers
  app.py         — Streamlit app: image classifier + Claude chatbot

scripts/
  setup_aws.py   — Provisions IAM role, S3 bucket, ECR repo, SageMaker domain
  cleanup_aws.py — Tears down all AWS resources

docker/
  Dockerfile           — Training container image
  build_and_push.sh    — Builds and pushes to ECR

notebooks/
  01_intro_pytorch.ipynb    — PyTorch basics
  02_train_sagemaker.ipynb  — SageMaker training job
  03_deploy_endpoint.ipynb  — Deploy & invoke endpoint

datasets/
  cifar-10-batches-py/  — CIFAR-10 data (downloaded locally)
```

## Setup

### Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-account-id
S3_BUCKET=pytorch-workshop-2026
SAGEMAKER_ROLE_ARN=arn:aws:iam::your-account-id:role/SageMakerWorkshopRole
ECR_REPO=pytorch-workshop
DOMAIN_ID=your-sagemaker-domain-id
```

Also set `ANTHROPIC_API_KEY` for the chatbot tab in the Streamlit app.

### Install Dependencies

```bash
pip install -r requirements.txt
pip install streamlit anthropic python-dotenv
```

## Common Commands

### Train Locally

```bash
python src/train.py --epochs 30 --batch-size 128 --lr 0.1
# Saves best model to /tmp/model/best_model.pth
```

### Run Streamlit App

```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run src/app.py
```

### AWS Infrastructure Setup

```bash
python scripts/setup_aws.py   # Provision all AWS resources
python scripts/cleanup_aws.py # Tear down resources (avoid surprise charges)
```

## Model Details

- **Architecture**: CIFAR10Net — stem + 3 residual stages (64→128→256 channels) + GlobalAvgPool + FC
- **Input**: (B, 3, 32, 32) normalized with CIFAR-10 mean/std
- **Output**: (B, 10) logits for classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Training**: SGD + Nesterov momentum + OneCycleLR scheduler, label smoothing 0.1
- **Target accuracy**: ~90% validation accuracy in 30 epochs

## SageMaker Notes

- Training job uses `SM_CHANNEL_TRAINING`, `SM_MODEL_DIR`, `SM_OUTPUT_DATA_DIR` env vars
- Inference endpoint expects `{"inputs": [[...]]}` JSON with shape `(B, 3, 32, 32)`
- Endpoint returns `{"predictions": [[...]]}` with softmax probabilities
- Default instance: `ml.m5.xlarge` for inference

## AWS Cost Warning

Always run `python scripts/cleanup_aws.py` after the workshop to avoid ongoing charges. A billing alarm is set at $100 via CloudWatch during setup.
