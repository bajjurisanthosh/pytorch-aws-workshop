#!/bin/bash
# Build Docker image and push to ECR

set -e

source ../.env

echo "Building Docker image..."
docker build -t ${ECR_REPO} -f Dockerfile ..

echo "Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS \
  --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

echo "Tagging image..."
docker tag ${ECR_REPO}:latest ${IMAGE_URI}

echo "Pushing to ECR: ${IMAGE_URI}"
docker push ${IMAGE_URI}

echo "Done! Image URI: ${IMAGE_URI}"
