"""
SageMaker inference handler.

SageMaker calls these four hooks:
  model_fn    — load model from /opt/ml/model
  input_fn    — deserialise raw request body
  predict_fn  — run forward pass
  output_fn   — serialise response
"""

import json
import os
import torch
import torch.nn.functional as F

from model import build_model


def model_fn(model_dir: str):
    """Load the trained model from model_dir."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    weight_path = os.path.join(model_dir, "model.pth")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def input_fn(request_body: str, content_type: str = "application/json"):
    """Deserialise request body into a tensor."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    data = json.loads(request_body)
    inputs = torch.tensor(data["inputs"], dtype=torch.float32)
    return inputs


def predict_fn(inputs: torch.Tensor, model):
    """Run inference and return softmax probabilities."""
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
    return probs


def output_fn(predictions: torch.Tensor, accept: str = "application/json"):
    """Serialise predictions to JSON."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps({"predictions": predictions.cpu().tolist()}), accept
