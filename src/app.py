"""
Real-time CIFAR-10 inference + Claude chatbot — Streamlit web app.

Usage:
    streamlit run src/app.py

Set your API key before running:
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, CIFAR10Net

TRANSFORM = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

MODEL_PATH = "/tmp/model/best_model.pth"

SYSTEM_PROMPT = """You are an AI assistant embedded in a CIFAR-10 image classification demo app.

The model being used is CIFAR10Net — a custom ResNet-style CNN with ~2.7M parameters trained on the CIFAR-10 dataset.
It classifies images into 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
The model was trained for ~30 epochs using SGD + OneCycleLR scheduler and achieves ~90% validation accuracy.
It was deployed on AWS SageMaker (ml.m5.xlarge endpoint) and is also running locally via MPS (Apple Silicon).

Answer questions about this model, deep learning, CIFAR-10, PyTorch, AWS SageMaker, or anything related.
Keep answers concise and practical."""


@st.cache_resource
def load_model():
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = build_model()
    ckpt = torch.load(MODEL_PATH, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict(image: Image.Image, model, device):
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    return {name: float(probs[i]) for i, name in enumerate(CIFAR10Net.CLASS_NAMES)}


# ── Page config ─────────────────────────────────────
st.set_page_config(page_title="CIFAR-10 Demo", layout="wide")
st.title("CIFAR-10 Image Classifier + AI Assistant")

tab1, tab2 = st.tabs(["Image Classifier", "Chat with Claude"])

# ── Tab 1: Image Classifier ──────────────────────────
with tab1:
    if not Path(MODEL_PATH).exists():
        st.error(f"Model not found at `{MODEL_PATH}`. Train first: `python3 src/train.py --epochs 30`")
    else:
        model, device = load_model()
        st.success(f"Model loaded on **{device}** | ~2.7M params | Best val acc ~90%")

        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

        if uploaded:
            image = Image.open(uploaded)
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Uploaded image", use_container_width=True)

            with col2:
                with st.spinner("Running inference..."):
                    scores = predict(image, model, device)

                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top_class, top_conf = sorted_scores[0]

                st.markdown(f"### Prediction: **{top_class.upper()}**")
                st.markdown(f"Confidence: **{top_conf:.1%}**")
                st.divider()
                st.markdown("**Top 5 scores:**")
                for name, score in sorted_scores[:5]:
                    st.progress(score, text=f"{name:<12} {score:.1%}")

# ── Tab 2: Chatbot ───────────────────────────────────
with tab2:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning("Set `ANTHROPIC_API_KEY` env var before running, then restart the app.")
        st.code("export ANTHROPIC_API_KEY=sk-ant-...")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about this model, deep learning, AWS..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            client = anthropic.Anthropic(api_key=api_key)
            response_placeholder = st.empty()
            full_response = ""

            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=st.session_state.messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
