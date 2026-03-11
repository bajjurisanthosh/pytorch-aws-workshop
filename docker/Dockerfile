FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

LABEL maintainer="pytorch-workshop"
LABEL description="PyTorch AWS Workshop environment"

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    boto3>=1.34.0 \
    sagemaker>=2.210.0 \
    transformers>=4.38.0 \
    datasets>=2.18.0 \
    scikit-learn>=1.4.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    tqdm>=4.66.0 \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    jupyterlab>=4.1.0 \
    ipywidgets>=8.1.0 \
    Pillow>=10.2.0

# Workshop files
WORKDIR /workshop
COPY ../notebooks/ /workshop/notebooks/

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
