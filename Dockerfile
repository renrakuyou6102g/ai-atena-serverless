FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir \
    -r /workspace/requirements.txt

COPY handler.py /workspace/handler.py

COPY ATENA_7B_v9.zip /workspace/ATENA_7B_v9.zip

CMD ["python3", "-u", "/workspace/handler.py"]
