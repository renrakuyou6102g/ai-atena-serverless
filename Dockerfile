FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface

# Hugging Face Xetを無効化
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# タイムアウト
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV HF_HUB_ETAG_TIMEOUT=120

WORKDIR /workspace

# ============================================================
# OSパッケージ
# ============================================================

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Pythonライブラリ
# ============================================================

COPY requirements.txt /workspace/requirements.txt

RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir \
    -r /workspace/requirements.txt

# Xetを完全に外す
RUN pip3 uninstall -y hf-xet || true


# ============================================================
# Qwen 7BをDockerビルド時に保存
# ============================================================

RUN mkdir -p /workspace/base_model

RUN python3 - <<'PY'
from huggingface_hub import snapshot_download

print("Downloading Qwen2.5-7B-Instruct...")

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/workspace/base_model"
)

print("Qwen download finished.")
PY


# ============================================================
# ATENA v9
# ============================================================

COPY ATENA_7B_v9.zip /workspace/ATENA_7B_v9.zip

RUN mkdir -p /workspace/atena_v9 && \
    unzip /workspace/ATENA_7B_v9.zip -d /workspace/atena_v9 && \
    rm /workspace/ATENA_7B_v9.zip


# ============================================================
# Handler
# ============================================================

COPY handler.py /workspace/handler.py


# ============================================================
# 起動
# ============================================================

CMD ["python3", "-u", "/workspace/handler.py"]
