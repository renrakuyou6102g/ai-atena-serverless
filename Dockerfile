FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace


# ============================================================
# 基本ツール
# ============================================================

RUN apt-get update && \
    apt-get install -y \
        git \
        git-lfs \
        curl \
        wget \
        unzip && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# Python packages
# ============================================================

COPY requirements.txt /workspace/requirements.txt

RUN pip install \
    --no-cache-dir \
    --ignore-installed \
    -r /workspace/requirements.txt


# ============================================================
# Qwen2.5-7BをDockerイメージ内に保存
# ============================================================

RUN python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/workspace/base_model"
)

print("Qwen download completed.")
PY


# ============================================================
# AI ATENA v9 LoRA
# ============================================================

COPY ATENA_7B_v9.zip /workspace/ATENA_7B_v9.zip

RUN mkdir -p /workspace/atena_v9 && \
    unzip /workspace/ATENA_7B_v9.zip \
        -d /workspace/atena_v9 && \
    rm /workspace/ATENA_7B_v9.zip


# ============================================================
# LoRA確認
# ============================================================

RUN echo "=== ATENA v9 files ===" && \
    find /workspace/atena_v9 \
        -maxdepth 4 \
        -type f | head -100


# ============================================================
# Handler
# ============================================================

COPY handler.py /workspace/handler.py


# ============================================================
# Run
# ============================================================

CMD ["python", "-u", "/workspace/handler.py"]
