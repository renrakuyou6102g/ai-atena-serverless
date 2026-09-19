FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/huggingface
ENV TRANSFORMERS_CACHE=/workspace/huggingface
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
        wget && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# Python packages
# ============================================================

COPY requirements.txt /workspace/requirements.txt

RUN pip install \
    --no-cache-dir \
    -r /workspace/requirements.txt


# ============================================================
# Qwen2.5-7BをDockerイメージ内に保存
# ============================================================
#
# 起動時ダウンロードを避けるのが重要
#
# ============================================================

RUN python - <<'PY'

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/workspace/base_model",
    local_dir_use_symlinks=False
)

print("Qwen download completed.")

PY


# ============================================================
# ATENA v9 LoRA
# ============================================================

COPY atena_v9 /workspace/atena_v9


# ============================================================
# Handler
# ============================================================

COPY handler.py /workspace/handler.py


# ============================================================
# Run
# ============================================================

CMD ["python", "-u", "/workspace/handler.py"]
