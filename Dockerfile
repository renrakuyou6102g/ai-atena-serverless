FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV HF_HOME=/workspace/huggingface
ENV HUGGINGFACE_HUB_CACHE=/workspace/huggingface/hub
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
# ATENA v9 LoRA
# ============================================================

COPY ATENA_7B_v9.zip /workspace/ATENA_7B_v9.zip

RUN mkdir -p /workspace/atena_v9 && \
    unzip /workspace/ATENA_7B_v9.zip \
        -d /workspace/atena_v9 && \
    rm /workspace/ATENA_7B_v9.zip


# ============================================================
# LoRA確認
# ============================================================

RUN echo "====================================" && \
    echo "ATENA v9 FILES" && \
    echo "====================================" && \
    find /workspace/atena_v9 \
        -maxdepth 5 \
        -type f | head -100


# ============================================================
# Handler
# ============================================================

COPY handler.py /workspace/handler.py


# ============================================================
# Run
# ============================================================

CMD ["python", "-u", "/workspace/handler.py"]
