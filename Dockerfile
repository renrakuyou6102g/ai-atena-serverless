FROM vllm/vllm-openai:v0.8.5.post1

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV HF_HOME=/workspace/huggingface
ENV HUGGINGFACE_HUB_CACHE=/workspace/huggingface/hub

# hf_transfer関連エラー回避
ENV HF_HUB_ENABLE_HF_TRANSFER=0

ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace


# ============================================================
# 必要ツール
# ============================================================

RUN apt-get update && \
    apt-get install -y \
        unzip \
        curl \
        wget && \
    rm -rf /var/lib/apt/lists/*


# ============================================================
# RunPod Serverless
# ============================================================

COPY requirements.txt /workspace/requirements.txt

RUN pip install \
    --no-cache-dir \
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

RUN echo "==========================" && \
    echo "ATENA v9 FILES" && \
    echo "==========================" && \
    find /workspace/atena_v9 \
        -maxdepth 5 \
        -type f | head -100


# ============================================================
# Handler
# ============================================================

COPY handler.py /workspace/handler.py


# vLLM公式イメージのデフォルトENTRYPOINTを解除
ENTRYPOINT []


# ============================================================
# 起動
# ============================================================

CMD ["python3", "-u", "/workspace/handler.py"]
