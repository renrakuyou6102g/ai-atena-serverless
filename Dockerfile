FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

COPY AI_ATENA_v3.zip /tmp/AI_ATENA_v3.zip

RUN mkdir -p /workspace/AI_ATENA_v3 \
    && unzip /tmp/AI_ATENA_v3.zip -d /workspace/AI_ATENA_v3 \
    && rm /tmp/AI_ATENA_v3.zip \
    && find /workspace/AI_ATENA_v3 -maxdepth 2 -type f

CMD ["python", "-u", "handler.py"]
