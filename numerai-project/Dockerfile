# =============================================================
# Numerai GPU Pipeline - Docker image
# Base: NVIDIA CUDA 12.2 + Python 3.10
# GPU: Compatible g4dn (T4), g5 (A10G), p3 (V100)
# =============================================================
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NUMERAI_PROGRESS=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip curl && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps
COPY environment.yml /tmp/environment.yml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pandas numpy scikit-learn pyarrow pyyaml joblib \
    lightgbm xgboost optuna \
    numerapi cloudpickle boto3

# App code
WORKDIR /app
COPY config/ config/
COPY src/ src/

# Data and models directories (mounted or downloaded at runtime)
RUN mkdir -p /app/data /app/models /app/output

# Entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
