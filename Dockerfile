ARG BUILDER_IMAGE=nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG FORCE_CUDA=1
ARG TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;8.9;9.0"

FROM ${BUILDER_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ARG TORCH_INDEX_URL
ARG FORCE_CUDA
ARG TORCH_CUDA_ARCH_LIST

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=${FORCE_CUDA} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        git \
        ninja-build \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.12 -m venv "$VIRTUAL_ENV"

COPY requirements.txt .

RUN uv pip install --no-cache --python "$VIRTUAL_ENV/bin/python" \
        --extra-index-url "${TORCH_INDEX_URL}" \
        --index-strategy unsafe-best-match \
        -r requirements.txt

RUN git init /tmp/detectron2 && \
    cd /tmp/detectron2 && \
    git remote add origin https://github.com/facebookresearch/detectron2 && \
    git fetch --depth 1 origin b599f139756bd3646a26a909caf86a1a159e53a7 && \
    git checkout FETCH_HEAD && \
    uv pip install --no-cache --no-build-isolation \
        --python "$VIRTUAL_ENV/bin/python" . && \
    rm -rf /tmp/detectron2

RUN uv pip install --no-cache --python "$VIRTUAL_ENV/bin/python" pycocotools==2.0.11


FROM ubuntu:24.04 AS runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/models/.cache/huggingface \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_VERBOSITY=error \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        adduser \
        ca-certificates \
        python3.12 \
        ffmpeg \
        libgomp1 \
        libsm6 \
        libxext6 \
        ocrmypdf \
        pandoc \
        pdftohtml \
        qpdf \
        tesseract-ocr-ara \
        tesseract-ocr-chi-sim \
        tesseract-ocr-deu \
        tesseract-ocr-ell \
        tesseract-ocr-fra \
        tesseract-ocr-hin \
        tesseract-ocr-kor \
        tesseract-ocr-kor-vert \
        tesseract-ocr-mya \
        tesseract-ocr-rus \
        tesseract-ocr-spa \
        tesseract-ocr-tam \
        tesseract-ocr-tha \
        tesseract-ocr-tur \
        tesseract-ocr-ukr \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/src /app/models && \
    addgroup --system python && \
    adduser --system --group --home /app python && \
    chown -R python:python /app

WORKDIR /app

COPY --from=builder --chown=python:python /app/.venv /app/.venv
COPY --chown=python:python --chmod=755 ./start.sh ./start.sh
COPY --chown=python:python --chmod=755 ./start-gpu.sh ./start-gpu.sh
COPY --chown=python:python ./src/download_models.py ./src/download_models.py
COPY --chown=python:python ./src/configuration.py ./src/configuration.py

USER python

RUN python src/download_models.py
COPY --chown=python:python ./src/. ./src