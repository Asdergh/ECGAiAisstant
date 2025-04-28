# ---- базовый слой ----
FROM python:3.11-slim AS base

# Poetry/pip cache ускорит последующие билды
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY . /srv
RUN pip install --upgrade pip && pip install -r requirements.txt
