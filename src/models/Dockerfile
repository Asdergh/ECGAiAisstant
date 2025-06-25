# ---- базовый слой ----
FROM python:3.11-slim AS base

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

# Копируем только requirements.txt — чтобы использовать кэш, если он не менялся
COPY requirements.txt /srv

# Устанавливаем зависимости — кэшируется отдельно
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем остальной код (этот слой уже не сбрасывает кэш зависимостей)
COPY . /srv

# При желании можно добавить:
# CMD ["python3", "api.py"]

