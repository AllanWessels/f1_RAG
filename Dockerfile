# syntax=docker/dockerfile:1.7

#############################
# Stage 1: builder
# - installs Python dependencies
# - runs ingestion ETLs to bake data into the image
#############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY ingestion ./ingestion
COPY app ./app
COPY eval ./eval

RUN pip install --upgrade pip && pip install -e .
RUN mkdir -p /build/data

# Bring the three corpora into the image. Two paths:
#   1. If `data/` is populated locally (recommended; run `make ingest` first),
#      it is copied in directly.
#   2. If `data/` is empty, the entrypoint will lazily run the ETLs against
#      the live APIs on container start.
# We avoid running ETLs inside `docker build` because the Jolpica and FIA
# rate limits are unreliable for ~3000 sequential requests under buildkit's
# network namespace; the previous design hit RetryError on Jolpica 429s.
COPY data ./data


#############################
# Stage 2: runtime
#############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY pyproject.toml ./
COPY app ./app
COPY ingestion ./ingestion
COPY eval ./eval
COPY templates ./templates
COPY static ./static
COPY scripts ./scripts
COPY --from=builder /build/data ./data

EXPOSE 8000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
