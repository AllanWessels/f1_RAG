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

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e .

# Copy source; ingestion runs at build time and writes to /build/data
COPY ingestion ./ingestion
COPY app ./app
COPY eval ./eval

RUN mkdir -p /build/data

# Ingestion ETLs — bake the three corpora into the image so the runtime
# container starts with stats SQLite, Wikipedia JSONL, and FIA PDF chunks
# already on disk. Qdrant indexing is deferred to container start
# (Qdrant isn't running during image build).
RUN python -m ingestion.stats_etl
RUN python -m ingestion.wikipedia_etl
RUN python -m ingestion.regulations_etl


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
