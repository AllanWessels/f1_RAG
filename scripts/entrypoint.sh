#!/usr/bin/env bash
# Wait for Qdrant, build indexes if needed, then start uvicorn.
# Models (BGE + bge-reranker-v2-m3) are large and download lazily on first use
# into the mounted HuggingFace cache volume, so the first start is slow but
# subsequent starts hit cache.

set -euo pipefail

QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"

echo "[entrypoint] waiting for Qdrant at $QDRANT_URL ..."
python - <<PY
import os, socket, time, urllib.parse
url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
parsed = urllib.parse.urlparse(url)
host, port = parsed.hostname, parsed.port or 6333
for i in range(60):
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"[entrypoint] Qdrant reachable at {host}:{port}")
            raise SystemExit(0)
    except OSError:
        time.sleep(1)
print(f"[entrypoint] Qdrant never came up at {host}:{port}", flush=True)
raise SystemExit(1)
PY

echo "[entrypoint] building Qdrant indexes (idempotent; skips populated collections) ..."
python -m ingestion.build_indexes || {
    echo "[entrypoint] index build failed; continuing anyway so /health stays up"
}

echo "[entrypoint] starting uvicorn on 0.0.0.0:8000 ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
