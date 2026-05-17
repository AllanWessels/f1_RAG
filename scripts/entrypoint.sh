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

# Fallback ETLs: if the image was built without a populated data/ dir,
# run the ingestion locally on first start. Each ETL is idempotent.
if [ ! -f /app/data/f1_stats.sqlite ]; then
    echo "[entrypoint] data/f1_stats.sqlite missing — running stats ETL"
    python -m ingestion.stats_etl || echo "[entrypoint] stats_etl failed (continuing)"
fi
if [ ! -f /app/data/wikipedia_races.jsonl ]; then
    echo "[entrypoint] data/wikipedia_races.jsonl missing — running Wikipedia ETL"
    python -m ingestion.wikipedia_etl || echo "[entrypoint] wikipedia_etl failed (continuing)"
fi
if [ ! -f /app/data/fia_regulations.jsonl ]; then
    echo "[entrypoint] data/fia_regulations.jsonl missing — running FIA regs ETL"
    python -m ingestion.regulations_etl || echo "[entrypoint] regulations_etl failed (continuing)"
fi

echo "[entrypoint] kicking off Qdrant indexing in the background (idempotent) ..."
# Indexing 2k+ chunks with BGE-large on CPU takes ~30 min, but it's idempotent
# and chunks land in Qdrant incrementally — so we don't block the HTTP server
# on it. Until indexing catches up, retrievers will return whatever's already
# indexed, which is correct behavior. Logs go to /tmp.
nohup python -m ingestion.build_indexes >/tmp/build_indexes.log 2>&1 &

echo "[entrypoint] starting uvicorn on 0.0.0.0:8000 ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
