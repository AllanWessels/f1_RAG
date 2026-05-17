# F1 RAG

A Formula 1 question-answering system with **four progressively more sophisticated retrieval architectures** (`v1` naive → `v2` hybrid → `v3` +rerank → `v4` +router) and a real per-bucket evaluation harness comparing them across 30 frozen questions.

The whole system is containerized: `docker compose up --build` and you have the chat UI, the eval dashboard, and self-hosted Langfuse tracing.

See [docs/DESIGN.md](docs/DESIGN.md) for the system design, [docs/EXECUTION.md](docs/EXECUTION.md) for the milestone plan, and [docs/](docs/) for the rest.

## Architecture

- **Three corpora** baked into the image at build time:
  - **Stats** — Jolpica-F1 (Ergast successor) results, qualifying, pit stops for 2014–2025 → SQLite.
  - **Narratives** — Wikipedia race-report articles for every Grand Prix 2014–2025 → JSONL chunks.
  - **Regulations** — FIA 2026 Sporting / Technical / Financial regulation PDFs → article-chunked JSONL.
- **Hybrid retrieval** via Qdrant (BGE-large-en-v1.5 dense + BM25 sparse, native fusion).
- **Reranking** via bge-reranker-v2-m3.
- **Routing** via Claude Haiku 4.5 with tool-use over the three retrievers.
- **Synthesis** via Claude Sonnet 4.6 with structured `[cite:ID]` citations and refusal detection.
- **Tracing** via self-hosted Langfuse (v2, Postgres-only mode).

## Quickstart

You need Docker and an Anthropic API key.

```bash
cp .env.example .env
$EDITOR .env   # set ANTHROPIC_API_KEY at minimum

docker compose up --build
```

On first start the api container will:

1. Wait for Qdrant.
2. Lazily download the BGE-large embedding model + bge-reranker-v2-m3 into the `hf_cache` volume (~1.5 GB total; one-time download cached for future starts).
3. Build three Qdrant collections from the baked corpora (idempotent — skips if already populated).
4. Start uvicorn on `:8000`.

Then visit:

| URL                           | What                                         |
| ----------------------------- | -------------------------------------------- |
| http://localhost:8000         | Chat UI — pick a version (v1/v2/v3/v4) and ask |
| http://localhost:8000/eval    | Eval dashboard — version × bucket performance |
| http://localhost:3000         | Langfuse — full trace per query              |

Example queries:

- *"Who won the 2017 Australian Grand Prix?"* (factual stats — v4 routes to `query_stats`)
- *"What was the controversy at the 2021 Abu Dhabi GP?"* (narrative — v4 routes to `search_narratives`)
- *"What's the minimum weight of an F1 car in the 2026 regulations?"* (regulation — v4 routes to `search_regulations`)
- *"Who won the 2020 European Grand Prix?"* (adversarial — there wasn't one; the system should decline)

## Running the eval harness

The eval set is `eval/questions.yaml`: 30 questions across 6 buckets (factual stats, multi-hop, narrative, regulation, temporal/comparative, adversarial).

```bash
docker compose exec api python -m eval.runner
```

This runs 4 versions × 30 questions = 120 grades, writes results to `eval/results/<timestamp>.json` and `eval/results/latest.json`, and the `/eval` dashboard renders the per-version per-bucket breakdown.

Grader kinds:
- **Exact-match** (factual stats): canonicalize and look for accepted variants.
- **Numeric** (regulations): regex-extract numbers + units, normalize, compare with tolerance; require article citation if specified.
- **LLM-judge** (narrative, multi-hop, temporal): Claude Sonnet scores against a rubric.
- **Refusal** (adversarial): detect `refused=True` or refusal phrasing.

Plus optional **RAGAS** metrics (faithfulness, context precision/recall, answer relevancy).

## Development

```bash
make venv      # create .venv + install dev deps
make test      # run pytest (283 tests)
make lint      # ruff
make up        # docker compose up --build
make eval      # run the eval harness in the running container
```

The package layout:

```
app/                       # runtime: pipelines, retrievers, synth, tracing, UI
  pipelines/v{1..4}_*.py   # four pipeline versions, selectable at runtime
  retrievers/*.py          # three retriever tools (stats / narratives / regs)
  embedding.py             # shared lazy Embedders + Reranker
  synth.py                 # Sonnet synthesis with structured citations
  tracing.py               # Langfuse decorator (no-ops without env keys)
  main.py + templates/     # FastAPI + Jinja + HTMX + Tailwind UI
ingestion/                 # ETLs + Qdrant indexer (run at image build + container start)
eval/                      # graders, runner, RAGAS, the 30 frozen questions
tests/                     # 283 unit tests across all of the above
```

## Configuration (`.env`)

- `ANTHROPIC_API_KEY` — required (router uses Haiku 4.5, synth uses Sonnet 4.6).
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — required for tracing; the compose bootstraps a project with these on first start.
- `API_PORT` / `QDRANT_PORT` / `LANGFUSE_PORT` — port overrides if 8000/6333/3000 are taken.
- `F1RAG_DEVICE` — force `cpu` or `cuda`. By default the embedder/reranker auto-detect CUDA and fall back to CPU.

Tracing is a transparent no-op if the Langfuse keys are unset, so unit tests and local dev runs don't need Langfuse.

## GPU acceleration

The host venv auto-uses CUDA if `torch.cuda.is_available()` is true. Indexing on a single NVIDIA RTX 5080 finishes ~4k chunks in ~3 minutes (≈50× faster than CPU). To rebuild the Qdrant indexes from the host (uses GPU, hits the dockerized Qdrant):

```bash
QDRANT_URL=http://localhost:6333 make index    # or: ... python -m ingestion.build_indexes --force
```

For GPU **inside** docker compose, you also need [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on the host and CUDA-enabled PyTorch in the image; the default Dockerfile uses CPU torch so the in-container indexing path is slow. The recommended workflow is: bring up the compose stack, then run `make index` from the host once for a fast cold start.

## License

MIT.
