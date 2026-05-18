# F1 RAG

A Formula 1 question-answering system with **four progressively more sophisticated retrieval architectures** (`v1` naive → `v2` hybrid → `v3` +rerank → `v4` +router) and a real per-bucket evaluation harness comparing them across 30 frozen questions.

The whole system is containerized: `docker compose up --build` and you have the chat UI, the eval dashboard, and self-hosted Langfuse tracing.

The primary differentiator is the **rigorous, per-bucket evaluation harness** showing measurable quality gains across the four retrieval architectures. The build emphasizes:

- A clean, reproducible `docker compose up` experience — reviewers only need Docker installed.
- A visible, well-instrumented eval pipeline — the "naive → router" comparison and the per-bucket breakdown are the central deliverable.
- All retrieval and embedding fully local; LLM calls are the one external dependency.

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

```
                    ┌──────────────────────────────────────┐
                    │  FastAPI (Jinja+HTMX UI + JSON API)  │
                    │  /chat  /eval/results  /traces       │
                    └─────────────┬────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │   Pipeline (version-selectable)   │
                │   v1 naive  v2 hybrid             │
                │   v3 +rerank  v4 +router          │
                └─┬───────┬───────┬──────────────┬──┘
                  │       │       │              │
        ┌─────────▼┐  ┌───▼───┐ ┌─▼─────────┐  ┌─▼────────────────┐
        │  Qdrant  │  │ BM25  │ │  Local    │  │  Claude Haiku    │
        │ (dense)  │  │(sparse│ │  rerank   │  │  router (tools)  │
        │          │  │ in QD)│ │  bge-v2m3 │  │  → which retr?   │
        └──────────┘  └───────┘ └───────────┘  └────────┬─────────┘
                                                        │
                            ┌───────────────────────────┼───────────────┐
                            │                           │               │
                  ┌─────────▼──────┐         ┌──────────▼──────┐  ┌─────▼──────┐
                  │ query_stats    │         │ search_         │  │ search_    │
                  │ (SQLite SQL)   │         │ narratives      │  │ regulations│
                  │ Jolpica clone  │         │ (Wikipedia race │  │ (FIA PDFs) │
                  │                │         │  reports)       │  │            │
                  └────────────────┘         └─────────────────┘  └────────────┘

                  All retrievers → context → Claude Sonnet synth → answer
                  Every step traced to self-hosted Langfuse
```

### Three retrievers behind the router tool-use

1. **`query_stats(sql)`** — Haiku composes SQL against the baked SQLite of Jolpica data. Tables follow the Ergast schema (races, results, drivers, constructors, qualifying, pitstops, laptimes).
2. **`search_narratives(query)`** — hybrid retrieval (BM25 + dense) over Wikipedia race reports, then bge-reranker-v2-m3 over the top-N. Chunks are paragraph-level with race-context metadata (year, round, GP name).
3. **`search_regulations(query)`** — same hybrid+rerank stack over the three 2026 FIA PDFs. Chunks include article number metadata.

### Pipeline versions (selectable via UI dropdown)

- **v1 naive** — dense-only retrieval over a single unified collection (no router, no rerank). Baseline.
- **v2 hybrid** — BM25 + dense fusion over a single unified collection, no rerank, no router.
- **v3 +rerank** — hybrid + bge-reranker-v2-m3, still single collection.
- **v4 +router** — separate collections per source; Haiku picks retriever(s) via tool-use; hybrid+rerank applied within each.

### Locked decisions

| Area              | Decision                                                                 |
| ----------------- | ------------------------------------------------------------------------ |
| Goal              | Eval-driven F1 RAG demo                                                  |
| UI                | FastAPI + Jinja + HTMX + Tailwind CDN (single process serves UI + API)   |
| Embeddings        | BGE-large-en-v1.5 (local, 1024-dim)                                      |
| Reranker          | bge-reranker-v2-m3 (local, via sentence-transformers CrossEncoder)       |
| Vector store      | Qdrant (in compose; native hybrid sparse+dense)                          |
| Router LLM        | Claude Haiku 4.5 (tool-use)                                              |
| Synthesis LLM     | Claude Sonnet 4.6                                                        |
| Tracing           | Self-hosted Langfuse in compose (+ Postgres)                             |
| Data era          | Modern era only (2014+)                                                  |
| Wikipedia corpus  | Scraped locally, copied into image                                       |
| Stats DB          | Jolpica/Ergast → SQLite, materialized locally, copied into image         |
| FIA regs          | 2026 Sporting + Technical + Financial PDFs, pypdf text extraction        |
| OpenF1            | Skipped                                                                  |
| Eval set          | 30 questions, 6 buckets, manually authored gold answers, frozen upfront  |
| Grading           | Exact/numeric matcher for factual buckets + LLM-judge (Sonnet) tiebreak  |
| Version UX        | All 4 versions selectable at runtime via dropdown                        |
| Tests / CI        | Pytest units + eval runner as integration; GitHub Actions builds image   |

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

### Example queries

- *"Who won the 2017 Australian Grand Prix?"* (factual stats — v4 routes to `query_stats`)
- *"What was the controversy at the 2021 Abu Dhabi GP?"* (narrative — v4 routes to `search_narratives`)
- *"What's the minimum weight of an F1 car in the 2026 regulations?"* (regulation — v4 routes to `search_regulations`)
- *"Who won the 2020 European Grand Prix?"* (adversarial — there wasn't one; the system should decline)

### End-to-end demo path

1. `git clone && docker compose up --build` — builds image, boots Qdrant, Postgres, Langfuse, and the FastAPI app.
2. Browse `http://localhost:8000` — chat UI loads. Pick version "v4 router" from the dropdown, ask "Who won the 2021 Brazilian GP?" — get an answer; trace visible at `http://localhost:3000` (Langfuse, login `admin@example.com / changeme`).
3. Browse `http://localhost:8000/eval` — eval dashboard shows two views: overall scores across v1→v4, and a per-bucket breakdown. Click a question to drill into the trace.
4. `docker compose exec api python -m eval.runner` re-runs the eval set; results materialize to `eval/results/*.json` and re-render in the dashboard.
5. `pytest` runs unit tests for retrievers, router, and graders.

## Eval set (30 questions, frozen)

Six buckets, 5 questions each:

1. **Factual stats** ("Who won the 2017 Australian GP?") — graded by exact string + canonical-name normalization.
2. **Multi-hop** ("Who was Verstappen's teammate the year he won his first title?") — graded by exact match on the entity; LLM-judge confirms reasoning if exact fails.
3. **Narrative** ("What was the controversy at the 2021 Abu Dhabi GP?") — graded by LLM-judge against a manual gold-answer rubric.
4. **Regulation** ("What's the minimum weight of an F1 car in the 2026 regulations?") — graded by numeric matcher (with units normalization) + article-number citation check.
5. **Temporal/comparative** ("How did Mercedes' qualifying pace in 2022 vs 2021?") — graded by LLM-judge against a structured rubric (directionality + magnitude + reasoning).
6. **Adversarial** ("Who won the 2020 European Grand Prix?" — there wasn't one) — graded by refusal detection (must abstain or say "no such race"). Tests grounding/refusal behavior.

Additionally **RAGAS metrics** (faithfulness, context precision, context recall, answer relevancy) reported per-version per-bucket.

### Running the eval harness

The eval set is `eval/questions.yaml`. To run:

```bash
docker compose exec api python -m eval.runner
```

This runs 4 versions × 30 questions = 120 grades, writes results to `eval/results/<timestamp>.json` and `eval/results/latest.json`, and the `/eval` dashboard renders the per-version per-bucket breakdown.

Grader kinds:
- **Exact-match** (factual stats): canonicalize and look for accepted variants.
- **Numeric** (regulations): regex-extract numbers + units, normalize, compare with tolerance; require article citation if specified.
- **LLM-judge** (narrative, multi-hop, temporal): Claude Sonnet scores against a rubric.
- **Refusal** (adversarial): detect `refused=True` or refusal phrasing.

## Development

```bash
make venv      # create .venv + install dev deps
make test      # run pytest (283 tests)
make lint      # ruff
make up        # docker compose up --build
make eval      # run the eval harness in the running container
make index     # rebuild Qdrant indexes (uses GPU on host if available)
make down      # stop the compose stack
```

### Repo structure

```
f1_rag/
├── docker-compose.yml          # api, qdrant, langfuse, postgres
├── Dockerfile                  # multi-stage: builder → runtime
├── scripts/entrypoint.sh       # waits for Qdrant, runs missing ETLs, backgrounds index build, execs uvicorn
├── pyproject.toml
├── Makefile
├── ingestion/
│   ├── stats_etl.py            # Jolpica → SQLite (2014+)
│   ├── wikipedia_etl.py        # GP race-report scraper → JSONL
│   ├── regulations_etl.py      # FIA PDF → article chunks → JSONL
│   └── build_indexes.py        # populate the 3 Qdrant collections
├── app/
│   ├── main.py                 # FastAPI + Jinja UI + HTMX endpoints
│   ├── retrievers/
│   │   ├── stats.py            # query_stats tool (read-only SQL)
│   │   ├── narratives.py       # search_narratives (hybrid + rerank)
│   │   └── regulations.py      # search_regulations (hybrid + rerank)
│   ├── pipelines/
│   │   ├── v1_naive.py
│   │   ├── v2_hybrid.py
│   │   ├── v3_rerank.py
│   │   └── v4_router.py
│   ├── embedding.py            # shared lazy Embedders + Reranker (CUDA-aware)
│   ├── synth.py                # Sonnet synthesis with [cite:ID] markers
│   ├── pipeline.py             # PipelineInput / PipelineResult / ToolCall
│   └── tracing.py              # Langfuse decorator (no-ops without env keys)
├── eval/
│   ├── questions.yaml          # the 30 frozen gold Q+A
│   ├── schema.py               # Pydantic discriminated-union validator
│   ├── graders.py              # exact / numeric / LLM-judge / refusal
│   ├── runner.py               # runs all versions × all questions
│   ├── ragas_runner.py         # RAGAS metrics
│   └── results/                # eval JSON output (host-mounted)
├── templates/                  # Jinja UI (chat + dashboard)
├── static/
└── tests/                      # 283 unit tests
```

## Configuration (`.env`)

- `ANTHROPIC_API_KEY` — required (router uses Haiku 4.5, synth uses Sonnet 4.6).
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — required for tracing; the compose bootstraps a project with these on first start.
- `LANGFUSE_USER_EMAIL` / `LANGFUSE_USER_PASSWORD` — Langfuse bootstrap account; the email must be RFC-valid (defaults to `admin@example.com / changeme`).
- `API_PORT` / `QDRANT_PORT` / `LANGFUSE_PORT` — port overrides if 8000/6333/3000 are taken.
- `F1RAG_DEVICE` — force `cpu` or `cuda`. By default the embedder/reranker auto-detect CUDA and fall back to CPU.

Tracing is a transparent no-op if the Langfuse keys are unset, so unit tests and local dev runs don't need Langfuse.

## GPU acceleration

Two valid paths to get GPU-accelerated embedding/reranking:

### Option A — GPU inside the container (recommended for NVIDIA hosts)

The image ships with **CUDA-enabled PyTorch** (cu121 wheel from the PyTorch wheel index). The `docker-compose.yml` declares an NVIDIA GPU reservation for the `api` service, so the container uses the host GPU automatically once `nvidia-container-toolkit` is installed.

#### 1. Install nvidia-container-toolkit (Debian/Ubuntu)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

For non-Debian distributions see the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

#### 2. Verify GPU passthrough before bringing the stack up

```bash
make gpu-check
```

This runs `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` and exits 0 if GPU passthrough is working.

#### 3. Bring up the stack normally

```bash
docker compose up --build
```

#### 4. Verify GPU visibility inside the running container

```bash
docker compose exec api python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### Image size note

Adding the cu121 PyTorch wheel increases the image from ~2 GB to roughly 3–4 GB (~2.5 GB for the wheel alone). The first `docker compose up --build` will be slower on a fresh pull; subsequent builds are cached.

#### Graceful CPU fallback

On hosts without `nvidia-container-toolkit`, Docker Compose prints a warning that the device reservation cannot be satisfied but **still starts the container**. The api service falls back to CPU automatically via `app.embedding._pick_device`. No flags in the compose file hard-require GPU.

### Option B — GPU from the host venv (fast path, no toolkit needed)

The host venv auto-uses CUDA if `torch.cuda.is_available()` is true. Indexing on a single NVIDIA RTX 5080 finishes ~4k chunks in ~3 minutes (≈50× faster than CPU). Bring up the compose stack first (so Qdrant is reachable), then run indexing from the host:

```bash
QDRANT_URL=http://localhost:6333 make index    # or: python -m ingestion.build_indexes --force
```

This is a good choice when you want GPU speed without installing the container toolkit, or when iterating on ingestion code outside Docker.

## How it was built

The system was built in 14 milestones across 6 execution waves, with Sonnet subagents doing parallel work inside each fan-out wave on disjoint file scopes.

### Milestones

| ID  | Milestone                                | Type        |
| --- | ---------------------------------------- | ----------- |
| M1  | Repo skeleton & tooling                  | Serial      |
| M2  | Ingestion: stats DB (Jolpica → SQLite)   | Parallel    |
| M3  | Ingestion: Wikipedia race reports        | Parallel    |
| M4  | Ingestion: FIA regulations (PDFs)        | Parallel    |
| M5  | Indexing: Qdrant collections + BGE       | Serial      |
| M6  | Retrievers (stats / narratives / regs)   | 3× parallel |
| M7  | Pipelines v1, v2, v3 (non-router)        | 3× parallel |
| M8  | Pipeline v4 (router with tool-use)       | Parallel w/ M7 |
| M9  | Langfuse tracing wired in                | Parallel    |
| M10 | FastAPI + HTMX UI                        | Parallel    |
| M11 | Eval set authoring (30 questions)        | Parallel    |
| M12 | Graders & eval runner                    | Parallel    |
| M13 | Eval dashboard rendering                 | Serial      |
| M14 | Polish, docs, smoke test                 | Serial      |

### Dependency graph

```
M1 ──┬── M2 ──┐
     ├── M3 ──┼── M5 ── M6 ──┬── M7 ──┐
     ├── M4 ──┘              └── M8 ──┴── M9 ── M10 ── M13 ── M14
     └── M11 ────────────────────────────────── M12 ──┘
```

### Execution waves

- **Wave 0 — serial bootstrap** (M1): pyproject, package layout, Dockerfile skeleton, compose, GH Actions, `.env.example`.
- **Wave A — fan-out 4** (after M1): four Sonnet agents implement M2, M3, M4, M11 on disjoint ingestion / eval-authoring files.
- **Wave B — serial choke** (M5): orchestrator writes the Qdrant indexer; needs the three datasets from Wave A.
- **Wave C — fan-out 3** (after M5): three Sonnet agents implement M6a/M6b/M6c (one retriever each).
- **Wave D — fan-out 4** (after M6): pre-step writes `app/synth.py`, then four Sonnet agents implement M7a/M7b/M7c/M8 (four pipelines).
- **Wave E — fan-out 3** (after Wave D): three Sonnet agents implement M9 tracing, M10 UI, M12 eval runner — coordinated so M9 only adds non-invasive decorators while M10/M12 import-only.
- **Wave F — serial** (M13): wire the dashboard to M12's JSON shape.
- **Wave G — serial** (M14): README, Makefile, end-to-end smoke test.

### Subagent protocol

Each Sonnet subagent is briefed with:

1. Pointers to this README and the relevant existing modules.
2. **Strict scope** — the specific files it owns. It must not edit files owned by other agents in the same wave.
3. **Exit criteria** matching the milestone's exit gate.
4. **Test requirement** — new code lands with at least one pytest, unless the milestone is a content task (M11).
5. **No commits** — the orchestrator commits at wave boundaries to keep history clean.

Compression: 14 sequential milestones collapse to ~7–8 wall-clock units once Waves A, C, D, E run in parallel. The serial choke points (M1, M5, M13, M14) are unavoidable.

## License

MIT.
