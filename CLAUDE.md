# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

F1 RAG comparing four progressively more sophisticated retrieval architectures (`v1` naive → `v2` hybrid → `v3` +rerank → `v4` +router) against a frozen 30-question eval set. Containerized via docker compose: FastAPI + Qdrant + self-hosted Langfuse + Postgres. README.md has the high-level architecture, pipeline-version explanations, and the locked-decisions table; read it first.

## Commands

```bash
make venv         # create .venv + install dev deps (Python 3.11+)
make test         # pytest -q  (283 tests; all mock Qdrant/embedders/Anthropic)
make lint         # ruff check .
make fmt          # ruff check . --fix

make up           # docker compose up --build (foreground)
make up-d         # detached
make down         # stop the stack
make logs         # tail compose logs
make gpu-check    # verify nvidia-container-toolkit passes a GPU through to docker
make index        # rebuild Qdrant indexes from host (uses host venv → GPU if available)
make eval         # docker compose exec api python -m eval.runner  (runs in container)
make eval-local   # python -m eval.runner from host venv against dockerized Qdrant
```

Single test: `.venv/bin/pytest tests/test_pipeline_v3.py::TestRerankPipeline::test_rerank_called_with_query_and_texts -q`.

In-container exec: `docker compose exec api python -m ingestion.build_indexes --force`.

## Architecture you can't see from a single file

**Pipelines share one contract.** `app/pipeline.py` defines `PipelineInput`, `PipelineResult`, and `ToolCall`. All four pipelines (`app/pipelines/v{1..4}_*.py`) consume the same input and return the same result shape — the UI, eval runner, and Langfuse tracing all treat them uniformly. Add a new version by following the same shape; don't introduce a side-channel.

**Single embedding/reranker singleton.** `app/embedding.py` holds `Embedders` (BGE-large dense + Qdrant/bm25 sparse via FastEmbed) and `Reranker` (bge-reranker-v2-m3 via sentence-transformers `CrossEncoder` — not FlagEmbedding, see Gotchas). Both are lazy-loaded on first call and auto-pick `cuda` if `torch.cuda.is_available()`. `_pick_device()` is the only place to change device logic; honor `F1RAG_DEVICE` env override.

**Single synthesizer.** `app/synth.py` is the only place that calls Anthropic for answer generation (Sonnet 4.6). It expects `ContextChunk` objects with a `cite_id`; it instructs Sonnet to emit `[cite:cite_id]` markers and extracts them post-hoc with `CITE_PATTERN`. All four pipelines call it.

**The v4 router is a Haiku tool-loop, not a generator.** `app/pipelines/v4_router.py` runs Haiku 4.5 with the three retrievers as tools, accumulates contexts across up to `MAX_STEPS=8` tool calls, then discards Haiku's final text and passes the **original user query** plus the collected contexts to `Synthesizer.synthesize`. Haiku routes; Sonnet writes.

**The stats DB schema is snake_case, NOT Ergast camelCase.** Ergast docs (and Haiku's training) use `raceId`, `driverId`. Our ETL writes `race_id`, `driver_id`. The v4 router's system prompt has the exact snake_case schema spelled out — if you regenerate the prompt or change column names, keep them in sync or the router will produce broken SQL on every stats question.

**Three Qdrant collections, hybrid retrieval.** Each collection (`stats_meta`, `narratives`, `regulations`) holds points with **two** vectors: `dense` (1024-dim cosine) and `sparse` (BM25 with IDF modifier). Retrievers issue `client.query_points(prefetch=[Prefetch(dense), Prefetch(sparse)], query=FusionQuery(Fusion.RRF))`. Single-vector queries (v1) just use `query=dense_vec, using="dense"`.

**Tracing is non-invasive.** `app/tracing.py` exposes `@traced` (a thin wrapper around `langfuse.decorators.observe`). It's a no-op when `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are unset — that's how all 283 tests run without Langfuse. Decorators are sprinkled on retriever `.run()`/`.search()`, `Synthesizer.synthesize`, and each pipeline's `.run()` — don't change signatures when adding them.

**Eval runner is dependency-injected.** `eval/runner.py`'s `run_eval(runners=...)` takes a dict of pipeline callables so tests can pass mocks. The default `_default_runners()` lazily imports `run_v1`/`run_v2`/`run_v3`/`run_v4` to avoid model loads at import time. Graders dispatch by `question.grading.kind` (exact_match / numeric / llm_judge / refusal); the LLM judge calls Sonnet with structured-output JSON.

## Container lifecycle

`scripts/entrypoint.sh` is the container entrypoint:
1. Waits for Qdrant via TCP probe.
2. Runs any missing ETLs (`ingestion.{stats,wikipedia,regulations}_etl`) — file-existence guards so populated images skip this.
3. Backgrounds `ingestion.build_indexes` (idempotent; skips populated collections). The HTTP server doesn't block on indexing.
4. Execs uvicorn on port 8000.

Data flow: ETLs write to `/app/data/` (host-mounted via the `data/` bind during dev / baked into the image via `COPY data ./data` from the build context — see Gotchas). `eval/results/` is host-mounted so eval runs persist across container rebuilds.

## Gotchas (learned the hard way)

- **`data/.gitkeep` is intentional.** `data/` is otherwise gitignored. The Dockerfile does `COPY data ./data` which needs the directory to exist in the build context — `.gitkeep` keeps it present in fresh clones (and in CI). Don't delete it.
- **Don't run ETLs inside `docker build`.** Jolpica rate-limits will exhaust tenacity retries on a ~12-season pull. Keep the ETLs in the entrypoint's lazy fallback or run them from the host venv before the build.
- **The cu130 torch wheel is required for Blackwell GPUs** (RTX 50-series, sm_120). cu121 only goes up to sm_90 (Hopper). The Dockerfile pins `pip install torch --index-url https://download.pytorch.org/whl/cu130` before `pip install -e .` so the editable install doesn't replace it with CPU torch from PyPI. Image is ~9.5 GB; CI sanity cap is 10 GB.
- **Reranker uses sentence-transformers `CrossEncoder`, not FlagEmbedding's `FlagReranker`.** FlagEmbedding 1.4 calls `tokenizer.prepare_for_model` which transformers 5.x removed; swapping caused 30/30 v3 failures and 13/30 v4 failures in the first eval run. Don't switch back.
- **`LANGFUSE_USER_EMAIL` must be RFC-valid.** Langfuse v2 uses Zod email validation — `admin@local` fails, `admin@example.com` works. The default in `.env.example` reflects this.
- **Qdrant client / server version drift** prints a warning at startup (`Qdrant client version 1.18.0 is incompatible with server version 1.12.5`). It's noise — major versions don't differ and the API surface we use is stable. Suppress if it gets annoying via `QdrantClient(..., check_compatibility=False)`.
- **`Synthesizer.synthesize` returns `refused=True` on empty context.** It also detects refusal phrases (`"I don't have"`, `"no such race"`, etc.) in the answer. Both are used by the adversarial-bucket grader.
- **The Anthropic API 529s under load.** The eval runner records them as `RunError` rather than crashing; check `eval/results/latest.json` for `error` fields after long runs.
- **GPU passthrough requires `nvidia-container-toolkit` on the host.** Without it, the `deploy.resources.reservations.devices` block in `docker-compose.yml` triggers a warning but compose still starts the container — `_pick_device()` falls back to CPU. Run `make gpu-check` to validate.

## What not to touch without good reason

- `pyproject.toml` torch pin: not pinned. The Dockerfile uses `--index-url https://download.pytorch.org/whl/cu130` for the container; the host venv resolves separately. Pinning here without coordinating both sides creates skew.
- `app/pipeline.py` shapes: changing `PipelineResult` ripples through UI, eval runner, dashboard JSON, and Langfuse traces.
- `eval/questions.yaml`: the eval set is frozen. Changing questions invalidates cross-run comparisons. If you need new questions for a future eval, add a separate file or a `held_out:` section rather than editing existing IDs.
- v4 router's stats-tool schema description: must stay in sync with the actual SQLite columns (snake_case).
