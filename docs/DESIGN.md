# F1 RAG with Real Evaluation Harness — Design

## Context

We are building a Formula 1 question-answering RAG system whose **primary differentiator is a rigorous, per-bucket evaluation harness** showing measurable quality gains across four progressively more sophisticated retrieval architectures (naive → hybrid → +rerank → +router). The build emphasizes:

- A clean, reproducible `docker compose up` experience — reviewers only need Docker installed.
- A visible, well-instrumented eval pipeline — the "naive → router" comparison and the per-bucket breakdown are the central deliverable.
- All retrieval and embedding fully local; LLM calls are the one external dependency.

## Locked decisions

| Area              | Decision                                                                 |
| ----------------- | ------------------------------------------------------------------------ |
| Goal              | Eval-driven F1 RAG demo                                                  |
| UI                | FastAPI + Jinja + HTMX + Tailwind CDN (single process serves UI + API)   |
| Embeddings        | BGE-large-en-v1.5 (local, 1024-dim)                                      |
| Reranker          | bge-reranker-v2-m3 (local)                                               |
| Vector store      | Qdrant (in compose; native hybrid sparse+dense)                          |
| Router LLM        | Claude Haiku 4.5 (tool-use)                                              |
| Synthesis LLM     | Claude Sonnet 4.6                                                        |
| Tracing           | Self-hosted Langfuse in compose (+ Postgres)                             |
| Data era          | Modern era only (2014+)                                                  |
| Wikipedia corpus  | Scraped at image build, baked into image                                 |
| Stats DB          | Jolpica/Ergast → SQLite, materialized at image build, baked into image   |
| FIA regs          | 2026 Sporting + Technical + Financial PDFs, pypdf text extraction, baked |
| OpenF1            | Skipped                                                                  |
| Eval set          | 30 questions, 6 buckets, manually authored gold answers, frozen upfront  |
| Grading           | Exact/numeric matcher for factual buckets + LLM-judge (Sonnet) tiebreak  |
| Version UX        | All 4 versions selectable at runtime via dropdown                        |
| Tests / CI        | Pytest units + eval runner as integration; GitHub Actions builds image   |

## Architecture

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

**Three retrievers behind the router tool-use:**

1. `query_stats(sql)` — Haiku composes SQL against the baked SQLite of Jolpica data. Tables follow the Ergast schema (races, results, drivers, constructors, qualifying, pitstops, laptimes).
2. `search_narratives(query)` — hybrid retrieval (BM25 + dense) over Wikipedia race reports, then bge-reranker-v2-m3 over the top-N. Chunks are paragraph-level with race-context metadata (year, round, GP name).
3. `search_regulations(query)` — same hybrid+rerank stack over the three 2026 FIA PDFs. Chunks include article number metadata.

**Pipeline versions** (selectable via UI dropdown):

- **v1 naive**: dense-only retrieval over a single unified collection (no router, no rerank). Baseline.
- **v2 hybrid**: BM25 + dense fusion over a single unified collection, no rerank, no router.
- **v3 +rerank**: hybrid + bge-reranker-v2-m3, still single collection.
- **v4 +router**: separate collections per source; Haiku picks retriever(s) via tool-use; hybrid+rerank applied within each.

## Eval set (30 questions, frozen)

Six buckets, 5 questions each:

1. **Factual stats** ("Who won the 2017 Australian GP?") — graded by exact string + canonical-name normalization.
2. **Multi-hop** ("Who was Verstappen's teammate the year he won his first title?") — graded by exact match on the entity; LLM-judge confirms reasoning if exact fails.
3. **Narrative** ("What was the controversy at the 2021 Abu Dhabi GP?") — graded by LLM-judge against a manual gold-answer rubric.
4. **Regulation** ("What's the minimum weight of an F1 car in the 2026 regulations?") — graded by numeric matcher (with units normalization) + article-number citation check.
5. **Temporal/comparative** ("How did Mercedes' qualifying pace in 2022 vs 2021?") — graded by LLM-judge against a structured rubric (directionality + magnitude + reasoning).
6. **Adversarial** ("Who won the 2020 European Grand Prix?" — there wasn't one) — graded by refusal detection (must abstain or say "no such race"). Tests grounding/refusal behavior.

Additionally **RAGAS metrics** (faithfulness, context precision, context recall, answer relevancy) reported per-version per-bucket.

## Repo structure

```
f1_rag/
├── docker-compose.yml          # api, qdrant, langfuse, postgres (for langfuse)
├── Dockerfile                  # multi-stage: ingestion → runtime
├── pyproject.toml              # uv/poetry
├── ingestion/
│   ├── stats_etl.py            # Jolpica → SQLite
│   ├── wikipedia_etl.py        # GP race-report scraper
│   ├── regulations_etl.py      # FIA PDF → chunks
│   └── build_indexes.py        # populate Qdrant collections
├── app/
│   ├── main.py                 # FastAPI app + Jinja UI
│   ├── retrievers/
│   │   ├── stats.py            # query_stats tool
│   │   ├── narratives.py       # search_narratives tool
│   │   └── regulations.py      # search_regulations tool
│   ├── pipelines/
│   │   ├── v1_naive.py
│   │   ├── v2_hybrid.py
│   │   ├── v3_rerank.py
│   │   └── v4_router.py
│   ├── synth.py                # Sonnet synthesis
│   └── tracing.py              # langfuse wiring
├── eval/
│   ├── questions.yaml          # 30 gold Q+A with bucket + grading hints
│   ├── graders.py              # exact/numeric/judge graders
│   ├── runner.py               # runs all versions × all questions
│   └── ragas_runner.py         # RAGAS metrics over the same set
├── templates/                  # Jinja UI
├── static/
└── tests/
    ├── test_retrievers.py
    ├── test_router.py
    └── test_graders.py
```

## Verification (end-to-end demo path)

1. `git clone && docker compose up --build` — builds image (scrapes Wikipedia, downloads Jolpica, fetches FIA PDFs, builds indexes), boots Qdrant, Postgres, Langfuse, and the FastAPI app.
2. Browse `http://localhost:8000` — chat UI loads. Pick version "v4 router" from dropdown, ask "Who won the 2021 Brazilian GP?" — get an answer; trace visible at `http://localhost:3000` (Langfuse).
3. Browse `http://localhost:8000/eval` — eval dashboard shows two views: overall RAGAS scores across v1→v4, and a per-bucket breakdown. Click a question to drill into the trace.
4. `docker compose exec api python -m eval.runner` re-runs the eval set; results materialize to `eval/results/*.json` and re-render in dashboard.
5. `pytest` runs unit tests for retrievers, router, and graders.

## Open items (defaults chosen, change before implementation if needed)

- **Synthesis prompt design** — single prompt template with citation requirement vs. version-specific prompts. Default: single template that requires citing source spans.
- **Wikipedia scope precision** — just `*_Grand_Prix` pages, or also season summaries + driver-season pages? Default: GP pages only; add season summaries if multi-hop questions need them.
- **Qdrant collection layout** — single collection with `source` filter vs. three collections. Default: three collections (cleaner for the router story).
- **Langfuse port choice** and whether to also self-host its ClickHouse, or use the simpler Postgres-only mode. Default: Postgres-only.
