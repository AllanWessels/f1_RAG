# Execution Plan — Milestones & Subagent Parallelization

This file lives next to [DESIGN.md](DESIGN.md) and describes **how** we build what's designed there: the 14 milestones, their dependency graph, and which can be parallelized via subagents.

All subagents are spawned with `model: "sonnet"` (Claude Sonnet 4.6) for speed and cost balance; the orchestrator (main thread) is Opus 4.7 with full design context.

## Milestone list

| ID  | Milestone                                | Owner          | Type        |
| --- | ---------------------------------------- | -------------- | ----------- |
| M1  | Repo skeleton & tooling                  | Orchestrator   | Serial      |
| M2  | Ingestion: stats DB (Jolpica → SQLite)   | Sonnet agent   | Parallel    |
| M3  | Ingestion: Wikipedia race reports        | Sonnet agent   | Parallel    |
| M4  | Ingestion: FIA regulations (PDFs)        | Sonnet agent   | Parallel    |
| M5  | Indexing: Qdrant collections + BGE       | Orchestrator   | Serial      |
| M6  | Retrievers (stats / narratives / regs)   | 3× Sonnet      | Parallel    |
| M7  | Pipelines v1, v2, v3 (non-router)        | 3× Sonnet      | Parallel    |
| M8  | Pipeline v4 (router with tool-use)       | Sonnet agent   | Parallel w/ M7 |
| M9  | Langfuse tracing wired in                | Sonnet agent   | Parallel    |
| M10 | FastAPI + HTMX UI                        | Sonnet agent   | Parallel    |
| M11 | Eval set authoring (30 questions)        | Sonnet agent   | Parallel    |
| M12 | Graders & eval runner                    | Sonnet agent   | Parallel    |
| M13 | Eval dashboard rendering                 | Orchestrator   | Serial      |
| M14 | Polish, docs, smoke test                 | Orchestrator   | Serial      |

## Dependency graph

```
M1 ──┬── M2 ──┐
     ├── M3 ──┼── M5 ── M6 ──┬── M7 ──┐
     ├── M4 ──┘              └── M8 ──┴── M9 ── M10 ── M13 ── M14
     └── M11 ────────────────────────────────── M12 ──┘
```

## Execution waves

### Wave 0 — Serial bootstrap (orchestrator)
- **M1**: pyproject, package layout, Dockerfile skeleton, compose, GH Actions, .env.example.

### Wave A — fan-out 4 (after M1)
Disjoint file scope:

| Agent | Milestone | Owns                                |
| ----- | --------- | ----------------------------------- |
| A1    | M2        | `ingestion/stats_etl.py`            |
| A2    | M3        | `ingestion/wikipedia_etl.py`        |
| A3    | M4        | `ingestion/regulations_etl.py`      |
| A4    | M11       | `eval/questions.yaml`               |

### Wave B — Serial choke (orchestrator)
- **M5**: `ingestion/build_indexes.py`, Qdrant collection bootstrap. Needs all three datasets from Wave A.

### Wave C — fan-out 3 (after M5)
Three retrievers, disjoint files:

| Agent | Milestone | Owns                                  |
| ----- | --------- | ------------------------------------- |
| C1    | M6a       | `app/retrievers/stats.py` + test     |
| C2    | M6b       | `app/retrievers/narratives.py` + test |
| C3    | M6c       | `app/retrievers/regulations.py` + test |

### Wave D — fan-out 4 (after M6)
Pre-step: orchestrator writes `app/synth.py` (small, ~15 min) so all four agents only import it.

| Agent | Milestone | Owns                              |
| ----- | --------- | --------------------------------- |
| D1    | M7a       | `app/pipelines/v1_naive.py`       |
| D2    | M7b       | `app/pipelines/v2_hybrid.py`      |
| D3    | M7c       | `app/pipelines/v3_rerank.py`      |
| D4    | M8        | `app/pipelines/v4_router.py`      |

### Wave E — fan-out 3 (after Wave D)
Coordination: M9 only adds non-invasive tracing decorators; M10 and M12 only import pipelines, never mutate them.

| Agent | Milestone | Owns                                              |
| ----- | --------- | ------------------------------------------------- |
| E1    | M9        | `app/tracing.py` + decorator additions            |
| E2    | M10       | `app/main.py`, `templates/`, `static/`            |
| E3    | M12       | `eval/graders.py`, `eval/runner.py`, `eval/ragas_runner.py` |

### Wave F — Serial (orchestrator)
- **M13**: Dashboard rendering — needs both M10 (UI shell) and M12 (results JSON shape).

### Wave G — Serial (orchestrator)
- **M14**: README, justfile, end-to-end smoke test on a fresh clone.

## Subagent protocol

Each Sonnet subagent is briefed with:

1. **Pointer to this file** and to `docs/DESIGN.md` for context.
2. **Strict scope:** the specific files it owns. It must not edit files owned by other agents in the same wave.
3. **Exit criteria** matching the milestone's exit gate.
4. **Test requirement:** new code lands with at least one pytest, unless the milestone is a content task (M11).
5. **No commits:** the orchestrator commits at wave boundaries to keep history clean.

## Compression estimate

14 sequential milestones → ~7-8 wall-clock units once Wave A, C, D, E run in parallel. The serial choke points (M1, M5, M13, M14) are unavoidable.
