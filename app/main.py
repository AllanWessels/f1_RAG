"""
FastAPI application for the F1 RAG system.

Serves:
  GET  /           — Jinja+HTMX chat UI with version dropdown
  POST /api/chat   — JSON API; accepts {query, version, top_k?}
  GET  /eval       — Eval dashboard (reads eval/results/latest.json)
  GET  /api/eval/results — JSON content of latest.json (or status no_results)
  GET  /health     — Health probe {"status":"ok"}

Heavy models are loaded lazily (on first request) via singleton helpers so
import time stays fast and tests can inject mocks via dependency_overrides.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import BaseModel

from app.pipeline import PipelineResult, VersionId

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = _PROJECT_ROOT / "templates"
STATIC_DIR = _PROJECT_ROOT / "static"
EVAL_RESULTS_PATH = _PROJECT_ROOT / "eval" / "results" / "latest.json"

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="F1 RAG", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Custom Jinja2 filters
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(r"\[cite:([^\]]+)\]")


def _replace_citations_filter(text: str) -> Markup:
    """Replace [cite:ID] markers with anchor links to the source footnote."""
    safe_text = str(escape(text))

    def _sub(m: re.Match) -> str:
        cite_id = escape(m.group(1))
        return (
            f'<a href="#cite-{cite_id}" '
            f'class="inline-flex items-center text-red-600 hover:text-red-800 '
            f'font-mono text-xs font-semibold ml-0.5" '
            f'title="See source {cite_id}">'
            f'[{cite_id}]</a>'
        )

    return Markup(_CITE_RE.sub(_sub, safe_text))


templates.env.filters["replace_citations"] = _replace_citations_filter

# ---------------------------------------------------------------------------
# Lazy singleton helpers — never import heavy model classes at module level
# ---------------------------------------------------------------------------

_embedders = None
_reranker = None


def get_embedders():
    global _embedders
    if _embedders is None:
        from app.embedding import Embedders

        _embedders = Embedders()
    return _embedders


def get_reranker():
    global _reranker
    if _reranker is None:
        from app.embedding import Reranker

        _reranker = Reranker()
    return _reranker


# ---------------------------------------------------------------------------
# Pipeline runner — injectable via Depends for test mocking
# ---------------------------------------------------------------------------


def _real_pipeline_runner(query: str, version: VersionId, top_k: int) -> PipelineResult:
    """Dispatch to the correct pipeline version and return a PipelineResult."""
    if version == "v1":
        from app.pipelines.v1_naive import run_v1

        return run_v1(query, top_k=top_k, embedders=get_embedders())

    elif version == "v2":
        from app.pipelines.v2_hybrid import run_v2

        return run_v2(query, top_k=top_k, embedders=get_embedders())

    elif version == "v3":
        from app.pipelines.v3_rerank import run_v3

        return run_v3(query, top_k=top_k, embedders=get_embedders(), reranker=get_reranker())

    elif version == "v4":
        from app.pipelines.v4_router import run_v4

        return run_v4(query, top_k=top_k)

    else:
        raise ValueError(f"Unknown version: {version}")


def get_pipeline_runner():
    """Dependency that returns the pipeline runner callable.

    Override in tests via app.dependency_overrides[get_pipeline_runner].
    """
    return _real_pipeline_runner


PipelineRunnerDep = Annotated[
    object,  # callable(query, version, top_k) -> PipelineResult
    Depends(get_pipeline_runner),
]

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str
    version: Literal["v1", "v2", "v3", "v4"]
    top_k: int = 5


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat UI  (GET /)
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(
        request,
        "chat.html",
        {"versions": ["v1", "v2", "v3", "v4"]},
    )


# ---------------------------------------------------------------------------
# Chat API  (POST /api/chat)
# ---------------------------------------------------------------------------


class ChatResponse(BaseModel):
    """JSON response shape returned by POST /api/chat."""

    version: str
    query: str
    answer: str
    citations: list[str]
    refused: bool
    retrieved_contexts: list[dict]
    tool_calls: list[dict]


@app.post("/api/chat")
def api_chat(
    body: ChatRequest,
    runner: PipelineRunnerDep,
) -> JSONResponse:
    """Run the selected pipeline and return a PipelineResult as JSON."""
    try:
        result: PipelineResult = runner(body.query, body.version, body.top_k)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(content=result.model_dump())


# ---------------------------------------------------------------------------
# HTMX chat partial  (POST /chat)  — returns HTML fragment
# ---------------------------------------------------------------------------


@app.post("/chat", response_class=HTMLResponse)
def chat_htmx(
    request: Request,
    query: Annotated[str, Form()],
    version: Annotated[str, Form()],
    top_k: Annotated[int, Form()] = 5,
    runner: Annotated[object, Depends(get_pipeline_runner)] = None,
):
    """HTMX endpoint: receives form POST, returns a chat response HTML partial."""
    if version not in ("v1", "v2", "v3", "v4"):
        raise HTTPException(status_code=422, detail=f"Invalid version: {version}")

    try:
        result: PipelineResult = runner(query, version, top_k)  # type: ignore[arg-type]
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        return templates.TemplateResponse(
            request,
            "chat_response.html",
            {
                "error": str(exc),
                "query": query,
                "version": version,
            },
        )

    return templates.TemplateResponse(
        request,
        "chat_response.html",
        {
            "result": result,
            "query": query,
            "version": version,
        },
    )


# ---------------------------------------------------------------------------
# Eval dashboard  (GET /eval)
# ---------------------------------------------------------------------------


@app.get("/eval", response_class=HTMLResponse)
def eval_page(request: Request):
    eval_data = _load_eval_results()
    return templates.TemplateResponse(
        request,
        "eval.html",
        {"eval_data": eval_data},
    )


# ---------------------------------------------------------------------------
# Eval results API  (GET /api/eval/results)
# ---------------------------------------------------------------------------


@app.get("/api/eval/results")
def api_eval_results() -> JSONResponse:
    data = _load_eval_results()
    if data is None:
        return JSONResponse(content={"status": "no_results"})
    return JSONResponse(content=data)


# ---------------------------------------------------------------------------
# Helper: load eval/results/latest.json
# ---------------------------------------------------------------------------


def _load_eval_results() -> dict | None:
    """Return parsed JSON from EVAL_RESULTS_PATH, or None if absent."""
    path = EVAL_RESULTS_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.exception("Failed to parse %s", path)
        return None
