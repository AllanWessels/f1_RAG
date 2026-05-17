"""
Tests for app/main.py  (M10 — FastAPI + HTMX UI).

All tests run without loading heavy models. The pipeline runner is replaced
by a lightweight MagicMock via FastAPI's dependency_overrides mechanism.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_pipeline_runner
from app.pipeline import PipelineResult
from app.synth import ContextChunk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(version: str = "v1", query: str = "Test query") -> PipelineResult:
    """Return a minimal PipelineResult for mock purposes."""
    return PipelineResult(
        version=version,  # type: ignore[arg-type]
        query=query,
        answer="Lewis Hamilton won the race. [cite:A]",
        citations=["A"],
        refused=False,
        retrieved_contexts=[
            ContextChunk(
                cite_id="A",
                source="narratives",
                text="Hamilton dominated from pole position.",
                metadata={"year": 2021},
            )
        ],
        tool_calls=[],
    )


def _make_runner(version: str = "v1"):
    """Return a callable that acts as a pipeline runner mock."""

    def _runner(query: str, ver: str, top_k: int) -> PipelineResult:
        return _make_result(version=ver, query=query)

    return _runner


@pytest.fixture()
def client():
    """TestClient with the pipeline runner overridden to avoid real model calls."""
    mock_runner = _make_runner()
    app.dependency_overrides[get_pipeline_runner] = lambda: mock_runner
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. Health probe
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 2. Chat page returns HTML with version dropdown and question form
# ---------------------------------------------------------------------------


def test_chat_page_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    # Should include the version dropdown options
    assert "v1" in body
    assert "v2" in body
    assert "v3" in body
    assert "v4" in body
    # Should include a form / input for the query
    assert "query" in body
    # Confirm it's HTML
    assert "<html" in body.lower() or "<!doctype" in body.lower()


# ---------------------------------------------------------------------------
# 3. POST /api/chat with mocked runner returns PipelineResult JSON
# ---------------------------------------------------------------------------


def test_api_chat_returns_pipeline_result(client):
    payload = {"query": "Who won the 2021 Brazilian GP?", "version": "v1", "top_k": 3}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    # PipelineResult fields
    assert data["version"] == "v1"
    assert data["query"] == "Who won the 2021 Brazilian GP?"
    assert isinstance(data["answer"], str)
    assert isinstance(data["citations"], list)
    assert isinstance(data["retrieved_contexts"], list)
    assert isinstance(data["tool_calls"], list)
    assert "refused" in data


# ---------------------------------------------------------------------------
# 4. GET /eval shows "no eval results yet" when file is absent
# ---------------------------------------------------------------------------


def test_eval_page_no_results(client, tmp_path, monkeypatch):
    # Point EVAL_RESULTS_PATH at a non-existent file
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "EVAL_RESULTS_PATH", tmp_path / "nonexistent.json")
    resp = client.get("/eval")
    assert resp.status_code == 200
    assert "no eval results yet" in resp.text.lower()


# ---------------------------------------------------------------------------
# 5a. GET /api/eval/results returns {"status":"no_results"} when file is missing
# ---------------------------------------------------------------------------


def test_api_eval_results_no_file(client, tmp_path, monkeypatch):
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "EVAL_RESULTS_PATH", tmp_path / "nonexistent.json")
    resp = client.get("/api/eval/results")
    assert resp.status_code == 200
    assert resp.json() == {"status": "no_results"}


# ---------------------------------------------------------------------------
# 5b. GET /api/eval/results returns file content when present
# ---------------------------------------------------------------------------


def test_api_eval_results_with_file(client, tmp_path, monkeypatch):
    import app.main as main_mod

    sample = {"versions": ["v1", "v2"], "questions": []}
    results_file = tmp_path / "latest.json"
    results_file.write_text(json.dumps(sample))
    monkeypatch.setattr(main_mod, "EVAL_RESULTS_PATH", results_file)

    resp = client.get("/api/eval/results")
    assert resp.status_code == 200
    assert resp.json() == sample


# ---------------------------------------------------------------------------
# 6. Invalid version returns 422
# ---------------------------------------------------------------------------


def test_api_chat_invalid_version(client):
    payload = {"query": "Test", "version": "v9"}
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 7. POST /api/chat honors the version param (correct pipeline called)
# ---------------------------------------------------------------------------


def test_api_chat_version_routing():
    """Assert the runner is called with the exact version the client requested."""
    call_log: list[str] = []

    def _tracking_runner(query: str, ver: str, top_k: int) -> PipelineResult:
        call_log.append(ver)
        return _make_result(version=ver, query=query)

    app.dependency_overrides[get_pipeline_runner] = lambda: _tracking_runner
    try:
        with TestClient(app) as c:
            for version in ("v1", "v2", "v3", "v4"):
                resp = c.post("/api/chat", json={"query": "Test", "version": version})
                assert resp.status_code == 200, f"Failed for {version}: {resp.text}"
                assert resp.json()["version"] == version
    finally:
        app.dependency_overrides.clear()

    # Every version was dispatched exactly once in order
    assert call_log == ["v1", "v2", "v3", "v4"]
