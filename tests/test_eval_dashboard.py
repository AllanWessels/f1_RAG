"""End-to-end test that the /eval dashboard renders a real-shape M12 JSON.

M10 built the template; M12 produces the JSON; M13 wires them together. This
test asserts the wiring works with the actual schema from eval.runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.main import app


@pytest.fixture
def eval_results_file(tmp_path: Path, monkeypatch) -> Path:
    """Point the app at a tmp results file with M12-shaped data."""
    target = tmp_path / "latest.json"
    monkeypatch.setattr(app_main, "EVAL_RESULTS_PATH", target)
    return target


def _sample_payload() -> dict:
    """Mimic the M12 eval.runner JSON shape exactly."""
    return {
        "timestamp": "2026-05-17T10:00:00Z",
        "versions": ["v1", "v2", "v3", "v4"],
        "questions": ["factual_stats_001", "narrative_001"],
        "results": [
            {
                "question_id": "factual_stats_001",
                "bucket": "factual_stats",
                "version": "v1",
                "question": "Who won the 2017 Australian Grand Prix?",
                "gold_answer": "Sebastian Vettel",
                "answer": "Sebastian Vettel.",
                "citations": ["A"],
                "refused": False,
                "grade": {"pass": True, "score": 1.0, "detail": "exact match"},
                "tool_calls": [],
                "trace_url": "http://langfuse:3000/trace/abc",
                "error": None,
            },
            {
                "question_id": "factual_stats_001",
                "bucket": "factual_stats",
                "version": "v4",
                "question": "Who won the 2017 Australian Grand Prix?",
                "gold_answer": "Sebastian Vettel",
                "answer": "Vettel won.",
                "citations": ["A"],
                "refused": False,
                "grade": {"pass": True, "score": 1.0, "detail": "exact match"},
                "tool_calls": [],
                "trace_url": None,
                "error": None,
            },
            {
                "question_id": "narrative_001",
                "bucket": "narrative",
                "version": "v1",
                "question": "What was the 2021 Abu Dhabi controversy?",
                "gold_answer": "Safety-car procedure...",
                "answer": "I don't have enough information.",
                "citations": [],
                "refused": True,
                "grade": {"pass": False, "score": 0.0, "detail": "no match"},
                "tool_calls": [],
                "trace_url": None,
                "error": None,
            },
        ],
        "aggregates": {
            "v1": {
                "overall_pass_rate": 0.5,
                "overall_mean_score": 0.5,
                "per_bucket": {"factual_stats": 1.0, "narrative": 0.0},
            },
            "v2": {
                "overall_pass_rate": 0.5,
                "overall_mean_score": 0.5,
                "per_bucket": {"factual_stats": 1.0, "narrative": 0.0},
            },
            "v3": {
                "overall_pass_rate": 0.5,
                "overall_mean_score": 0.6,
                "per_bucket": {"factual_stats": 1.0, "narrative": 0.2},
            },
            "v4": {
                "overall_pass_rate": 1.0,
                "overall_mean_score": 0.95,
                "per_bucket": {"factual_stats": 1.0, "narrative": 0.9},
            },
        },
        "ragas": None,
    }


def test_eval_dashboard_renders_m12_shape(eval_results_file: Path) -> None:
    eval_results_file.write_text(json.dumps(_sample_payload()))
    client = TestClient(app)
    resp = client.get("/eval")
    assert resp.status_code == 200
    body = resp.text
    # Aggregates show up on chart payload (data is dumped inline as JSON).
    assert "overall_mean_score" in body
    assert "per_bucket" in body
    # Per-question table renders rows.
    assert "factual_stats" in body
    assert "Who won the 2017 Australian Grand Prix?" in body
    # Score column reads grade.score (1.00 for passing, 0.00 for failing).
    assert "1.00" in body
    assert "0.00" in body
    # Trace link is present for the result that has trace_url.
    assert "http://langfuse:3000/trace/abc" in body
    # Refusal flag rendered.
    assert ">Yes<" in body  # refused=True row


def test_eval_dashboard_handles_missing_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_main, "EVAL_RESULTS_PATH", tmp_path / "nope.json")
    client = TestClient(app)
    resp = client.get("/eval")
    assert resp.status_code == 200
    assert "no eval results yet" in resp.text


def test_eval_api_returns_m12_shape(eval_results_file: Path) -> None:
    payload = _sample_payload()
    eval_results_file.write_text(json.dumps(payload))
    client = TestClient(app)
    resp = client.get("/api/eval/results")
    assert resp.status_code == 200
    assert resp.json() == payload
