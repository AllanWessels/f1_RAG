"""
Unit tests for v2 hybrid pipeline (dense + sparse BM25 fusion, cross-collection RRF).

All external dependencies (Qdrant client, Embedders, Synthesizer) are fully mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from app.pipeline import PipelineInput, PipelineResult
from app.pipelines.v2_hybrid import HybridPipeline, _rrf_merge, run_v2
from app.synth import ContextChunk, SynthResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hit(
    hit_id: int,
    *,
    chunk_id: str | None = None,
    text: str = "Some text.",
    source: str = "narratives",
    extra: dict | None = None,
) -> MagicMock:
    """Create a mock Qdrant ScoredPoint."""
    hit = MagicMock()
    hit.id = hit_id
    payload = {
        "chunk_id": chunk_id or f"chunk_{hit_id}",
        "text": text,
        "source": source,
        **(extra or {}),
    }
    hit.payload = payload
    return hit


def _make_query_result(hits: list[MagicMock]) -> MagicMock:
    """Wrap a list of hits in a mock that has a .points attribute."""
    result = MagicMock()
    result.points = hits
    return result


def _make_embedders(
    dense_vec: list[float] | None = None,
    sparse_indices: list[int] | None = None,
    sparse_values: list[float] | None = None,
) -> MagicMock:
    embedders = MagicMock()
    embedders.encode_dense.return_value = dense_vec or [0.1, 0.2, 0.3]
    embedders.encode_sparse.return_value = (
        sparse_indices or [0, 5, 10],
        sparse_values or [0.5, 0.3, 0.2],
    )
    return embedders


def _make_synthesizer(
    answer: str = "Test answer.",
    citations: list[str] | None = None,
    refused: bool = False,
) -> MagicMock:
    synth = MagicMock()
    synth.synthesize.return_value = SynthResult(
        answer=answer,
        citations=citations or [],
        refused=refused,
        raw_response=answer,
    )
    return synth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_3x5() -> MagicMock:
    """Mock Qdrant client returning 5 unique hits per collection."""
    client = MagicMock()

    def query_points_side_effect(collection_name, **kwargs):
        base_id = {"stats_meta": 100, "narratives": 200, "regulations": 300}[
            collection_name
        ]
        source = {"stats_meta": "stats", "narratives": "narratives", "regulations": "regulations"}[
            collection_name
        ]
        hits = [
            _make_hit(base_id + i, chunk_id=f"{source}_{i}", source=source)
            for i in range(5)
        ]
        return _make_query_result(hits)

    client.query_points.side_effect = query_points_side_effect
    return client


# ---------------------------------------------------------------------------
# Test 1 — Happy path: 3 collections x 5 hits -> top_k=5 passed to synthesizer
# ---------------------------------------------------------------------------


def test_happy_path_top_k_contexts_sent_to_synth(client_3x5):
    embedders = _make_embedders()
    synth = _make_synthesizer(answer="Verstappen won.", citations=["narratives_0"])

    pipeline = HybridPipeline(client=client_3x5, embedders=embedders, synthesizer=synth)
    result = pipeline.run(PipelineInput(query="Who won?", top_k=5))

    # Exactly 5 contexts should reach the synthesizer
    call_args = synth.synthesize.call_args
    contexts_sent = call_args[0][1]  # positional arg 1
    assert len(contexts_sent) == 5

    # Result has correct structure
    assert isinstance(result, PipelineResult)
    assert result.version == "v2"
    assert result.answer == "Verstappen won."
    assert result.query == "Who won?"


# ---------------------------------------------------------------------------
# Test 2 — Hybrid query shape: each query_points call uses Prefetch + FusionQuery
# ---------------------------------------------------------------------------


def test_hybrid_query_shape(client_3x5):
    embedders = _make_embedders()
    synth = _make_synthesizer()

    pipeline = HybridPipeline(client=client_3x5, embedders=embedders, synthesizer=synth)
    pipeline.run(PipelineInput(query="test query", top_k=5))

    for call_obj in client_3x5.query_points.call_args_list:
        kwargs = call_obj.kwargs

        # Must have prefetch with exactly 2 items
        prefetch = kwargs.get("prefetch", [])
        assert len(prefetch) == 2, f"Expected 2 Prefetch items, got {len(prefetch)}"

        # First Prefetch is dense
        dense_pf = prefetch[0]
        assert isinstance(dense_pf, models.Prefetch)
        assert dense_pf.using == "dense"
        assert isinstance(dense_pf.query, list)  # dense vector is a list of floats

        # Second Prefetch is sparse
        sparse_pf = prefetch[1]
        assert isinstance(sparse_pf, models.Prefetch)
        assert sparse_pf.using == "sparse"
        assert isinstance(sparse_pf.query, models.SparseVector)

        # Top-level query must be FusionQuery with RRF
        fusion_query = kwargs.get("query")
        assert isinstance(fusion_query, models.FusionQuery)
        assert fusion_query.fusion == models.Fusion.RRF


# ---------------------------------------------------------------------------
# Test 3 — Exactly 3 collection queries
# ---------------------------------------------------------------------------


def test_three_collection_queries(client_3x5):
    embedders = _make_embedders()
    synth = _make_synthesizer()

    pipeline = HybridPipeline(client=client_3x5, embedders=embedders, synthesizer=synth)
    pipeline.run(PipelineInput(query="test", top_k=5))

    assert client_3x5.query_points.call_count == 3

    queried_collections = {
        c.kwargs["collection_name"]
        for c in client_3x5.query_points.call_args_list
    }
    assert queried_collections == {"stats_meta", "narratives", "regulations"}


# ---------------------------------------------------------------------------
# Test 4 — RRF math: deterministic fixture with hand-calculated scores
# ---------------------------------------------------------------------------


def test_rrf_merge_math():
    """
    Two collections, 3 hits each.

    Collection A ranked: hit_1, hit_2, hit_3
      hit_1: 1/(60+1)  = 1/61
      hit_2: 1/(60+2)  = 1/62
      hit_3: 1/(60+3)  = 1/63

    Collection B ranked: hit_4, hit_2, hit_5
      hit_4: 1/(60+1)  = 1/61
      hit_2: 1/(60+2)  = 1/62   (accumulates with collection A)
      hit_5: 1/(60+3)  = 1/63

    Expected merged scores:
      hit_2: 1/62 + 1/62 = 2/62 ≈ 0.032258
      hit_1: 1/61         ≈ 0.016393
      hit_4: 1/61         ≈ 0.016393
      hit_3: 1/63         ≈ 0.015873
      hit_5: 1/63         ≈ 0.015873

    Merged order: hit_2, hit_1 or hit_4 (equal), hit_3 or hit_5 (equal).
    """
    hit_1 = _make_hit(1, chunk_id="h1")
    hit_2 = _make_hit(2, chunk_id="h2")
    hit_3 = _make_hit(3, chunk_id="h3")
    hit_4 = _make_hit(4, chunk_id="h4")
    hit_5 = _make_hit(5, chunk_id="h5")

    list_a = [hit_1, hit_2, hit_3]
    list_b = [hit_4, hit_2, hit_5]

    merged = _rrf_merge([list_a, list_b], k=60)

    # hit_2 must be first (highest score, appearing in both lists)
    assert merged[0][0].id == 2

    # Verify hit_2 score = 2/62
    expected_hit2 = 1 / (60 + 2) + 1 / (60 + 2)
    assert abs(merged[0][1] - expected_hit2) < 1e-9

    # hit_1 and hit_4 each have score 1/61 (tied)
    id_score_map = {hit.id: score for hit, score in merged}
    assert abs(id_score_map[1] - 1 / 61) < 1e-9
    assert abs(id_score_map[4] - 1 / 61) < 1e-9
    assert abs(id_score_map[3] - 1 / 63) < 1e-9
    assert abs(id_score_map[5] - 1 / 63) < 1e-9

    # 5 unique hits total
    assert len(merged) == 5

    # Overall ordering: hit_2 first, then the 1/61 group above 1/63 group
    merged_ids = [hit.id for hit, _ in merged]
    assert merged_ids[0] == 2
    assert set(merged_ids[1:3]) == {1, 4}
    assert set(merged_ids[3:5]) == {3, 5}


# ---------------------------------------------------------------------------
# Test 5 — Hits missing `text` in payload are dropped
# ---------------------------------------------------------------------------


def test_hits_without_text_are_dropped():
    client = MagicMock()

    # Collection returns 3 hits: 2 with text, 1 without
    hit_with_text_a = _make_hit(1, chunk_id="c1", text="Real text A", source="stats")
    hit_no_text = _make_hit(2, chunk_id="c2", text="", source="stats")
    hit_no_text.payload["text"] = ""  # explicitly empty
    hit_with_text_b = _make_hit(3, chunk_id="c3", text="Real text B", source="stats")

    # All three collections return the same 3 hits (ids unique enough for this test)
    client.query_points.return_value = _make_query_result(
        [hit_with_text_a, hit_no_text, hit_with_text_b]
    )

    embedders = _make_embedders()
    synth = _make_synthesizer()

    pipeline = HybridPipeline(client=client, embedders=embedders, synthesizer=synth)
    # top_k large enough to see all
    pipeline.run(PipelineInput(query="test", top_k=20))

    call_args = synth.synthesize.call_args
    contexts_sent: list[ContextChunk] = call_args[0][1]

    # No empty-text contexts should be present
    assert all(ctx.text for ctx in contexts_sent)
    cite_ids = [ctx.cite_id for ctx in contexts_sent]
    assert "c2" not in cite_ids


# ---------------------------------------------------------------------------
# Test 6 — Empty retrieval → synth called with [] → refused result
# ---------------------------------------------------------------------------


def test_empty_retrieval_produces_refused_result():
    client = MagicMock()
    # All collections return no hits
    client.query_points.return_value = _make_query_result([])

    embedders = _make_embedders()
    synth = _make_synthesizer(
        answer="I don't have enough information to answer that.",
        refused=True,
    )

    pipeline = HybridPipeline(client=client, embedders=embedders, synthesizer=synth)
    result = pipeline.run(PipelineInput(query="unknown question", top_k=5))

    # Synth must have been called with empty list
    call_args = synth.synthesize.call_args
    assert call_args[0][1] == []

    assert result.refused is True


# ---------------------------------------------------------------------------
# Test 7 — Result has version="v2" and tool_calls=[]
# ---------------------------------------------------------------------------


def test_result_version_and_tool_calls(client_3x5):
    embedders = _make_embedders()
    synth = _make_synthesizer()

    pipeline = HybridPipeline(client=client_3x5, embedders=embedders, synthesizer=synth)
    result = pipeline.run(PipelineInput(query="test", top_k=5))

    assert result.version == "v2"
    assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Extra — run_v2 convenience function wires through correctly
# ---------------------------------------------------------------------------


def test_run_v2_uses_provided_dependencies(client_3x5):
    embedders = _make_embedders()
    synth = _make_synthesizer(answer="Convenience answer.")

    result = run_v2("Hamilton or Verstappen?", top_k=3, client=client_3x5, embedders=embedders, synthesizer=synth)

    assert result.version == "v2"
    assert result.answer == "Convenience answer."
    assert client_3x5.query_points.call_count == 3
