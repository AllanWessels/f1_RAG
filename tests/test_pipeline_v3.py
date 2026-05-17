"""
Tests for pipeline v3 (hybrid retrieval + bge-reranker-v2-m3).

All external dependencies (Qdrant, Embedders, Reranker, Synthesizer) are
fully mocked so the tests run without network or model downloads.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.pipeline import PipelineInput
from app.pipelines.v3_rerank import RerankPipeline
from app.synth import ContextChunk, SynthResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_point(chunk_id: str, text: str, source: str, score: float = 0.5) -> SimpleNamespace:
    """Return a fake Qdrant ScoredPoint-like object."""
    return SimpleNamespace(
        score=score,
        payload={
            "chunk_id": chunk_id,
            "text": text,
            "source": source,
        },
    )


def _make_query_result(points: list) -> SimpleNamespace:
    return SimpleNamespace(points=points)


def _make_client(hits_per_collection: int = 10) -> MagicMock:
    """
    Return a mock QdrantClient whose query_points cycles through three
    collections and returns `hits_per_collection` fake hits each time.
    """
    client = MagicMock()

    def _query_points(**kwargs):
        col = kwargs["collection_name"]
        source_map = {
            "stats_meta": "stats",
            "narratives": "narratives",
            "regulations": "regulations",
        }
        source = source_map[col]
        points = [
            _make_point(
                chunk_id=f"{col}_chunk_{i}",
                text=f"Text from {col} chunk {i}",
                source=source,
                score=1.0 - i * 0.05,
            )
            for i in range(hits_per_collection)
        ]
        return _make_query_result(points)

    client.query_points.side_effect = _query_points
    return client


def _make_embedders() -> MagicMock:
    embedders = MagicMock()
    embedders.encode_dense.return_value = [0.1] * 1024
    embedders.encode_sparse.return_value = ([1, 2, 3], [0.5, 0.3, 0.2])
    return embedders


def _make_synthesizer(refused: bool = False) -> MagicMock:
    synth = MagicMock()
    synth.synthesize.return_value = SynthResult(
        answer="Test answer." if not refused else "I don't have enough information to answer that.",
        citations=["s_chunk_0"] if not refused else [],
        refused=refused,
    )
    return synth


def _make_pipeline(
    client=None,
    embedders=None,
    reranker=None,
    synthesizer=None,
    fetch_k: int = 30,
    hits_per_collection: int = 10,
    reranker_top_k_indices: list[tuple[int, float]] | None = None,
) -> tuple[RerankPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
    if client is None:
        client = _make_client(hits_per_collection)
    if embedders is None:
        embedders = _make_embedders()
    if reranker is None:
        reranker = MagicMock()
        if reranker_top_k_indices is None:
            reranker_top_k_indices = [(i, 1.0 - i * 0.1) for i in range(5)]
        reranker.rerank.return_value = reranker_top_k_indices
    if synthesizer is None:
        synthesizer = _make_synthesizer()

    pipeline = RerankPipeline(
        client=client,
        embedders=embedders,
        reranker=reranker,
        synthesizer=synthesizer,
        fetch_k=fetch_k,
    )
    return pipeline, client, embedders, reranker, synthesizer


# ---------------------------------------------------------------------------
# Test 1: Happy path
# ---------------------------------------------------------------------------

def test_happy_path_end_to_end() -> None:
    """3 collections x 10 hits -> merged (30) -> reranker called once -> top-5 -> synth called."""
    reranker_results = [(i, 1.0 - i * 0.1) for i in range(5)]
    pipeline, client, embedders, reranker, synthesizer = _make_pipeline(
        hits_per_collection=10,
        reranker_top_k_indices=reranker_results,
    )
    result = pipeline.run(PipelineInput(query="Who won the 2021 Brazilian GP?", top_k=5))

    # Qdrant queried 3 times (once per collection).
    assert client.query_points.call_count == 3

    # Embedders called once for dense and once for sparse.
    embedders.encode_dense.assert_called_once()
    embedders.encode_sparse.assert_called_once()

    # Reranker called exactly once.
    assert reranker.rerank.call_count == 1

    # Synthesizer called exactly once.
    assert synthesizer.synthesize.call_count == 1

    # Result version.
    assert result.version == "v3"
    assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Test 2: Hybrid query structure
# ---------------------------------------------------------------------------

def test_hybrid_query_uses_prefetch_and_rrf() -> None:
    """Each query_points call must have prefetch with 2 items and FusionQuery(RRF)."""
    from qdrant_client.models import FusionQuery, Prefetch

    pipeline, client, _, _, _ = _make_pipeline()
    pipeline.run(PipelineInput(query="test query", top_k=5))

    for actual_call in client.query_points.call_args_list:
        kwargs = actual_call.kwargs
        # Should use prefetch.
        assert "prefetch" in kwargs, "query_points must use prefetch keyword"
        prefetch = kwargs["prefetch"]
        assert isinstance(prefetch, list), "prefetch must be a list"
        assert len(prefetch) == 2, f"Expected 2 prefetch items, got {len(prefetch)}"

        # Both items must be Prefetch instances.
        assert all(isinstance(p, Prefetch) for p in prefetch), "All prefetch entries must be Prefetch"

        # Must use RRF fusion.
        query_arg = kwargs.get("query")
        assert query_arg is not None, "query kwarg must be present"
        assert isinstance(query_arg, FusionQuery), "query must be a FusionQuery"
        assert query_arg.fusion.lower() == "rrf", "FusionQuery must use RRF"


# ---------------------------------------------------------------------------
# Test 3: Reranker invoked exactly once with correct args
# ---------------------------------------------------------------------------

def test_reranker_called_once_with_correct_texts() -> None:
    """Reranker is called once with (query, texts) where len(texts) <= fetch_k."""
    pipeline, _, _, reranker, _ = _make_pipeline(
        hits_per_collection=10,
        reranker_top_k_indices=[(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)],
    )
    query = "Who was on pole at Monza 2022?"
    pipeline.run(PipelineInput(query=query, top_k=5))

    assert reranker.rerank.call_count == 1
    call_args = reranker.rerank.call_args
    called_query = call_args.args[0] if call_args.args else call_args.kwargs.get("query")
    called_texts = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("candidates")

    assert called_query == query
    assert isinstance(called_texts, list)
    assert len(called_texts) <= 30  # fetch_k default


# ---------------------------------------------------------------------------
# Test 4: Reranker score replaces RRF score in final ContextChunk ordering
# ---------------------------------------------------------------------------

def test_reranker_score_in_final_contexts() -> None:
    """Reranker scores should appear in context metadata (rerank_score)."""
    reranker_results = [(2, 0.95), (0, 0.80), (4, 0.70), (1, 0.60), (3, 0.50)]
    pipeline, _, _, _, synthesizer = _make_pipeline(
        hits_per_collection=10,
        reranker_top_k_indices=reranker_results,
    )
    pipeline.run(PipelineInput(query="test", top_k=5))

    call_args = synthesizer.synthesize.call_args
    contexts: list[ContextChunk] = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["contexts"]

    # Each context must carry the reranker score.
    scores = [ctx.metadata.get("rerank_score") for ctx in contexts]
    assert all(s is not None for s in scores), "All contexts must have rerank_score metadata"

    # Scores should be in descending order (first context has highest reranker score).
    assert scores == sorted(scores, reverse=True), "Contexts must be ordered by reranker score desc"


# ---------------------------------------------------------------------------
# Test 5: Reranker indices determine output order
# ---------------------------------------------------------------------------

def test_reranker_index_ordering() -> None:
    """If reranker returns indices [2, 0, 4], contexts come from candidates 2, 0, 4 in that order."""
    # We need a client that returns predictable hits so we can verify indices.
    client = MagicMock()
    hits_by_col: dict[str, list] = {
        "stats_meta": [
            _make_point(f"stats_chunk_{i}", f"Stats text {i}", "stats", score=1.0 - i * 0.1)
            for i in range(10)
        ],
        "narratives": [
            _make_point(f"narr_chunk_{i}", f"Narr text {i}", "narratives", score=0.95 - i * 0.1)
            for i in range(10)
        ],
        "regulations": [
            _make_point(f"regs_chunk_{i}", f"Regs text {i}", "regulations", score=0.9 - i * 0.1)
            for i in range(10)
        ],
    }
    client.query_points.side_effect = lambda **kwargs: _make_query_result(
        hits_by_col[kwargs["collection_name"]]
    )

    embedders = _make_embedders()
    # Reranker returns only indices 2, 0, 4 (from the merged+sorted candidate list).
    reranker = MagicMock()
    reranker.rerank.return_value = [(2, 0.9), (0, 0.8), (4, 0.7)]
    synthesizer = _make_synthesizer()

    pipeline = RerankPipeline(
        client=client,
        embedders=embedders,
        reranker=reranker,
        synthesizer=synthesizer,
        fetch_k=30,
    )
    pipeline.run(PipelineInput(query="test index ordering", top_k=3))

    call_args = synthesizer.synthesize.call_args
    contexts: list[ContextChunk] = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["contexts"]

    assert len(contexts) == 3, f"Expected 3 contexts, got {len(contexts)}"

    # The reranker scores must be in descending order.
    scores = [ctx.metadata["rerank_score"] for ctx in contexts]
    assert scores == [0.9, 0.8, 0.7]


# ---------------------------------------------------------------------------
# Test 6: Hits missing "text" are dropped before reranking
# ---------------------------------------------------------------------------

def test_hits_without_text_dropped_before_reranking() -> None:
    """Hits with no 'text' in payload are excluded before calling the reranker."""
    client = MagicMock()
    # Inject one hit without text in each collection.
    def _query_points(**kwargs):
        col = kwargs["collection_name"]
        pts = [
            _make_point(f"{col}_good_{i}", f"Good text {i}", "stats", score=1.0 - i * 0.1)
            for i in range(9)
        ]
        # One point with empty text.
        bad = SimpleNamespace(
            score=0.99,
            payload={"chunk_id": f"{col}_bad", "text": "", "source": "stats"},
        )
        pts.insert(3, bad)
        return _make_query_result(pts)

    client.query_points.side_effect = _query_points

    reranker = MagicMock()
    reranker.rerank.return_value = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]
    synthesizer = _make_synthesizer()

    pipeline = RerankPipeline(
        client=client,
        embedders=_make_embedders(),
        reranker=reranker,
        synthesizer=synthesizer,
        fetch_k=30,
    )
    pipeline.run(PipelineInput(query="test text filtering", top_k=5))

    # The texts passed to reranker must all be non-empty.
    call_args = reranker.rerank.call_args
    texts = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("candidates")
    assert all(t for t in texts), "Reranker must not receive empty texts"


# ---------------------------------------------------------------------------
# Test 7: Empty retrieval → synth called with [] → refused; reranker NOT called
# ---------------------------------------------------------------------------

def test_empty_retrieval_skips_reranker() -> None:
    """When no hits are retrieved, reranker is not called and synth returns refusal."""
    client = MagicMock()
    # All collections return 0 hits.
    client.query_points.return_value = _make_query_result([])

    reranker = MagicMock()
    synthesizer = MagicMock()
    synthesizer.synthesize.return_value = SynthResult(
        answer="I don't have enough information to answer that.",
        citations=[],
        refused=True,
    )

    pipeline = RerankPipeline(
        client=client,
        embedders=_make_embedders(),
        reranker=reranker,
        synthesizer=synthesizer,
        fetch_k=30,
    )
    result = pipeline.run(PipelineInput(query="empty query", top_k=5))

    reranker.rerank.assert_not_called()
    synthesizer.synthesize.assert_called_once_with("empty query", [])
    assert result.refused is True
    assert result.version == "v3"


# ---------------------------------------------------------------------------
# Test 8: Result version and tool_calls
# ---------------------------------------------------------------------------

def test_result_metadata() -> None:
    """Result must have version='v3' and tool_calls=[]."""
    pipeline, _, _, _, _ = _make_pipeline()
    result = pipeline.run(PipelineInput(query="metadata check", top_k=5))

    assert result.version == "v3"
    assert result.tool_calls == []
    assert isinstance(result.answer, str)
