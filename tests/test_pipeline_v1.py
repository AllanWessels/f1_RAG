"""
Unit tests for the v1 naive pipeline.

All external dependencies (Qdrant client, Embedders, Synthesizer) are fully
mocked -- no network or model weights required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.pipeline import PipelineInput, PipelineResult
from app.pipelines.v1_naive import NaivePipeline, _make_cite_id
from app.synth import SynthResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

COLLECTIONS = ("stats_meta", "narratives", "regulations")


def _make_point(chunk_id: str, source: str, score: float, extra_payload: dict | None = None):
    """Build a fake Qdrant ScoredPoint-like object."""
    payload = {"chunk_id": chunk_id, "source": source, "text": f"Text for {chunk_id}"}
    if extra_payload:
        payload.update(extra_payload)
    return SimpleNamespace(score=score, payload=payload)


def _make_query_result(points):
    """Wrap a list of points in the QueryResult envelope Qdrant returns."""
    return SimpleNamespace(points=points)


def _make_synth_result(answer: str = "Test answer.", refused: bool = False) -> SynthResult:
    return SynthResult(answer=answer, citations=[], refused=refused, raw_response=answer)


def _build_pipeline(
    *,
    client: MagicMock,
    dense_vec: list[float] | None = None,
    synth_result: SynthResult | None = None,
) -> tuple[NaivePipeline, MagicMock, MagicMock]:
    """Return (pipeline, mock_embedders, mock_synthesizer)."""
    mock_embedders = MagicMock()
    mock_embedders.encode_dense.return_value = dense_vec or [0.1] * 1024

    mock_synthesizer = MagicMock()
    mock_synthesizer.synthesize.return_value = synth_result or _make_synth_result()

    pipeline = NaivePipeline(
        client=client,
        embedders=mock_embedders,
        synthesizer=mock_synthesizer,
    )
    return pipeline, mock_embedders, mock_synthesizer


# ---------------------------------------------------------------------------
# Test 1 - Happy path: 3 hits per collection -> merged, sorted, top_k=5
# ---------------------------------------------------------------------------

def test_happy_path_merges_and_sorts_to_top_k():
    """9 total hits (3 per collection); top_k=5 should be returned."""
    client = MagicMock()

    # Give each collection 3 hits with distinct scores.
    # stats_meta: scores 0.9, 0.6, 0.3
    # narratives:  scores 0.85, 0.55, 0.25
    # regulations: scores 0.8, 0.5, 0.2
    # Expected top 5 after merge: 0.9, 0.85, 0.8, 0.6, 0.55
    def side_effect(collection_name, **kwargs):
        if collection_name == "stats_meta":
            pts = [
                _make_point("s1", "stats", 0.9),
                _make_point("s2", "stats", 0.6),
                _make_point("s3", "stats", 0.3),
            ]
        elif collection_name == "narratives":
            pts = [
                _make_point("n1", "narratives", 0.85),
                _make_point("n2", "narratives", 0.55),
                _make_point("n3", "narratives", 0.25),
            ]
        else:  # regulations
            pts = [
                _make_point("r1", "regulations", 0.8),
                _make_point("r2", "regulations", 0.5),
                _make_point("r3", "regulations", 0.2),
            ]
        return _make_query_result(pts)

    client.query_points.side_effect = side_effect

    pipeline, _, mock_synthesizer = _build_pipeline(client=client)
    result = pipeline.run(PipelineInput(query="Who won?", top_k=5))

    # Synthesizer called exactly once.
    mock_synthesizer.synthesize.assert_called_once()
    _, called_contexts = mock_synthesizer.synthesize.call_args.args

    assert len(called_contexts) == 5
    # Verify ordering by inspecting cite_ids; the pipeline builds them as
    # <prefix>_<chunk_id> so s_s1, n_n1, r_r1, s_s2, n_n2.
    cite_ids = [c.cite_id for c in called_contexts]
    assert cite_ids == ["s_s1", "n_n1", "r_r1", "s_s2", "n_n2"]

    assert isinstance(result, PipelineResult)
    assert result.version == "v1"
    assert len(result.retrieved_contexts) == 5


# ---------------------------------------------------------------------------
# Test 2 - Dense-only: no Prefetch / FusionQuery in the call args
# ---------------------------------------------------------------------------

def test_dense_only_no_prefetch_or_fusion():
    """query_points calls must NOT contain prefetch or query as FusionQuery."""
    client = MagicMock()
    client.query_points.return_value = _make_query_result([])

    pipeline, _, _ = _build_pipeline(client=client)
    pipeline.run(PipelineInput(query="test", top_k=3))

    for c in client.query_points.call_args_list:
        kwargs = c.kwargs
        # No prefetch argument.
        assert "prefetch" not in kwargs, "prefetch must not be used in v1 naive pipeline"
        # The query must be a plain list (dense vector), not a FusionQuery object.
        query_arg = kwargs.get("query")
        assert isinstance(query_arg, list), (
            f"query must be a plain dense vector list, got {type(query_arg)}"
        )


# ---------------------------------------------------------------------------
# Test 3 - Exactly 3 query_points calls, one per collection
# ---------------------------------------------------------------------------

def test_queries_all_three_collections():
    """query_points is called exactly 3 times, once per collection."""
    client = MagicMock()
    client.query_points.return_value = _make_query_result([])

    pipeline, _, _ = _build_pipeline(client=client)
    pipeline.run(PipelineInput(query="test", top_k=5))

    assert client.query_points.call_count == 3

    queried_collections = {c.kwargs["collection_name"] for c in client.query_points.call_args_list}
    assert queried_collections == {"stats_meta", "narratives", "regulations"}


# ---------------------------------------------------------------------------
# Test 4 - Hits without "text" in payload are silently dropped
# ---------------------------------------------------------------------------

def test_hits_without_text_are_dropped():
    """Points with no 'text' key in their payload should not reach the synthesizer."""
    client = MagicMock()

    def side_effect(collection_name, **kwargs):
        if collection_name == "stats_meta":
            # One good hit, one missing text.
            good = _make_point("s1", "stats", 0.9)
            bad = SimpleNamespace(
                score=0.95,  # High score but no text -- should be dropped.
                payload={"chunk_id": "s_no_text", "source": "stats"},  # no "text" key
            )
            return _make_query_result([good, bad])
        return _make_query_result([])

    client.query_points.side_effect = side_effect

    pipeline, _, mock_synthesizer = _build_pipeline(client=client)
    pipeline.run(PipelineInput(query="test", top_k=5))

    _, called_contexts = mock_synthesizer.synthesize.call_args.args
    # Only the hit with text should be present.
    assert len(called_contexts) == 1
    # cite_id is built as s_<chunk_id>, confirming it's the right point.
    assert called_contexts[0].cite_id == "s_s1"


# ---------------------------------------------------------------------------
# Test 5 - Empty retrieval -> synthesizer called with []; result has refused=True
# ---------------------------------------------------------------------------

def test_empty_retrieval_synthesizer_called_with_empty_contexts():
    """When all collections return 0 hits, synthesize([]) is called and refused=True."""
    client = MagicMock()
    client.query_points.return_value = _make_query_result([])

    # Real Synthesizer returns refused=True on empty contexts; mock the same.
    refused_result = _make_synth_result(
        answer="I don't have enough information to answer that.", refused=True
    )
    pipeline, _, mock_synthesizer = _build_pipeline(client=client, synth_result=refused_result)
    result = pipeline.run(PipelineInput(query="Does this exist?", top_k=5))

    mock_synthesizer.synthesize.assert_called_once()
    _, called_contexts = mock_synthesizer.synthesize.call_args.args
    assert called_contexts == []

    assert result.refused is True


# ---------------------------------------------------------------------------
# Test 6 - cite_ids in returned contexts are unique
# ---------------------------------------------------------------------------

def test_cite_ids_are_unique():
    """Every context chunk in the result must have a unique cite_id."""
    client = MagicMock()

    def side_effect(collection_name, **kwargs):
        # Return 3 hits per collection, all with distinct chunk_ids.
        prefix = collection_name[0]
        pts = [
            _make_point(f"{prefix}{i}", collection_name.split("_")[0], 0.9 - i * 0.1)
            for i in range(3)
        ]
        return _make_query_result(pts)

    client.query_points.side_effect = side_effect

    pipeline, _, _ = _build_pipeline(client=client)
    result = pipeline.run(PipelineInput(query="test", top_k=9))

    cite_ids = [c.cite_id for c in result.retrieved_contexts]
    assert len(cite_ids) == len(set(cite_ids)), "cite_ids must be unique"


# ---------------------------------------------------------------------------
# Test 7 - PipelineResult has version="v1" and tool_calls=[]
# ---------------------------------------------------------------------------

def test_result_has_correct_version_and_no_tool_calls():
    """PipelineResult must declare version='v1' and have an empty tool_calls list."""
    client = MagicMock()
    client.query_points.return_value = _make_query_result([_make_point("x1", "stats", 0.8)])

    pipeline, _, _ = _build_pipeline(client=client)
    result = pipeline.run(PipelineInput(query="anything", top_k=5))

    assert result.version == "v1"
    assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Test - _make_cite_id stays under 50 chars
# ---------------------------------------------------------------------------

def test_make_cite_id_length():
    """cite_id must be strictly less than 50 characters."""
    very_long_chunk_id = "a" * 200
    cite_id = _make_cite_id("stats_meta", very_long_chunk_id)
    assert len(cite_id) < 50


# ---------------------------------------------------------------------------
# Test - using="dense" is passed to every query_points call
# ---------------------------------------------------------------------------

def test_dense_vector_index_name():
    """Every query_points call must use using='dense'."""
    client = MagicMock()
    client.query_points.return_value = _make_query_result([])

    pipeline, _, _ = _build_pipeline(client=client)
    pipeline.run(PipelineInput(query="test", top_k=3))

    for c in client.query_points.call_args_list:
        assert c.kwargs.get("using") == "dense"
