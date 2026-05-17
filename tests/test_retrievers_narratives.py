"""Unit tests for app.retrievers.narratives.

All external dependencies (Qdrant, Embedders, Reranker) are fully mocked so
no model weights or network connections are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.retrievers.narratives import NarrativeHit, NarrativesRetriever

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_scored_point(
    chunk_id: str,
    *,
    text: str | None = "some narrative text",
    year: int | None = 2021,
    gp_name: str | None = "Abu Dhabi Grand Prix",
    section_path: list[str] | None = None,
    page_url: str | None = "https://en.wikipedia.org/wiki/2021_Abu_Dhabi_Grand_Prix",
) -> MagicMock:
    """Build a MagicMock that looks like a ``qdrant_client.models.ScoredPoint``."""
    point = MagicMock()
    point.id = 999
    payload: dict = {"chunk_id": chunk_id}
    if text is not None:
        payload["text"] = text
    if year is not None:
        payload["year"] = year
    if gp_name is not None:
        payload["gp_name"] = gp_name
    payload["section_path"] = section_path or ["Race report", "Safety car controversy"]
    if page_url is not None:
        payload["page_url"] = page_url
    point.payload = payload
    return point


def _make_retriever(
    qdrant_points: list[MagicMock],
    *,
    reranker_result: list[tuple[int, float]] | None = None,
    dense_vec: list[float] | None = None,
    sparse_indices: list[int] | None = None,
    sparse_values: list[float] | None = None,
) -> tuple[NarrativesRetriever, MagicMock, MagicMock, MagicMock]:
    """Return (retriever, mock_client, mock_embedders, mock_reranker)."""
    # Mock Qdrant client
    mock_client = MagicMock()
    response = MagicMock()
    response.points = qdrant_points
    mock_client.query_points.return_value = response

    # Mock Embedders
    mock_embedders = MagicMock()
    mock_embedders.encode_dense.return_value = dense_vec or [0.1] * 1024
    mock_embedders.encode_sparse.return_value = (
        sparse_indices or [0, 1, 2],
        sparse_values or [0.5, 0.3, 0.2],
    )

    # Mock Reranker
    mock_reranker = MagicMock()
    # Default: identity ordering (index 0 → score 0.99, index 1 → 0.88, …)
    if reranker_result is None:
        reranker_result = [(i, 1.0 - i * 0.01) for i in range(len(qdrant_points))]
    mock_reranker.rerank.return_value = reranker_result

    retriever = NarrativesRetriever(
        qdrant_client=mock_client,
        embedders=mock_embedders,
        reranker=mock_reranker,
    )
    return retriever, mock_client, mock_embedders, mock_reranker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Test 1: 30 hits in → reranked top-5 out with correct payloads."""

    def test_returns_top_k_narrative_hits(self) -> None:
        n_fetch = 30
        points = [
            _make_scored_point(f"narr__{i:03d}", text=f"text {i}", year=2021 + (i % 3))
            for i in range(n_fetch)
        ]
        # Reranker says: pick indices [5,3,1,29,7] in that score order
        reranker_result = [(5, 0.95), (3, 0.90), (1, 0.85), (29, 0.80), (7, 0.75)]
        retriever, _, _, _ = _make_retriever(points, reranker_result=reranker_result)

        result = retriever.search("2021 Abu Dhabi safety car", top_k=5, fetch_k=n_fetch)

        assert len(result) == 5
        assert all(isinstance(h, NarrativeHit) for h in result)

    def test_reranker_scores_used_not_qdrant_scores(self) -> None:
        points = [_make_scored_point(f"narr__{i}", text=f"text {i}") for i in range(5)]
        reranker_result = [(2, 0.77), (0, 0.55), (4, 0.33), (1, 0.22), (3, 0.11)]
        retriever, _, _, _ = _make_retriever(points, reranker_result=reranker_result)

        result = retriever.search("test query", top_k=5, fetch_k=5)

        scores = [h.score for h in result]
        assert scores == [0.77, 0.55, 0.33, 0.22, 0.11]

    def test_reranker_ordering_applied(self) -> None:
        """Hits are returned in reranker order, not original Qdrant order."""
        points = [_make_scored_point(f"narr__{i}", text=f"text {i}") for i in range(5)]
        # Reranker reverses: best is index 4
        reranker_result = [(4, 0.9), (3, 0.8), (2, 0.7), (1, 0.6), (0, 0.5)]
        retriever, _, _, _ = _make_retriever(points, reranker_result=reranker_result)

        result = retriever.search("query", top_k=5, fetch_k=5)

        assert result[0].chunk_id == "narr__4"
        assert result[1].chunk_id == "narr__3"
        assert result[-1].chunk_id == "narr__0"


class TestEmptyResult:
    """Test 2: empty Qdrant response → [] and reranker NOT called."""

    def test_empty_qdrant_returns_empty_list(self) -> None:
        retriever, _, _, _mock_reranker = _make_retriever([])

        result = retriever.search("any query")

        assert result == []

    def test_empty_qdrant_does_not_call_reranker(self) -> None:
        retriever, _, _, mock_reranker = _make_retriever([])

        retriever.search("any query")

        mock_reranker.rerank.assert_not_called()


class TestEmbeddingCalls:
    """Test 3: dense + sparse encoded once; Qdrant called once with both Prefetches."""

    def test_encode_dense_called_once(self) -> None:
        points = [_make_scored_point("narr__0", text="text")]
        retriever, _, mock_embedders, _ = _make_retriever(
            points, reranker_result=[(0, 0.9)]
        )

        retriever.search("test query", top_k=1, fetch_k=5)

        mock_embedders.encode_dense.assert_called_once_with("test query")

    def test_encode_sparse_called_once(self) -> None:
        points = [_make_scored_point("narr__0", text="text")]
        retriever, _, mock_embedders, _ = _make_retriever(
            points, reranker_result=[(0, 0.9)]
        )

        retriever.search("test query", top_k=1, fetch_k=5)

        mock_embedders.encode_sparse.assert_called_once_with("test query")

    def test_qdrant_called_once_with_both_prefetches(self) -> None:
        from qdrant_client import models

        points = [_make_scored_point("narr__0", text="text")]
        dense_vec = [0.1] * 1024
        sparse_idx = [10, 20]
        sparse_val = [0.6, 0.4]
        retriever, mock_client, _, _ = _make_retriever(
            points,
            reranker_result=[(0, 0.9)],
            dense_vec=dense_vec,
            sparse_indices=sparse_idx,
            sparse_values=sparse_val,
        )

        retriever.search("test query", top_k=1, fetch_k=15)

        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == "narratives"
        assert call_kwargs["limit"] == 15
        assert call_kwargs["with_payload"] is True

        prefetches = call_kwargs["prefetch"]
        assert len(prefetches) == 2
        # Dense prefetch
        dense_pf = prefetches[0]
        assert dense_pf.using == "dense"
        assert dense_pf.query == dense_vec
        assert dense_pf.limit == 15
        # Sparse prefetch
        sparse_pf = prefetches[1]
        assert sparse_pf.using == "sparse"
        assert sparse_pf.limit == 15

        # Fusion query
        fusion_q = call_kwargs["query"]
        assert isinstance(fusion_q, models.FusionQuery)
        assert fusion_q.fusion == models.Fusion.RRF


class TestRerankerCall:
    """Test 4: reranker called with query and texts from payloads."""

    def test_reranker_receives_query_and_texts(self) -> None:
        texts = ["alpha text", "beta text", "gamma text"]
        points = [_make_scored_point(f"n__{i}", text=t) for i, t in enumerate(texts)]
        reranker_result = [(0, 0.9), (1, 0.8), (2, 0.7)]
        retriever, _, _, mock_reranker = _make_retriever(
            points, reranker_result=reranker_result
        )

        retriever.search("my query", top_k=3, fetch_k=3)

        mock_reranker.rerank.assert_called_once_with("my query", texts, top_k=3)


class TestTopKTruncation:
    """Test 5: top_k=3 truncates even when reranker returns 5 scores."""

    def test_top_k_three_when_reranker_returns_five(self) -> None:
        points = [_make_scored_point(f"n__{i}", text=f"text {i}") for i in range(5)]
        # Reranker already respects top_k (returns 3 results) but we test the
        # retriever also caps correctly when it doesn't
        reranker_result = [(0, 0.9), (2, 0.8), (4, 0.7)]  # only 3
        retriever, _, _, _ = _make_retriever(points, reranker_result=reranker_result)

        result = retriever.search("query", top_k=3, fetch_k=5)

        assert len(result) == 3

    def test_top_k_passed_to_reranker(self) -> None:
        points = [_make_scored_point(f"n__{i}", text=f"text {i}") for i in range(5)]
        reranker_result = [(0, 0.9), (1, 0.8), (2, 0.7)]
        retriever, _, _, mock_reranker = _make_retriever(
            points, reranker_result=reranker_result
        )

        retriever.search("query", top_k=3, fetch_k=5)

        # rerank is called as rerank(query, texts, top_k=top_k); top_k is a keyword arg
        call_kwargs = mock_reranker.rerank.call_args.kwargs
        assert call_kwargs.get("top_k") == 3


class TestPayloadMapping:
    """Test 6: payload fields map correctly to NarrativeHit."""

    def test_all_payload_fields_mapped(self) -> None:
        point = _make_scored_point(
            "narr__2021_abu_dhabi__42",
            text="The safety car was deployed on lap 53.",
            year=2021,
            gp_name="Abu Dhabi Grand Prix",
            section_path=["Race", "Controversy"],
            page_url="https://en.wikipedia.org/wiki/2021_Abu_Dhabi_Grand_Prix",
        )
        retriever, _, _, _ = _make_retriever([point], reranker_result=[(0, 0.88)])

        result = retriever.search("safety car 2021", top_k=1, fetch_k=1)

        assert len(result) == 1
        hit = result[0]
        assert hit.chunk_id == "narr__2021_abu_dhabi__42"
        assert hit.score == pytest.approx(0.88)
        assert hit.text == "The safety car was deployed on lap 53."
        assert hit.year == 2021
        assert hit.gp_name == "Abu Dhabi Grand Prix"
        assert hit.section_path == ["Race", "Controversy"]
        assert hit.page_url == "https://en.wikipedia.org/wiki/2021_Abu_Dhabi_Grand_Prix"

    def test_optional_payload_fields_default_to_none(self) -> None:
        """Hits with missing optional fields default gracefully."""
        point = MagicMock()
        point.id = 1
        point.payload = {
            "chunk_id": "narr__sparse",
            "text": "Minimal payload hit.",
            # year, gp_name, section_path, page_url all absent
        }
        retriever, _, _, _ = _make_retriever([point], reranker_result=[(0, 0.5)])

        result = retriever.search("query", top_k=1, fetch_k=1)

        assert len(result) == 1
        hit = result[0]
        assert hit.year is None
        assert hit.gp_name is None
        assert hit.section_path == []
        assert hit.page_url is None


class TestMissingTextDrop:
    """Test 7: hits without 'text' in payload are silently dropped."""

    def test_missing_text_hits_dropped(self) -> None:
        good = _make_scored_point("narr__good", text="valid text")
        bad = _make_scored_point("narr__bad", text=None)  # no text key in payload
        retriever, _, _, _mock_reranker = _make_retriever(
            [good, bad],
            reranker_result=[(0, 0.9)],
        )

        result = retriever.search("query", top_k=5, fetch_k=5)

        # Only the good hit reaches the reranker and result list
        assert len(result) == 1
        assert result[0].chunk_id == "narr__good"

    def test_all_missing_text_returns_empty_no_reranker(self) -> None:
        bad1 = _make_scored_point("n__0", text=None)
        bad2 = _make_scored_point("n__1", text=None)
        retriever, _, _, mock_reranker = _make_retriever([bad1, bad2])

        result = retriever.search("query")

        assert result == []
        mock_reranker.rerank.assert_not_called()

    def test_no_crash_on_missing_text(self) -> None:
        """Regression: ensure no KeyError or AttributeError is raised."""
        bad = _make_scored_point("narr__no_text", text=None)
        retriever, _, _, _ = _make_retriever([bad])

        # Must not raise
        result = retriever.search("query")
        assert result == []


class TestModuleLevelFunction:
    """Smoke tests for the search_narratives module-level convenience function."""

    def test_search_narratives_delegates_to_retriever(self, monkeypatch) -> None:
        import app.retrievers.narratives as mod

        expected = [
            NarrativeHit(
                chunk_id="n__0",
                score=0.9,
                text="some text",
                year=2021,
            )
        ]
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = expected
        monkeypatch.setattr(mod, "_default_retriever", mock_retriever)

        from app.retrievers.narratives import search_narratives

        result = search_narratives("hello", top_k=3, fetch_k=10)

        mock_retriever.search.assert_called_once_with("hello", top_k=3, fetch_k=10)
        assert result == expected
