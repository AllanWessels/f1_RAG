"""
Unit tests for app.retrievers.regulations.

All external dependencies (Qdrant, Embedders, Reranker) are fully mocked.
No network calls; no model weights loaded.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.retrievers.regulations import RegulationHit, RegulationsRetriever

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_qdrant_hit(
    chunk_id: str,
    text: str,
    regulation_type: str = "technical",
    season: int = 2026,
    article_number: str | None = "4.2.1",
    article_path: list[str] | None = None,
    source_pdf: str | None = "FIA_2026_Technical.pdf",
    page_number: int | None = 12,
) -> SimpleNamespace:
    """Create a fake Qdrant ScoredPoint-like object."""
    payload = {
        "chunk_id": chunk_id,
        "text": text,
        "regulation_type": regulation_type,
        "season": season,
        "source_pdf": source_pdf,
        "page_number": page_number,
    }
    if article_number is not None:
        payload["article_number"] = article_number
    if article_path is not None:
        payload["article_path"] = article_path
    return SimpleNamespace(payload=payload, score=0.5)


def _make_qdrant_response(hits: list) -> SimpleNamespace:
    """Wrap a list of hits in a fake QueryResponse."""
    return SimpleNamespace(points=hits)


def _make_embedders(
    dense_vec: list[float] | None = None,
    sparse_indices: list[int] | None = None,
    sparse_values: list[float] | None = None,
) -> MagicMock:
    embedders = MagicMock()
    embedders.encode_dense.return_value = dense_vec or [0.1] * 1024
    embedders.encode_sparse.return_value = (
        sparse_indices or [0, 1, 2],
        sparse_values or [0.9, 0.5, 0.3],
    )
    return embedders


def _make_reranker(ranked: list[tuple[int, float]]) -> MagicMock:
    reranker = MagicMock()
    reranker.rerank.return_value = ranked
    return reranker


def _make_retriever(
    hits: list,
    ranked: list[tuple[int, float]],
    *,
    dense_vec: list[float] | None = None,
    sparse_indices: list[int] | None = None,
    sparse_values: list[float] | None = None,
) -> tuple[RegulationsRetriever, MagicMock, MagicMock, MagicMock]:
    """Convenience: build retriever + all mocks in one call."""
    client = MagicMock()
    client.query_points.return_value = _make_qdrant_response(hits)
    embedders = _make_embedders(dense_vec, sparse_indices, sparse_values)
    reranker = _make_reranker(ranked)
    retriever = RegulationsRetriever(
        qdrant_client=client,
        embedders=embedders,
        reranker=reranker,
    )
    return retriever, client, embedders, reranker


# ---------------------------------------------------------------------------
# Test 1: Happy path — 30 hits → reranker reorders → top-5 returned
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_returns_top_k_hits_in_reranker_order(self) -> None:
        hits = [
            _make_qdrant_hit(
                chunk_id=f"reg__{i}",
                text=f"Regulation text {i}",
                article_number=f"4.{i}",
                article_path=["Chapter 4", f"Article 4.{i}"],
                page_number=i + 1,
            )
            for i in range(30)
        ]
        # Reranker picks hits at indices 7, 2, 15, 0, 22 with descending scores
        ranked = [(7, 0.98), (2, 0.91), (15, 0.85), (0, 0.77), (22, 0.70)]
        retriever, _, _, _ = _make_retriever(hits, ranked)

        results = retriever.search("minimum car weight 2026", top_k=5, fetch_k=30)

        assert len(results) == 5
        # Scores come from reranker
        assert results[0].score == pytest.approx(0.98)
        assert results[1].score == pytest.approx(0.91)
        assert results[4].score == pytest.approx(0.70)
        # chunk_ids match reranker's ordering
        assert results[0].chunk_id == "reg__7"
        assert results[1].chunk_id == "reg__2"
        assert results[2].chunk_id == "reg__15"
        assert results[3].chunk_id == "reg__0"
        assert results[4].chunk_id == "reg__22"

    def test_payload_fields_mapped_correctly(self) -> None:
        hits = [
            _make_qdrant_hit(
                chunk_id="reg__0",
                text="The car must weigh at least 798 kg.",
                regulation_type="technical",
                season=2026,
                article_number="B1.1",
                article_path=["Part B", "Article B1", "B1.1"],
                source_pdf="FIA_2026_Technical.pdf",
                page_number=42,
            )
        ]
        ranked = [(0, 0.95)]
        retriever, _, _, _ = _make_retriever(hits, ranked)

        results = retriever.search("minimum weight", top_k=1, fetch_k=30)

        assert len(results) == 1
        hit = results[0]
        assert isinstance(hit, RegulationHit)
        assert hit.chunk_id == "reg__0"
        assert hit.text == "The car must weigh at least 798 kg."
        assert hit.regulation_type == "technical"
        assert hit.season == 2026
        assert hit.article_number == "B1.1"
        assert hit.article_path == ["Part B", "Article B1", "B1.1"]
        assert hit.source_pdf == "FIA_2026_Technical.pdf"
        assert hit.page_number == 42
        assert hit.score == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Test 2: Empty Qdrant returns [] without calling reranker
# ---------------------------------------------------------------------------


class TestEmptyQdrant:
    def test_empty_result_no_reranker_call(self) -> None:
        retriever, _, _, reranker = _make_retriever(hits=[], ranked=[])

        results = retriever.search("some query", top_k=5, fetch_k=30)

        assert results == []
        reranker.rerank.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Embedders called once each; query_points called once with both Prefetches
# ---------------------------------------------------------------------------


class TestCallCounts:
    def test_embedders_called_once_each_and_qdrant_once(self) -> None:
        from qdrant_client import models

        hits = [_make_qdrant_hit("reg__0", "Some text")]
        ranked = [(0, 0.9)]
        retriever, client, embedders, _ = _make_retriever(
            hits,
            ranked,
            dense_vec=[0.2] * 1024,
            sparse_indices=[10, 20],
            sparse_values=[0.8, 0.4],
        )

        retriever.search("fuel flow rate", top_k=1, fetch_k=30)

        # Embedders called exactly once each
        embedders.encode_dense.assert_called_once_with("fuel flow rate")
        embedders.encode_sparse.assert_called_once_with("fuel flow rate")

        # Qdrant called exactly once
        client.query_points.assert_called_once()
        call_kwargs = client.query_points.call_args.kwargs

        assert call_kwargs["collection_name"] == "regulations"
        assert call_kwargs["limit"] == 30

        prefetches = call_kwargs["prefetch"]
        assert len(prefetches) == 2

        # Verify dense prefetch
        dense_pf = prefetches[0]
        assert dense_pf.using == "dense"
        assert dense_pf.query == [0.2] * 1024
        assert dense_pf.limit == 30

        # Verify sparse prefetch
        sparse_pf = prefetches[1]
        assert sparse_pf.using == "sparse"
        assert isinstance(sparse_pf.query, models.SparseVector)
        assert list(sparse_pf.query.indices) == [10, 20]
        assert list(sparse_pf.query.values) == [0.8, 0.4]
        assert sparse_pf.limit == 30

        # RRF fusion
        fusion_query = call_kwargs["query"]
        assert isinstance(fusion_query, models.FusionQuery)
        assert fusion_query.fusion == models.Fusion.RRF


# ---------------------------------------------------------------------------
# Test 4: Reranker called with query + list of texts from payloads
# ---------------------------------------------------------------------------


class TestRerankerInput:
    def test_reranker_receives_correct_query_and_texts(self) -> None:
        texts = [f"Regulation passage {i}" for i in range(5)]
        hits = [_make_qdrant_hit(f"reg__{i}", texts[i]) for i in range(5)]
        ranked = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)]
        retriever, _, _, reranker = _make_retriever(hits, ranked)

        retriever.search("tyre compound rules", top_k=5, fetch_k=30)

        reranker.rerank.assert_called_once()
        call_args = reranker.rerank.call_args
        assert call_args.args[0] == "tyre compound rules"
        assert call_args.args[1] == texts
        assert call_args.kwargs.get("top_k") == 5 or call_args.args[2] == 5


# ---------------------------------------------------------------------------
# Test 5: top_k=3 truncates correctly
# ---------------------------------------------------------------------------


class TestTopKTruncation:
    def test_top_k_3_returns_3_hits(self) -> None:
        hits = [_make_qdrant_hit(f"reg__{i}", f"Text {i}") for i in range(10)]
        # Reranker already returns only 3 per top_k
        ranked = [(2, 0.95), (5, 0.88), (0, 0.72)]
        retriever, _, _, reranker = _make_retriever(hits, ranked)

        results = retriever.search("DRS zone length", top_k=3, fetch_k=30)

        assert len(results) == 3
        # Verify top_k was forwarded to reranker
        reranker.rerank.assert_called_once()
        call_args = reranker.rerank.call_args
        top_k_passed = (
            call_args.kwargs.get("top_k")
            if call_args.kwargs.get("top_k") is not None
            else call_args.args[2]
        )
        assert top_k_passed == 3


# ---------------------------------------------------------------------------
# Test 6: Hits without `text` in payload are silently dropped
# ---------------------------------------------------------------------------


class TestMissingTextDropped:
    def test_hits_without_text_are_dropped(self) -> None:
        good_hit = _make_qdrant_hit("reg__good", "Valid regulation text")
        # Hit with no text in payload
        bad_hit = SimpleNamespace(
            payload={"chunk_id": "reg__bad", "regulation_type": "sporting"},
            score=0.4,
        )
        # Hit with empty string text
        empty_hit = SimpleNamespace(
            payload={"chunk_id": "reg__empty", "text": "", "regulation_type": "sporting"},
            score=0.3,
        )

        hits = [bad_hit, empty_hit, good_hit]
        ranked = [(0, 0.92)]  # only one valid hit remains after filtering

        retriever, _, _, reranker = _make_retriever(hits, ranked)

        results = retriever.search("any query", top_k=5, fetch_k=30)

        # Only the good hit survives
        assert len(results) == 1
        assert results[0].chunk_id == "reg__good"

        # Reranker was called with only the valid text
        reranker.rerank.assert_called_once()
        texts_passed = reranker.rerank.call_args.args[1]
        assert texts_passed == ["Valid regulation text"]

    def test_all_hits_without_text_returns_empty(self) -> None:
        hits = [
            SimpleNamespace(
                payload={"chunk_id": f"reg__{i}", "regulation_type": "financial"},
                score=0.5,
            )
            for i in range(5)
        ]
        ranked = []
        retriever, _, _, reranker = _make_retriever(hits, ranked)

        results = retriever.search("prize money distribution", top_k=5, fetch_k=30)

        assert results == []
        reranker.rerank.assert_not_called()


# ---------------------------------------------------------------------------
# Test 7: article_path list is preserved correctly
# ---------------------------------------------------------------------------


class TestArticlePathPreserved:
    def test_multi_element_article_path_preserved(self) -> None:
        article_path = ["Part C - Financial Regulations", "Article C4", "C4.1 - Revenue"]
        hits = [
            _make_qdrant_hit(
                chunk_id="reg__fin_01",
                text="Revenue sharing among constructors.",
                regulation_type="financial",
                season=2026,
                article_number="C4.1",
                article_path=article_path,
                source_pdf="FIA_2026_Financial.pdf",
                page_number=88,
            )
        ]
        ranked = [(0, 0.87)]
        retriever, _, _, _ = _make_retriever(hits, ranked)

        results = retriever.search("prize money distribution", top_k=1, fetch_k=30)

        assert len(results) == 1
        assert results[0].article_path == [
            "Part C - Financial Regulations",
            "Article C4",
            "C4.1 - Revenue",
        ]

    def test_empty_article_path_defaults_to_empty_list(self) -> None:
        hits = [
            _make_qdrant_hit(
                chunk_id="reg__sport_01",
                text="Safety car procedures.",
                regulation_type="sporting",
                article_number="55.1",
                article_path=[],
            )
        ]
        ranked = [(0, 0.80)]
        retriever, _, _, _ = _make_retriever(hits, ranked)

        results = retriever.search("safety car rules", top_k=1, fetch_k=30)

        assert results[0].article_path == []

    def test_missing_article_path_in_payload_defaults_to_empty_list(self) -> None:
        hit = SimpleNamespace(
            payload={
                "chunk_id": "reg__sport_02",
                "text": "Race director authority.",
                "regulation_type": "sporting",
                "article_number": "11.1",
                "source_pdf": "FIA_2026_Sporting.pdf",
                "page_number": 5,
            },
            score=0.6,
        )
        ranked = [(0, 0.75)]
        retriever, _, _, _ = _make_retriever([hit], ranked)

        results = retriever.search("race director", top_k=1, fetch_k=30)

        assert results[0].article_path == []


# ---------------------------------------------------------------------------
# Additional: functional interface smoke test (module-level singletons mocked)
# ---------------------------------------------------------------------------


class TestFunctionalInterface:
    def test_search_regulations_function_uses_singletons(self, monkeypatch) -> None:
        """search_regulations() should call _run_search with lazy singletons."""
        import app.retrievers.regulations as reg_module

        fake_hit = RegulationHit(
            chunk_id="reg__x",
            score=0.9,
            text="Some regulation",
            regulation_type="technical",
            article_path=[],
        )

        mock_run = MagicMock(return_value=[fake_hit])
        monkeypatch.setattr(reg_module, "_run_search", mock_run)

        # Also stub the singleton getters so no real clients are created
        monkeypatch.setattr(reg_module, "_get_client", MagicMock(return_value="client"))
        monkeypatch.setattr(
            reg_module, "_get_embedders", MagicMock(return_value="embedders")
        )
        monkeypatch.setattr(
            reg_module, "_get_reranker", MagicMock(return_value="reranker")
        )

        from app.retrievers.regulations import search_regulations

        results = search_regulations("minimum weight", top_k=3, fetch_k=20)

        assert results == [fake_hit]
        mock_run.assert_called_once_with(
            client="client",
            embedders="embedders",
            reranker="reranker",
            query="minimum weight",
            top_k=3,
            fetch_k=20,
        )
