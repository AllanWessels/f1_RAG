"""
Hybrid (dense + sparse BM25) retrieval over the Qdrant ``narratives`` collection,
with bge-reranker-v2-m3 reranking on top.

Assumption (Issue note):
    Qdrant payload **must** contain a ``text`` field for each point so the reranker
    can operate on it.  The current ``ingestion/build_indexes.py::load_jsonl_chunks``
    strips ``text`` from the payload (line: ``{k: v for k, v in obj.items() if k != text_field}``).
    The orchestrator must adjust ``build_indexes.py`` to keep ``text`` in the payload
    before this retriever can work against a live collection.
    In the meantime, any hit whose payload lacks ``text`` is silently dropped.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from app.embedding import Embedders, Reranker
from app.tracing import traced

logger = logging.getLogger(__name__)

COLLECTION = "narratives"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class NarrativeHit(BaseModel):
    chunk_id: str
    score: float
    text: str
    year: int | None = None
    gp_name: str | None = None
    section_path: list[str] = []
    page_url: str | None = None


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------


class NarrativesRetriever:
    """Hybrid retriever over the ``narratives`` Qdrant collection.

    Parameters
    ----------
    qdrant_client:
        A ``QdrantClient`` instance.  When *None* a default client is built
        from the ``QDRANT_URL`` environment variable (default
        ``http://localhost:6333``).
    embedders:
        An ``Embedders`` instance for dense + sparse query encoding.  When
        *None* a fresh ``Embedders()`` is created (lazy model loading).
    reranker:
        A ``Reranker`` instance.  When *None* a fresh ``Reranker()`` is
        created (lazy model loading).
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        embedders: Embedders | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._client = qdrant_client
        self._embedders = embedders
        self._reranker = reranker

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            url = os.environ.get("QDRANT_URL", "http://localhost:6333")
            self._client = QdrantClient(url=url)
        return self._client

    @property
    def embedders(self) -> Embedders:
        if self._embedders is None:
            self._embedders = Embedders()
        return self._embedders

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    @traced(name="narratives_retriever_search")
    def search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 30,
    ) -> list[NarrativeHit]:
        """Return up to *top_k* reranked ``NarrativeHit`` objects.

        Pipeline
        --------
        1. Encode query → dense vector + sparse (BM25) vector.
        2. Run Qdrant hybrid query with RRF fusion; fetch *fetch_k* candidates.
        3. Drop hits whose payload has no ``text`` field (see module docstring).
        4. Rerank remaining candidates with bge-reranker-v2-m3.
        5. Map back to ``NarrativeHit`` objects using reranker scores.
        """
        # Step 1 — encode query
        dense_vec: list[float] = self.embedders.encode_dense(query)
        sparse_indices, sparse_values = self.embedders.encode_sparse(query)

        # Step 2 — hybrid Qdrant query
        response = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=fetch_k),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=fetch_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_k,
            with_payload=True,
        )

        hits: list[Any] = response.points  # ScoredPoint list

        # Step 3 — drop hits without text (see module-level assumption note)
        valid_hits = [h for h in hits if h.payload and "text" in h.payload]

        if not valid_hits:
            return []

        # Step 4 — rerank
        texts = [h.payload["text"] for h in valid_hits]
        ranked: list[tuple[int, float]] = self.reranker.rerank(query, texts, top_k=top_k)

        # Step 5 — map back to NarrativeHit
        results: list[NarrativeHit] = []
        for original_idx, score in ranked:
            hit = valid_hits[original_idx]
            payload: dict[str, Any] = hit.payload or {}
            results.append(
                NarrativeHit(
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                    score=score,
                    text=payload["text"],
                    year=payload.get("year"),
                    gp_name=payload.get("gp_name"),
                    section_path=payload.get("section_path", []),
                    page_url=payload.get("page_url"),
                )
            )

        return results


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_retriever: NarrativesRetriever | None = None


def search_narratives(
    query: str,
    top_k: int = 5,
    fetch_k: int = 30,
) -> list[NarrativeHit]:
    """Module-level convenience wrapper around :class:`NarrativesRetriever`.

    A single ``NarrativesRetriever`` instance is created on first call and
    reused for subsequent calls (lazy singleton pattern).
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = NarrativesRetriever()
    return _default_retriever.search(query, top_k=top_k, fetch_k=fetch_k)
