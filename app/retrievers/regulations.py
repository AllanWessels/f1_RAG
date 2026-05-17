"""
Regulations retriever: hybrid dense + sparse BM25 retrieval over the Qdrant
`regulations` collection, with bge-reranker-v2-m3 reranking.

Exposes:
  - search_regulations(query, top_k, fetch_k) -> list[RegulationHit]  (functional)
  - RegulationsRetriever(qdrant_client, embedders, reranker).search(...)  (class form)

Assumption: each Qdrant point payload contains a `text` field.  The ingestion
orchestrator is responsible for ensuring text is stored in the payload
(build_indexes.py currently strips it; the orchestrator will adjust that).
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

COLLECTION_NAME = "regulations"

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class RegulationHit(BaseModel):
    chunk_id: str
    score: float
    text: str
    regulation_type: str | None = None  # "sporting" | "technical" | "financial"
    season: int | None = None
    article_number: str | None = None  # e.g. "B1.1", "4.2.1", "C4.1"
    article_path: list[str] = []
    source_pdf: str | None = None
    page_number: int | None = None


# ---------------------------------------------------------------------------
# Module-level lazy singletons (used by the functional interface)
# ---------------------------------------------------------------------------

_client: QdrantClient | None = None
_embedders: Embedders | None = None
_reranker: Reranker | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        _client = QdrantClient(url=url)
    return _client


def _get_embedders() -> Embedders:
    global _embedders
    if _embedders is None:
        _embedders = Embedders()
    return _embedders


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


# ---------------------------------------------------------------------------
# Core retrieval logic
# ---------------------------------------------------------------------------


def _payload_to_hit(payload: dict[str, Any], score: float) -> RegulationHit:
    """Map a Qdrant point payload + reranker score to a RegulationHit."""
    return RegulationHit(
        chunk_id=payload.get("chunk_id", ""),
        score=score,
        text=payload.get("text", ""),
        regulation_type=payload.get("regulation_type"),
        season=payload.get("season"),
        article_number=payload.get("article_number"),
        article_path=payload.get("article_path", []),
        source_pdf=payload.get("source_pdf"),
        page_number=payload.get("page_number"),
    )


def _run_search(
    client: QdrantClient,
    embedders: Embedders,
    reranker: Reranker,
    query: str,
    top_k: int,
    fetch_k: int,
) -> list[RegulationHit]:
    """Inner implementation: hybrid query → rerank → map to RegulationHit."""
    # 1. Encode query
    dense_vec = embedders.encode_dense(query)
    sparse_indices, sparse_values = embedders.encode_sparse(query)

    # 2. Hybrid Qdrant query with RRF fusion
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=fetch_k,
            ),
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

    hits = response.points

    # 3. Guard: return empty list if no results (don't call reranker)
    if not hits:
        return []

    # 4. Filter out hits that have no `text` in payload; keep the rest
    valid_hits = [h for h in hits if h.payload and h.payload.get("text")]
    if not valid_hits:
        return []

    # 5. Rerank
    texts = [h.payload["text"] for h in valid_hits]
    ranked = reranker.rerank(query, texts, top_k=top_k)

    # 6. Map back to RegulationHit using reranker score
    results: list[RegulationHit] = []
    for original_idx, score in ranked:
        hit = valid_hits[original_idx]
        results.append(_payload_to_hit(hit.payload, score))

    return results


# ---------------------------------------------------------------------------
# Class interface
# ---------------------------------------------------------------------------


class RegulationsRetriever:
    """
    Class-based regulations retriever.

    Args:
        qdrant_client: Pre-constructed QdrantClient.
        embedders:     Embedders instance (dense + sparse).
        reranker:      Reranker instance (bge-reranker-v2-m3).
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedders: Embedders,
        reranker: Reranker,
    ) -> None:
        self._client = qdrant_client
        self._embedders = embedders
        self._reranker = reranker

    @traced(name="regulations_retriever_search")
    def search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 30,
    ) -> list[RegulationHit]:
        """
        Search the regulations collection.

        Args:
            query:   Natural-language query.
            top_k:   Number of results to return after reranking.
            fetch_k: Number of candidates to retrieve from Qdrant before reranking.

        Returns:
            List of RegulationHit sorted by reranker score (descending).
        """
        return _run_search(
            client=self._client,
            embedders=self._embedders,
            reranker=self._reranker,
            query=query,
            top_k=top_k,
            fetch_k=fetch_k,
        )


# ---------------------------------------------------------------------------
# Functional interface
# ---------------------------------------------------------------------------


def search_regulations(
    query: str,
    top_k: int = 5,
    fetch_k: int = 30,
) -> list[RegulationHit]:
    """
    Search the regulations collection using hybrid retrieval + reranking.

    Args:
        query:   Natural-language query.
        top_k:   Number of results to return after reranking.
        fetch_k: Number of candidates to retrieve from Qdrant before reranking.

    Returns:
        List of RegulationHit sorted by reranker score (descending).
    """
    return _run_search(
        client=_get_client(),
        embedders=_get_embedders(),
        reranker=_get_reranker(),
        query=query,
        top_k=top_k,
        fetch_k=fetch_k,
    )
