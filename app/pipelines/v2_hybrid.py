"""
Pipeline v2 — hybrid dense + sparse BM25 fusion, no rerank, no router.

For each of the three Qdrant collections (stats_meta, narratives, regulations),
a single ``client.query_points`` call issues a hybrid query via:
  - Prefetch dense (bge-large-en-v1.5)
  - Prefetch sparse (bm25)
  - FusionQuery(RRF) to merge them inside Qdrant

The per-collection ranked lists are then merged with a second RRF pass
(cross-collection fusion) before being truncated to top_k and passed to the
Sonnet synthesizer.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from qdrant_client import QdrantClient, models

from app.embedding import Embedders
from app.pipeline import PipelineInput, PipelineResult
from app.synth import ContextChunk, Synthesizer

logger = logging.getLogger(__name__)

COLLECTIONS = ("stats_meta", "narratives", "regulations")

# Map collection name → source tag used in ContextChunk
_COLLECTION_SOURCE: dict[str, str] = {
    "stats_meta": "stats",
    "narratives": "narratives",
    "regulations": "regulations",
}

# Standard RRF constant
_RRF_K = 60


def _rrf_merge(
    ranked_lists: list[list[Any]],
    *,
    k: int = _RRF_K,
) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of Qdrant ScoredPoints.

    Args:
        ranked_lists: Each element is an ordered list of ``ScoredPoint`` objects
                      (rank 0 = best).
        k: RRF smoothing constant (default 60).

    Returns:
        List of ``(ScoredPoint, rrf_score)`` sorted descending by score.
        Hits that appear in multiple lists accumulate scores; each point is
        identified by its ``id`` field.
    """
    scores: dict[Any, float] = {}
    hits_by_id: dict[Any, Any] = {}

    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list, start=1):
            pid = hit.id
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            hits_by_id[pid] = hit

    merged = [(hits_by_id[pid], score) for pid, score in scores.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged


class HybridPipeline:
    """Dense + sparse BM25 hybrid retrieval across three Qdrant collections."""

    def __init__(
        self,
        client: QdrantClient,
        embedders: Embedders,
        synthesizer: Synthesizer,
        *,
        fetch_k: int = 20,
    ) -> None:
        self.client = client
        self.embedders = embedders
        self.synthesizer = synthesizer
        self.fetch_k = fetch_k

    def _query_collection(
        self,
        collection: str,
        dense_vec: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> list[Any]:
        """Issue a single hybrid query against one collection."""
        result = self.client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=self.fetch_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=self.fetch_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.fetch_k,
            with_payload=True,
        )
        return result.points

    def run(self, input: PipelineInput) -> PipelineResult:
        query = input.query
        top_k = input.top_k

        # 1. Encode query with both embedders
        dense_vec = self.embedders.encode_dense(query)
        sparse_indices, sparse_values = self.embedders.encode_sparse(query)

        # 2. Per-collection hybrid queries
        ranked_lists: list[list[Any]] = []
        for collection in COLLECTIONS:
            hits = self._query_collection(
                collection, dense_vec, sparse_indices, sparse_values
            )
            logger.debug("Collection %s returned %d hits", collection, len(hits))
            ranked_lists.append(hits)

        # 3. Cross-collection RRF merge
        merged = _rrf_merge(ranked_lists, k=_RRF_K)

        # 4. Truncate to top_k
        top_hits = merged[:top_k]

        # 5. Convert to ContextChunk, drop hits without text
        contexts: list[ContextChunk] = []
        for hit, _score in top_hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            if not text:
                logger.debug("Dropping hit %s — missing text in payload", hit.id)
                continue
            chunk_id = payload.get("chunk_id", str(hit.id))
            source = payload.get("source", "unknown")
            metadata = {k: v for k, v in payload.items() if k not in {"text", "source", "chunk_id"}}
            contexts.append(
                ContextChunk(
                    cite_id=chunk_id,
                    source=source,
                    text=text,
                    metadata=metadata,
                )
            )

        # 6. Synthesize answer
        synth_result = self.synthesizer.synthesize(query, contexts)

        return PipelineResult(
            version="v2",
            query=query,
            answer=synth_result.answer,
            citations=synth_result.citations,
            refused=synth_result.refused,
            retrieved_contexts=contexts,
            tool_calls=[],
        )


def run_v2(
    query: str,
    top_k: int = 5,
    *,
    client: QdrantClient | None = None,
    embedders: Embedders | None = None,
    synthesizer: Synthesizer | None = None,
) -> PipelineResult:
    """Convenience function: construct HybridPipeline with defaults and run."""
    if client is None:
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)
    if embedders is None:
        embedders = Embedders()
    if synthesizer is None:
        synthesizer = Synthesizer()

    pipeline = HybridPipeline(client=client, embedders=embedders, synthesizer=synthesizer)
    return pipeline.run(PipelineInput(query=query, top_k=top_k))
