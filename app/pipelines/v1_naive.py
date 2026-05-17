"""
Pipeline v1 — Naive dense-only baseline.

For each of the three Qdrant collections (stats_meta, narratives, regulations),
issues a pure dense vector query (no sparse, no fusion/prefetch). Results are
merged across collections, sorted by raw cosine score descending, and the top-k
are passed to the Synthesizer.

No reranking. No router. Just vanilla dense retrieval.
"""

from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient

from app.embedding import Embedders
from app.pipeline import PipelineInput, PipelineResult
from app.synth import ContextChunk, Synthesizer

logger = logging.getLogger(__name__)

COLLECTIONS = ("stats_meta", "narratives", "regulations")

# Prefix letters for cite_id construction (one per collection, same order).
_CITE_PREFIX: dict[str, str] = {
    "stats_meta": "s",
    "narratives": "n",
    "regulations": "r",
}


def _make_cite_id(collection: str, chunk_id: str) -> str:
    """Build a short, stable cite_id that stays well under 50 chars."""
    prefix = _CITE_PREFIX.get(collection, collection[0])
    raw = f"{prefix}_{chunk_id}"
    # Trim to 49 chars to stay safely under the 50-char ceiling.
    return raw[:49]


class NaivePipeline:
    """Dense-only retrieval pipeline (v1 baseline)."""

    def __init__(
        self,
        client: QdrantClient,
        embedders: Embedders,
        synthesizer: Synthesizer,
    ) -> None:
        self.client = client
        self.embedders = embedders
        self.synthesizer = synthesizer

    def run(self, input: PipelineInput) -> PipelineResult:
        query = input.query
        top_k = input.top_k

        # 1. Encode the query as a dense vector.
        dense_vec = self.embedders.encode_dense(query)

        # 2. Query each collection independently with pure dense retrieval.
        all_hits: list[tuple[float, str, dict]] = []  # (score, collection, payload)
        for collection in COLLECTIONS:
            results = self.client.query_points(
                collection_name=collection,
                query=dense_vec,
                using="dense",
                limit=top_k,
                with_payload=True,
            )
            for point in results.points:
                payload = dict(point.payload) if point.payload else {}
                all_hits.append((point.score, collection, payload))

        # 3. Merge and sort by score descending, truncate to top_k.
        all_hits.sort(key=lambda t: t[0], reverse=True)
        all_hits = all_hits[:top_k]

        # 4. Convert surviving hits to ContextChunk objects; drop hits with no text.
        contexts: list[ContextChunk] = []
        seen_cite_ids: set[str] = set()
        for _score, collection, payload in all_hits:
            text = payload.get("text", "")
            if not text:
                continue
            chunk_id = str(payload.get("chunk_id", id(payload)))
            cite_id = _make_cite_id(collection, chunk_id)
            # Deduplicate cite_ids (shouldn't happen in practice but be safe).
            original_cite_id = cite_id
            suffix = 0
            while cite_id in seen_cite_ids:
                suffix += 1
                cite_id = f"{original_cite_id[:46]}_{suffix}"
            seen_cite_ids.add(cite_id)

            source = payload.get("source", collection)
            metadata = {
                k: v for k, v in payload.items() if k not in {"text", "source", "chunk_id"}
            }
            contexts.append(
                ContextChunk(
                    cite_id=cite_id,
                    source=source,
                    text=text,
                    metadata=metadata,
                )
            )

        # 5. Synthesize an answer.
        synth_result = self.synthesizer.synthesize(query, contexts)

        return PipelineResult(
            version="v1",
            query=query,
            answer=synth_result.answer,
            citations=synth_result.citations,
            refused=synth_result.refused,
            retrieved_contexts=contexts,
            tool_calls=[],
        )


def run_v1(
    query: str,
    top_k: int = 5,
    *,
    client: QdrantClient | None = None,
    embedders: Embedders | None = None,
    synthesizer: Synthesizer | None = None,
) -> PipelineResult:
    """Convenience factory: builds defaults from QDRANT_URL env if not provided."""
    if client is None:
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)
    if embedders is None:
        embedders = Embedders()
    if synthesizer is None:
        synthesizer = Synthesizer()

    pipeline = NaivePipeline(client=client, embedders=embedders, synthesizer=synthesizer)
    return pipeline.run(PipelineInput(query=query, top_k=top_k))
