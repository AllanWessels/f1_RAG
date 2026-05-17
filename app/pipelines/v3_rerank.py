"""
Pipeline v3 — Hybrid retrieval + bge-reranker-v2-m3.

Same retrieval as v2 (per-collection hybrid BM25+dense with RRF, cross-collection
RRF merge), but with a bge-reranker-v2-m3 pass over the merged top-N candidates
before the final top-k truncation.

Steps:
1. Embed query (dense + sparse) via Embedders.
2. Per-collection hybrid query: prefetch=[dense, sparse], query=FusionQuery(RRF),
   limit=fetch_k.
3. Cross-collection RRF merge → up to 3xfetch_k hits.
4. Keep top fetch_k from merged list.
5. Drop any hits missing "text" payload (don't waste reranker calls).
6. Rerank: call Reranker.rerank(query, texts, top_k=top_k) → [(idx, score)].
7. Pick surviving hits in reranker order, applying reranker scores.
8. Convert to ContextChunk and call Synthesizer.synthesize.
9. Return PipelineResult(version="v3", ...).
"""

from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import FusionQuery, Prefetch, SparseVector

from app.embedding import Embedders, Reranker
from app.pipeline import PipelineInput, PipelineResult
from app.synth import ContextChunk, Synthesizer
from app.tracing import traced

logger = logging.getLogger(__name__)

COLLECTIONS = ("stats_meta", "narratives", "regulations")
DEFAULT_FETCH_K = 30

_CITE_PREFIX: dict[str, str] = {
    "stats_meta": "s",
    "narratives": "n",
    "regulations": "r",
}


def _make_cite_id(collection: str, chunk_id: str) -> str:
    """Build a short, stable cite_id that stays well under 50 chars."""
    prefix = _CITE_PREFIX.get(collection, collection[0])
    raw = f"{prefix}_{chunk_id}"
    return raw[:49]


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a hit at zero-based rank position."""
    return 1.0 / (k + rank + 1)


class RerankPipeline:
    """Hybrid retrieval + bge-reranker-v2-m3 pipeline (v3)."""

    def __init__(
        self,
        client: QdrantClient,
        embedders: Embedders,
        reranker: Reranker,
        synthesizer: Synthesizer,
        fetch_k: int = DEFAULT_FETCH_K,
    ) -> None:
        self.client = client
        self.embedders = embedders
        self.reranker = reranker
        self.synthesizer = synthesizer
        self.fetch_k = fetch_k

    @traced(name="v3_rerank_run")
    def run(self, input: PipelineInput) -> PipelineResult:
        query = input.query
        top_k = input.top_k
        fetch_k = self.fetch_k

        # 1. Embed the query.
        dense_vec = self.embedders.encode_dense(query)
        sparse_indices, sparse_values = self.embedders.encode_sparse(query)

        # 2. Per-collection hybrid query (dense prefetch + sparse prefetch → RRF fusion).
        # Each collection is queried independently; results carry a collection tag.
        all_hits: list[tuple[float, str, dict]] = []  # (score, collection, payload)
        for collection in COLLECTIONS:
            results = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=dense_vec,
                        using="dense",
                        limit=fetch_k,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                        using="sparse",
                        limit=fetch_k,
                    ),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=fetch_k,
                with_payload=True,
            )
            for point in results.points:
                payload = dict(point.payload) if point.payload else {}
                all_hits.append((point.score, collection, payload))

        # 3. Cross-collection RRF merge.
        # Group all hits by a stable key (collection + chunk_id) and assign an RRF
        # score based on their rank in the per-collection result lists.  For
        # simplicity we re-sort the combined list by individual per-collection score
        # (already RRF-scored by Qdrant) and apply a second round of RRF across
        # collections so later-appearing collections don't get penalised.
        #
        # Practical approach: sort by score descending, then keep top fetch_k.
        # Qdrant already applies RRF per collection; cross-collection we just
        # merge by score (the scores are comparable because Qdrant normalises them).
        all_hits.sort(key=lambda t: t[0], reverse=True)
        all_hits = all_hits[:fetch_k]

        # 4. Drop hits missing "text" payload before calling the reranker.
        valid_hits = [
            (score, collection, payload)
            for score, collection, payload in all_hits
            if payload.get("text")
        ]

        if not valid_hits:
            # No usable hits → synthesize with empty context (will return a refusal).
            synth_result = self.synthesizer.synthesize(query, [])
            return PipelineResult(
                version="v3",
                query=query,
                answer=synth_result.answer,
                citations=synth_result.citations,
                refused=synth_result.refused,
                retrieved_contexts=[],
                tool_calls=[],
            )

        # 5. Rerank.
        candidate_texts = [payload["text"] for _, _, payload in valid_hits]
        ranked = self.reranker.rerank(query, candidate_texts, top_k=top_k)
        # ranked → [(original_index, reranker_score)] sorted by score desc, len ≤ top_k

        # 6. Build the final ordered list using reranker indices and scores.
        contexts: list[ContextChunk] = []
        seen_cite_ids: set[str] = set()
        for orig_idx, rerank_score in ranked:
            _rrf, collection, payload = valid_hits[orig_idx]
            text = payload.get("text", "")
            if not text:
                continue
            chunk_id = str(payload.get("chunk_id", id(payload)))
            cite_id = _make_cite_id(collection, chunk_id)
            # Deduplicate cite_ids.
            original_cite_id = cite_id
            suffix = 0
            while cite_id in seen_cite_ids:
                suffix += 1
                cite_id = f"{original_cite_id[:46]}_{suffix}"
            seen_cite_ids.add(cite_id)

            source = payload.get("source", collection)
            metadata = {
                k: v
                for k, v in payload.items()
                if k not in {"text", "source", "chunk_id"}
            }
            # Store the reranker score in metadata for observability.
            metadata["rerank_score"] = rerank_score
            contexts.append(
                ContextChunk(
                    cite_id=cite_id,
                    source=source,
                    text=text,
                    metadata=metadata,
                )
            )

        # 7. Synthesize.
        synth_result = self.synthesizer.synthesize(query, contexts)

        return PipelineResult(
            version="v3",
            query=query,
            answer=synth_result.answer,
            citations=synth_result.citations,
            refused=synth_result.refused,
            retrieved_contexts=contexts,
            tool_calls=[],
        )


def run_v3(
    query: str,
    top_k: int = 5,
    *,
    client: QdrantClient | None = None,
    embedders: Embedders | None = None,
    reranker: Reranker | None = None,
    synthesizer: Synthesizer | None = None,
) -> PipelineResult:
    """Convenience factory: builds defaults from QDRANT_URL env if not provided."""
    if client is None:
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)
    if embedders is None:
        embedders = Embedders()
    if reranker is None:
        reranker = Reranker()
    if synthesizer is None:
        synthesizer = Synthesizer()

    pipeline = RerankPipeline(
        client=client,
        embedders=embedders,
        reranker=reranker,
        synthesizer=synthesizer,
    )
    return pipeline.run(PipelineInput(query=query, top_k=top_k))
