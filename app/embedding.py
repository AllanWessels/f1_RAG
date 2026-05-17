"""
Shared lazy-loaded models for embedding + reranking.

- Embedders: dense (BAAI/bge-large-en-v1.5) + sparse (Qdrant/bm25)
- Reranker:  BAAI/bge-reranker-v2-m3

Models load on first access so unit tests can stub them without touching weights.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DENSE_MODEL = "BAAI/bge-large-en-v1.5"
DENSE_DIM = 1024
SPARSE_MODEL = "Qdrant/bm25"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _pick_device() -> str:
    """Honor F1RAG_DEVICE, else use CUDA if available, else CPU."""
    forced = os.environ.get("F1RAG_DEVICE")
    if forced:
        return forced
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Embedders:
    """Holds the dense + sparse embedding models; both load lazily."""

    def __init__(self) -> None:
        self._dense: Any | None = None
        self._sparse: Any | None = None

    def dense(self) -> Any:
        if self._dense is None:
            from sentence_transformers import SentenceTransformer

            device = _pick_device()
            logger.info("Loading dense model %s on device=%s", DENSE_MODEL, device)
            self._dense = SentenceTransformer(DENSE_MODEL, device=device)
        return self._dense

    def sparse(self) -> Any:
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            logger.info("Loading sparse model %s", SPARSE_MODEL)
            self._sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)
        return self._sparse

    def encode_dense(self, text: str) -> list[float]:
        vec = self.dense().encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return list(vec)

    def encode_sparse(self, text: str) -> tuple[list[int], list[float]]:
        emb = next(iter(self.sparse().embed([text])))
        return list(emb.indices), list(emb.values)


class Reranker:
    """Lazy bge-reranker-v2-m3 via sentence-transformers' CrossEncoder."""

    def __init__(self) -> None:
        self._model: Any | None = None

    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = _pick_device()
            logger.info("Loading reranker %s on device=%s", RERANKER_MODEL, device)
            self._model = CrossEncoder(RERANKER_MODEL, device=device)
        return self._model

    def rerank(
        self, query: str, candidates: list[str], top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Return [(original_index, score)] sorted by score desc, truncated to top_k."""
        if not candidates:
            return []
        pairs = [(query, c) for c in candidates]
        scores = self.model().predict(pairs, show_progress_bar=False)
        ranked = sorted(enumerate(scores), key=lambda p: float(p[1]), reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k]]
