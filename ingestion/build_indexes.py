"""
Build the three Qdrant collections (stats_meta, narratives, regulations)
with hybrid dense + sparse vectors.

- dense: BAAI/bge-large-en-v1.5 (1024-dim, cosine)
- sparse: Qdrant/bm25 (BM25 sparse via FastEmbed; IDF modifier)

Idempotent: skips a collection if it already has points unless --force.

Usage:
    python -m ingestion.build_indexes
    python -m ingestion.build_indexes --force
    python -m ingestion.build_indexes --collection narratives
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

from app.embedding import DENSE_DIM, Embedders

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
STATS_DB = DATA_DIR / "f1_stats.sqlite"
WIKIPEDIA_JSONL = DATA_DIR / "wikipedia_races.jsonl"
REGULATIONS_JSONL = DATA_DIR / "fia_regulations.jsonl"

COLLECTIONS = ("stats_meta", "narratives", "regulations")
DEFAULT_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    source: str  # "stats" | "narratives" | "regulations"
    text: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def load_stats_chunks(db_path: Path = STATS_DB) -> Iterator[Chunk]:
    """One chunk per race, summarizing the result with podium + key facts."""
    if not db_path.exists():
        logger.warning("No stats DB at %s; skipping stats_meta", db_path)
        return
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    races = conn.execute(
        "SELECT race_id, season, round, name, date, circuit_id FROM races ORDER BY season, round"
    ).fetchall()
    for race in races:
        # Top 3 finishers
        top3 = conn.execute(
            """
            SELECT r.position, d.forename || ' ' || d.surname AS driver, c.name AS team
            FROM results r
            JOIN drivers d ON d.driver_id = r.driver_id
            JOIN constructors c ON c.constructor_id = r.constructor_id
            WHERE r.race_id = ? AND r.position BETWEEN 1 AND 3
            ORDER BY r.position
            """,
            (race["race_id"],),
        ).fetchall()
        # Pole sitter (qualifying position 1)
        pole_row = conn.execute(
            """
            SELECT d.forename || ' ' || d.surname AS driver, c.name AS team
            FROM qualifying q
            JOIN drivers d ON d.driver_id = q.driver_id
            JOIN constructors c ON c.constructor_id = q.constructor_id
            WHERE q.race_id = ? AND q.position = 1
            LIMIT 1
            """,
            (race["race_id"],),
        ).fetchone()
        if not top3:
            continue
        podium_text = "; ".join(
            f"P{r['position']} {r['driver']} ({r['team']})" for r in top3
        )
        pole_text = (
            f"Pole: {pole_row['driver']} ({pole_row['team']}). " if pole_row else ""
        )
        text = (
            f"{race['season']} {race['name']} (Round {race['round']}, {race['date']}, "
            f"Circuit: {race['circuit_id']}). {pole_text}Podium: {podium_text}."
        )
        yield Chunk(
            chunk_id=f"race__{race['season']}_{race['round']:02d}",
            source="stats",
            text=text,
            payload={
                "race_id": race["race_id"],
                "season": race["season"],
                "round": race["round"],
                "name": race["name"],
                "date": race["date"],
                "circuit_id": race["circuit_id"],
            },
        )
    conn.close()


def load_jsonl_chunks(
    path: Path, source: str, id_field: str = "chunk_id", text_field: str = "text"
) -> Iterator[Chunk]:
    if not path.exists():
        logger.warning("No JSONL at %s; skipping %s", path, source)
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field, "")
            if not text:
                continue
            chunk_id = obj.get(id_field)
            if not chunk_id:
                continue
            yield Chunk(
                chunk_id=chunk_id,
                source=source,
                text=text,
                payload={k: v for k, v in obj.items() if k != text_field},
            )


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


def ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    logger.info("Created collection %s", name)


def collection_is_empty(client: QdrantClient, name: str) -> bool:
    return client.count(collection_name=name, exact=True).count == 0


def _batched(it: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    batch: list[Chunk] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _point_id(chunk_id: str) -> int:
    """Qdrant point IDs must be int or UUID. Hash chunk_id to a stable 63-bit int."""
    import hashlib

    digest = hashlib.sha1(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: Iterable[Chunk],
    embedders: Embedders,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    total = 0
    for batch in _batched(chunks, batch_size):
        texts = [c.text for c in batch]
        dense_vecs = embedders.dense().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        sparse_embeddings = list(embedders.sparse().embed(texts))
        points = []
        for chunk, dense, sparse in zip(batch, dense_vecs, sparse_embeddings, strict=True):
            payload = {
                **chunk.payload,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            }
            points.append(
                PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector={
                        "dense": list(dense),
                        "sparse": SparseVector(
                            indices=list(sparse.indices),
                            values=list(sparse.values),
                        ),
                    },
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection, points=points, wait=True)
        total += len(points)
    return total


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build(
    client: QdrantClient,
    embedders: Embedders,
    collections: tuple[str, ...] = COLLECTIONS,
    force: bool = False,
) -> dict[str, int]:
    """Build/refresh the requested collections. Returns counts upserted per collection."""
    counts: dict[str, int] = {}
    for collection in collections:
        ensure_collection(client, collection)
        if not force and not collection_is_empty(client, collection):
            logger.info("Skipping %s (non-empty; use --force to rebuild)", collection)
            counts[collection] = 0
            continue
        if force:
            client.delete_collection(collection)
            ensure_collection(client, collection)
        match collection:
            case "stats_meta":
                source = load_stats_chunks()
            case "narratives":
                source = load_jsonl_chunks(WIKIPEDIA_JSONL, source="narratives")
            case "regulations":
                source = load_jsonl_chunks(REGULATIONS_JSONL, source="regulations")
            case _:
                raise ValueError(f"Unknown collection: {collection}")
        logger.info("Indexing %s …", collection)
        n = upsert_chunks(client, collection, tqdm(source, desc=collection), embedders)
        counts[collection] = n
        logger.info("Indexed %d points into %s", n, collection)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant indexes for F1 RAG.")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild")
    parser.add_argument(
        "--collection",
        choices=COLLECTIONS,
        action="append",
        help="Restrict to specific collection(s); repeat to pick multiple.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = QdrantClient(url=args.qdrant_url)
    embedders = Embedders()
    collections = tuple(args.collection) if args.collection else COLLECTIONS
    counts = build(client, embedders, collections=collections, force=args.force)
    for name, n in counts.items():
        print(f"{name}: {n} points")


if __name__ == "__main__":
    main()
