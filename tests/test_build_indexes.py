"""Unit tests for ingestion.build_indexes.

The dense/sparse embedding models are never loaded — Embedders is stubbed.
The Qdrant client is fully mocked.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion import build_indexes
from ingestion.build_indexes import (
    Chunk,
    _batched,
    _point_id,
    build,
    collection_is_empty,
    ensure_collection,
    load_jsonl_chunks,
    load_stats_chunks,
    upsert_chunks,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeSparse:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


class _StubDense:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def encode(self, texts, normalize_embeddings: bool = True, show_progress_bar: bool = False):
        return [[0.0] * self.dim for _ in texts]


class _StubSparseModel:
    def embed(self, texts: Iterable[str]):
        for i, _ in enumerate(texts):
            yield _FakeSparse(indices=[i, i + 1], values=[0.1, 0.2])


class _StubEmbedders:
    def __init__(self) -> None:
        self._dense = _StubDense()
        self._sparse = _StubSparseModel()

    def dense(self):
        return self._dense

    def sparse(self):
        return self._sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def stats_db(tmp_path: Path) -> Path:
    """Build a 1-race SQLite that matches the M2 schema."""
    db = tmp_path / "stats.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE races (race_id INTEGER, season INTEGER, round INTEGER,
                            circuit_id TEXT, name TEXT, date TEXT);
        CREATE TABLE drivers (driver_id TEXT, forename TEXT, surname TEXT);
        CREATE TABLE constructors (constructor_id TEXT, name TEXT);
        CREATE TABLE results (race_id INTEGER, driver_id TEXT, constructor_id TEXT,
                              position INTEGER);
        CREATE TABLE qualifying (race_id INTEGER, driver_id TEXT, constructor_id TEXT,
                                 position INTEGER);
        INSERT INTO races VALUES (1, 2021, 21, 'yas_marina', 'Abu Dhabi Grand Prix', '2021-12-12');
        INSERT INTO drivers VALUES ('max', 'Max', 'Verstappen'),
                                   ('ham', 'Lewis', 'Hamilton'),
                                   ('sai', 'Carlos', 'Sainz');
        INSERT INTO constructors VALUES ('red_bull', 'Red Bull'),
                                        ('mercedes', 'Mercedes'),
                                        ('ferrari', 'Ferrari');
        INSERT INTO results VALUES (1, 'max', 'red_bull', 1),
                                   (1, 'ham', 'mercedes', 2),
                                   (1, 'sai', 'ferrari', 3);
        INSERT INTO qualifying VALUES (1, 'max', 'red_bull', 1);
        """
    )
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def jsonl_file(tmp_path: Path) -> Path:
    p = tmp_path / "chunks.jsonl"
    p.write_text(
        json.dumps({"chunk_id": "a__0", "year": 2021, "text": "first chunk"})
        + "\n"
        + json.dumps({"chunk_id": "a__1", "year": 2021, "text": "second chunk"})
        + "\n"
        + json.dumps({"chunk_id": "a__2", "year": 2021, "text": ""})  # skipped (no text)
        + "\n"
    )
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoaders:
    def test_stats_chunks_emit_correct_summary(self, stats_db: Path) -> None:
        chunks = list(load_stats_chunks(stats_db))
        assert len(chunks) == 1
        c = chunks[0]
        assert c.source == "stats"
        assert "Abu Dhabi Grand Prix" in c.text
        assert "P1 Max Verstappen (Red Bull)" in c.text
        assert "P2 Lewis Hamilton (Mercedes)" in c.text
        assert "P3 Carlos Sainz (Ferrari)" in c.text
        assert "Pole: Max Verstappen (Red Bull)" in c.text
        assert c.payload["season"] == 2021
        assert c.payload["round"] == 21
        assert c.chunk_id == "race__2021_21"

    def test_stats_chunks_missing_db_is_silent(self, tmp_path: Path) -> None:
        assert list(load_stats_chunks(tmp_path / "nope.sqlite")) == []

    def test_jsonl_chunks_skips_empty_text(self, jsonl_file: Path) -> None:
        chunks = list(load_jsonl_chunks(jsonl_file, source="narratives"))
        assert len(chunks) == 2
        assert all(c.text for c in chunks)
        assert all(c.source == "narratives" for c in chunks)
        # text removed from payload, chunk_id preserved on payload via upsert step
        assert "text" not in chunks[0].payload


class TestBatching:
    def test_batched_yields_full_batches_then_remainder(self) -> None:
        items = [
            Chunk(chunk_id=str(i), source="x", text=str(i), payload={})
            for i in range(5)
        ]
        batches = list(_batched(items, 2))
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_batched_empty(self) -> None:
        assert list(_batched([], 4)) == []


class TestPointId:
    def test_stable_across_calls(self) -> None:
        assert _point_id("foo") == _point_id("foo")

    def test_distinct_inputs_distinct_ids(self) -> None:
        assert _point_id("a") != _point_id("b")

    def test_fits_positive_signed_64bit(self) -> None:
        pid = _point_id("any-chunk-id")
        assert 0 <= pid < (1 << 63)


class TestQdrantWrappers:
    def test_ensure_collection_creates_when_missing(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = False
        ensure_collection(client, "narratives")
        client.create_collection.assert_called_once()
        kwargs = client.create_collection.call_args.kwargs
        assert kwargs["collection_name"] == "narratives"
        assert "dense" in kwargs["vectors_config"]
        assert "sparse" in kwargs["sparse_vectors_config"]

    def test_ensure_collection_noop_when_exists(self) -> None:
        client = MagicMock()
        client.collection_exists.return_value = True
        ensure_collection(client, "narratives")
        client.create_collection.assert_not_called()

    def test_collection_is_empty(self) -> None:
        client = MagicMock()
        client.count.return_value.count = 0
        assert collection_is_empty(client, "foo") is True
        client.count.return_value.count = 5
        assert collection_is_empty(client, "foo") is False


class TestUpsert:
    def test_upsert_emits_both_vectors_per_point(self) -> None:
        client = MagicMock()
        chunks = [
            Chunk(chunk_id="x__1", source="narratives", text="hello", payload={"y": 2021}),
            Chunk(chunk_id="x__2", source="narratives", text="world", payload={"y": 2022}),
        ]
        n = upsert_chunks(client, "narratives", chunks, _StubEmbedders(), batch_size=10)
        assert n == 2
        client.upsert.assert_called_once()
        call = client.upsert.call_args
        assert call.kwargs["collection_name"] == "narratives"
        points = call.kwargs["points"]
        assert len(points) == 2
        for p in points:
            assert set(p.vector.keys()) == {"dense", "sparse"}
            assert len(p.vector["dense"]) == 1024
            assert p.payload["source"] == "narratives"
            assert "chunk_id" in p.payload


class TestBuildOrchestration:
    def test_build_skips_non_empty_unless_force(self, monkeypatch) -> None:
        client = MagicMock()
        client.collection_exists.return_value = True
        client.count.return_value.count = 100  # non-empty
        monkeypatch.setattr(build_indexes, "load_stats_chunks", lambda *a, **kw: iter([]))
        counts = build(client, _StubEmbedders(), collections=("stats_meta",), force=False)
        assert counts == {"stats_meta": 0}
        client.upsert.assert_not_called()

    def test_build_rebuilds_when_force(self, monkeypatch) -> None:
        client = MagicMock()
        client.collection_exists.return_value = True
        client.count.return_value.count = 100
        sample_chunk = Chunk(chunk_id="x__0", source="stats", text="hi", payload={})
        monkeypatch.setattr(
            build_indexes, "load_stats_chunks", lambda *a, **kw: iter([sample_chunk])
        )
        counts = build(client, _StubEmbedders(), collections=("stats_meta",), force=True)
        assert counts == {"stats_meta": 1}
        client.delete_collection.assert_called_once_with("stats_meta")
        client.upsert.assert_called_once()

    def test_build_runs_only_requested_collections(self, monkeypatch) -> None:
        client = MagicMock()
        client.collection_exists.return_value = True
        client.count.return_value.count = 0  # empty so it ingests
        monkeypatch.setattr(
            build_indexes,
            "load_jsonl_chunks",
            lambda *a, **kw: iter(
                [Chunk(chunk_id="n__0", source="narratives", text="x", payload={})]
            ),
        )
        counts = build(
            client, _StubEmbedders(), collections=("narratives",), force=False
        )
        assert list(counts.keys()) == ["narratives"]
        assert counts["narratives"] == 1
