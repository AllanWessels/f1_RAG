"""
Tests for ingestion/wikipedia_etl.py.

All tests use local HTML fixtures — no live HTTP calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from ingestion.wikipedia_etl import (
    CHUNK_TARGET_CHARS,
    Section,
    build_chunks,
    chunk_section,
    load_state,
    make_chunk_id,
    parse_sections,
    run_etl,
    save_state,
    state_key,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_HTML = (FIXTURES_DIR / "wikipedia_sample.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section parsing tests
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_returns_list_of_sections(self) -> None:
        sections = parse_sections(SAMPLE_HTML)
        assert isinstance(sections, list)
        assert len(sections) > 0

    def test_sections_have_content(self) -> None:
        sections = parse_sections(SAMPLE_HTML)
        for s in sections:
            assert isinstance(s, Section)
            assert len(s.paragraphs) > 0
            for p in s.paragraphs:
                assert isinstance(p, str)
                assert len(p) > 0

    def test_skips_references_section(self) -> None:
        sections = parse_sections(SAMPLE_HTML)
        paths = [s.path for s in sections]
        flat = [title for path in paths for title in path]
        assert "References" not in flat

    def test_skips_see_also_section(self) -> None:
        sections = parse_sections(SAMPLE_HTML)
        paths = [s.path for s in sections]
        flat = [title for path in paths for title in path]
        assert "See also" not in flat

    def test_correct_section_breadcrumbs(self) -> None:
        sections = parse_sections(SAMPLE_HTML)
        # Find the Safety car subsection
        safety_car_sections = [s for s in sections if "Safety car" in s.path]
        assert len(safety_car_sections) >= 1, "Expected a Safety car section"
        sc = safety_car_sections[0]
        assert "Race" in sc.path
        assert "Safety car" in sc.path

    def test_background_section_exists(self) -> None:
        sections = parse_sections(SAMPLE_HTML)
        bg_sections = [s for s in sections if "Background" in s.path]
        assert len(bg_sections) >= 1

    def test_intro_section_has_paragraphs(self) -> None:
        """The introductory section (before any heading) should have text."""
        sections = parse_sections(SAMPLE_HTML)
        # There should be at least one section with paragraphs about Abu Dhabi
        all_text = " ".join(p for s in sections for p in s.paragraphs)
        assert "Abu Dhabi" in all_text or "Verstappen" in all_text

    def test_wikitable_content_excluded_from_text(self) -> None:
        """Race result tables (wikitable class) should not pollute text chunks."""
        sections = parse_sections(SAMPLE_HTML)
        all_text = " ".join(p for s in sections for p in s.paragraphs)
        # The table cell contents are bare, but our parser shouldn't pick up
        # the raw table structure — just paragraphs
        assert "Pos" not in all_text or "Driver" not in all_text


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunkSection:
    def _make_section(self, paragraphs: list[str], path: list[str] | None = None) -> Section:
        s = Section(path or ["Race"])
        for p in paragraphs:
            s.add_paragraph(p)
        return s

    def test_single_small_para_is_one_chunk(self) -> None:
        s = self._make_section(["This is a short paragraph about the race."])
        chunks = chunk_section(s)
        assert len(chunks) == 1
        assert "short paragraph" in chunks[0]["text"]

    def test_chunk_respects_target_size(self) -> None:
        """Multiple paragraphs totalling > CHUNK_TARGET_CHARS should produce multiple chunks."""
        para = "A" * 1000  # 1000 chars
        # 5 paragraphs x 1000 chars = 5000 chars > CHUNK_TARGET_CHARS (3200)
        s = self._make_section([para] * 5)
        chunks = chunk_section(s, target_chars=3200)
        assert len(chunks) >= 2

    def test_chunk_text_not_exceeding_double_target(self) -> None:
        """No single chunk should be more than 2x the target chars (rough guard)."""
        para = "B" * 500
        s = self._make_section([para] * 10)
        chunks = chunk_section(s, target_chars=CHUNK_TARGET_CHARS)
        for c in chunks:
            assert len(c["text"]) <= CHUNK_TARGET_CHARS * 2

    def test_chunk_carries_section_path(self) -> None:
        s = self._make_section(["Some text here."], path=["Race", "Safety car"])
        chunks = chunk_section(s)
        assert chunks[0]["section_path"] == ["Race", "Safety car"]

    def test_very_long_paragraph_split(self) -> None:
        """A paragraph longer than target_chars should still produce finite chunks."""
        long_para = ". ".join(["Word"] * 2000)  # very long paragraph
        s = self._make_section([long_para])
        chunks = chunk_section(s, target_chars=800)
        assert len(chunks) >= 1
        # Each chunk text should not be astronomically larger than target
        for c in chunks:
            assert len(c["text"]) < 800 * 10  # loose upper bound

    def test_empty_section_returns_no_chunks(self) -> None:
        s = Section(["Race"])
        chunks = chunk_section(s)
        assert chunks == []


# ---------------------------------------------------------------------------
# build_chunks integration test
# ---------------------------------------------------------------------------


class TestBuildChunks:
    def test_metadata_fields_present(self) -> None:
        chunks = build_chunks(
            year=2021,
            gp_name="Abu Dhabi Grand Prix",
            page_title="2021_Abu_Dhabi_Grand_Prix",
            html=SAMPLE_HTML,
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "year" in chunk
            assert "gp_name" in chunk
            assert "page_title" in chunk
            assert "page_url" in chunk
            assert "section_path" in chunk
            assert "text" in chunk

    def test_metadata_types(self) -> None:
        chunks = build_chunks(
            year=2021,
            gp_name="Abu Dhabi Grand Prix",
            page_title="2021_Abu_Dhabi_Grand_Prix",
            html=SAMPLE_HTML,
        )
        for chunk in chunks:
            assert isinstance(chunk["chunk_id"], str)
            assert isinstance(chunk["year"], int)
            assert chunk["year"] == 2021
            assert isinstance(chunk["gp_name"], str)
            assert chunk["gp_name"] == "Abu Dhabi Grand Prix"
            assert isinstance(chunk["page_title"], str)
            assert isinstance(chunk["page_url"], str)
            assert chunk["page_url"].startswith("https://en.wikipedia.org/wiki/")
            assert isinstance(chunk["section_path"], list)
            assert isinstance(chunk["text"], str)
            assert len(chunk["text"]) > 0

    def test_chunk_ids_unique(self) -> None:
        chunks = build_chunks(
            year=2021,
            gp_name="Abu Dhabi Grand Prix",
            page_title="2021_Abu_Dhabi_Grand_Prix",
            html=SAMPLE_HTML,
        )
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_ids_contain_year(self) -> None:
        chunks = build_chunks(
            year=2021,
            gp_name="Abu Dhabi Grand Prix",
            page_title="2021_Abu_Dhabi_Grand_Prix",
            html=SAMPLE_HTML,
        )
        for chunk in chunks:
            assert "2021" in chunk["chunk_id"]

    def test_controversy_text_present(self) -> None:
        """The fixture has a 'Controversy' section — it should appear in chunks."""
        chunks = build_chunks(
            year=2021,
            gp_name="Abu Dhabi Grand Prix",
            page_title="2021_Abu_Dhabi_Grand_Prix",
            html=SAMPLE_HTML,
        )
        all_text = " ".join(c["text"] for c in chunks)
        assert "safety car" in all_text.lower() or "controversy" in all_text.lower()


# ---------------------------------------------------------------------------
# make_chunk_id tests
# ---------------------------------------------------------------------------


class TestMakeChunkId:
    def test_format(self) -> None:
        cid = make_chunk_id(2021, "Abu Dhabi Grand Prix", ["Race", "Safety car"], 3)
        assert cid == "2021_abu_dhabi_grand_prix__race_safety_car__0003"

    def test_empty_section_path(self) -> None:
        cid = make_chunk_id(2021, "British Grand Prix", [], 0)
        assert "intro" in cid

    def test_different_index_different_id(self) -> None:
        cid1 = make_chunk_id(2021, "Abu Dhabi Grand Prix", ["Race"], 0)
        cid2 = make_chunk_id(2021, "Abu Dhabi Grand Prix", ["Race"], 1)
        assert cid1 != cid2


# ---------------------------------------------------------------------------
# State management tests
# ---------------------------------------------------------------------------


class TestStateManagement:
    def test_load_missing_state(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        state = load_state(p)
        assert state == {}

    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        original = {"2021::Abu Dhabi Grand Prix": True, "2020::Austrian Grand Prix": True}
        save_state(p, original)
        loaded = load_state(p)
        assert loaded == original

    def test_state_key_format(self) -> None:
        key = state_key(2021, "Abu Dhabi Grand Prix")
        assert key == "2021::Abu Dhabi Grand Prix"


# ---------------------------------------------------------------------------
# run_etl resume / --force logic tests (mocked HTTP)
# ---------------------------------------------------------------------------


def _make_mock_response(html: str) -> MagicMock:
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


class TestRunEtlResumeLogic:
    """Test idempotency and --force flag without hitting live Wikipedia."""

    def test_skips_completed_pages(self, tmp_path: Path) -> None:
        """Pages already in state should not be fetched again."""
        output = tmp_path / "out.jsonl"
        state_p = tmp_path / "state.json"

        # Pre-populate state: mark Abu Dhabi 2021 as done
        save_state(state_p, {"2021::Abu Dhabi Grand Prix": True})

        fetch_count = 0

        def fake_get(session: Any, url: str) -> MagicMock:
            nonlocal fetch_count
            fetch_count += 1
            return _make_mock_response(SAMPLE_HTML)

        with patch("ingestion.wikipedia_etl._get", side_effect=fake_get):
            run_etl(
                years=[2021],
                force=False,
                output_path=output,
                state_path=state_p,
                delay=0,
            )

        # Abu Dhabi was pre-done; we should have fetched the other 21 races in 2021
        # but NOT Abu Dhabi
        assert fetch_count == 21  # 22 races - 1 already done

    def test_force_refetches_completed_pages(self, tmp_path: Path) -> None:
        """With --force, all pages in the year should be fetched."""
        output = tmp_path / "out.jsonl"
        state_p = tmp_path / "state.json"

        # Mark all 2021 races as done
        from ingestion.wikipedia_etl import GP_CALENDAR

        state: dict[str, bool] = {}
        for gp_name, _ in GP_CALENDAR[2021]:
            state[f"2021::{gp_name}"] = True
        save_state(state_p, state)

        fetch_count = 0

        def fake_get(session: Any, url: str) -> MagicMock:
            nonlocal fetch_count
            fetch_count += 1
            return _make_mock_response(SAMPLE_HTML)

        with patch("ingestion.wikipedia_etl._get", side_effect=fake_get):
            run_etl(
                years=[2021],
                force=True,
                output_path=output,
                state_path=state_p,
                delay=0,
            )

        # With force, all 22 races should be fetched regardless of state
        assert fetch_count == len(GP_CALENDAR[2021])

    def test_output_jsonl_has_valid_json(self, tmp_path: Path) -> None:
        """Each line in output JSONL must be valid JSON with required fields."""
        output = tmp_path / "out.jsonl"
        state_p = tmp_path / "state.json"

        def fake_get(session: Any, url: str) -> MagicMock:
            return _make_mock_response(SAMPLE_HTML)

        with patch("ingestion.wikipedia_etl._get", side_effect=fake_get):
            run_etl(
                years=[2021],
                force=True,
                output_path=output,
                state_path=state_p,
                delay=0,
            )

        assert output.exists()
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 0

        required_fields = {"chunk_id", "year", "gp_name", "page_title", "page_url", "section_path", "text"}
        for line in lines:
            obj = json.loads(line)
            missing = required_fields - set(obj.keys())
            assert not missing, f"Missing fields: {missing} in chunk {obj.get('chunk_id')}"

    def test_state_written_after_each_page(self, tmp_path: Path) -> None:
        """State should be saved incrementally so progress is not lost on crash."""
        output = tmp_path / "out.jsonl"
        state_p = tmp_path / "state.json"

        pages_fetched: list[str] = []

        def fake_get(session: Any, url: str) -> MagicMock:
            pages_fetched.append(url)
            return _make_mock_response(SAMPLE_HTML)

        with patch("ingestion.wikipedia_etl._get", side_effect=fake_get):
            run_etl(
                years=[2021],
                force=True,
                output_path=output,
                state_path=state_p,
                delay=0,
            )

        # State file should exist and have entries for all 2021 races
        from ingestion.wikipedia_etl import GP_CALENDAR

        state = load_state(state_p)
        for gp_name, _ in GP_CALENDAR[2021]:
            assert state.get(f"2021::{gp_name}") is True, f"Missing state for {gp_name}"

    def test_chunk_counts_returned(self, tmp_path: Path) -> None:
        """run_etl should return a dict of chunk counts per race."""
        output = tmp_path / "out.jsonl"
        state_p = tmp_path / "state.json"

        def fake_get(session: Any, url: str) -> MagicMock:
            return _make_mock_response(SAMPLE_HTML)

        with patch("ingestion.wikipedia_etl._get", side_effect=fake_get):
            counts = run_etl(
                years=[2021],
                force=True,
                output_path=output,
                state_path=state_p,
                delay=0,
            )

        assert isinstance(counts, dict)
        assert len(counts) > 0
        for label, count in counts.items():
            assert isinstance(label, str)
            assert isinstance(count, int)
            assert count >= 0
