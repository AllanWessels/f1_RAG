"""Tests for ingestion/regulations_etl.py — M4.

All tests are self-contained: no live network calls, no fia.com downloads.
The fixture PDF at tests/fixtures/sample_regulation.pdf is a tiny synthetic
document using the FIA 2026 lettered article format (e.g. T1.1, T4.2.1).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from ingestion.regulations_etl import (
    ARTICLE_PATTERN,
    MAX_CHUNK_CHARS,
    SEASON,
    _char_to_page,
    _load_existing_source_pdfs,
    _make_chunk_id,
    chunk_article_text,
    extract_chunks,
    extract_text_with_pages,
    split_into_articles,
    url_to_filename,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample_regulation.pdf"


# ===========================================================================
# 1. Article-splitting regex
# ===========================================================================


class TestArticlePattern:
    """Verify the ARTICLE_PATTERN regex against synthetic multi-article text.

    The 2026 FIA regulations use a lettered prefix scheme (e.g. C4.1, B1.2).
    Classic dotted numbers (4.2.1) are also supported.  Bare numbers like
    "07", "2026", and "0" must NOT match.
    """

    # Synthetic text using FIA 2026 lettered format
    SAMPLE_TEXT = """\
SECTION T: TEST REGULATIONS

ARTICLE T1: GENERAL PROVISIONS

T1.1 Purpose
The purpose of these regulations is to ensure fair competition.

T1.2 Scope
These regulations apply to all Formula 1 Championship events.

T2.1 Car
A car is a vehicle.

T4.2.1 Ballast Material
Ballast must be homogeneous.

07 May 2026
0

C
28 April 2026
"""

    # Synthetic text using classic dotted format (no letter prefix)
    CLASSIC_TEXT = """\
1.1 Purpose
The purpose of these regulations is to ensure fair competition.

1.2 Scope
These regulations apply to all Formula 1 Championship events.

2.1 Car
A car is a vehicle.

4.2.1 Ballast Material
Ballast must be homogeneous.

07 May 2026
28 April 2026
"""

    def test_lettered_articles_detected(self) -> None:
        matches = list(ARTICLE_PATTERN.finditer(self.SAMPLE_TEXT))
        numbers = [m.group(1) for m in matches]
        assert "T1.1" in numbers
        assert "T1.2" in numbers
        assert "T2.1" in numbers
        assert "T4.2.1" in numbers

    def test_classic_dotted_articles_detected(self) -> None:
        matches = list(ARTICLE_PATTERN.finditer(self.CLASSIC_TEXT))
        numbers = [m.group(1) for m in matches]
        assert "1.1" in numbers
        assert "1.2" in numbers
        assert "2.1" in numbers
        assert "4.2.1" in numbers

    def test_date_numbers_not_matched(self) -> None:
        """Bare numbers like '07', '28', '2026' must not be caught as articles."""
        matches = list(ARTICLE_PATTERN.finditer(self.SAMPLE_TEXT))
        numbers = [m.group(1) for m in matches]
        assert "07" not in numbers, "Day number '07' incorrectly matched"
        assert "28" not in numbers, "Day number '28' incorrectly matched"
        assert "2026" not in numbers, "Year '2026' incorrectly matched"
        assert "0" not in numbers, "Page artefact '0' incorrectly matched"

    def test_chapter_heading_not_matched(self) -> None:
        """Lines starting with 'SECTION T:' should not be caught as articles."""
        matches = list(ARTICLE_PATTERN.finditer(self.SAMPLE_TEXT))
        titles = [m.group(0) for m in matches]
        for title in titles:
            assert not title.upper().startswith("SECTION"), (
                f"SECTION heading incorrectly matched: {title!r}"
            )

    def test_split_into_articles_returns_tuples(self) -> None:
        articles = split_into_articles(self.SAMPLE_TEXT)
        assert len(articles) >= 4  # T1.1, T1.2, T2.1, T4.2.1
        for art_num, art_title, start in articles:
            assert isinstance(art_num, str)
            # Allow optional letter prefix
            assert re.match(r"^[A-Z]?\d+(\.\d+)*$", art_num), f"Bad art_num: {art_num!r}"
            assert isinstance(art_title, str) and art_title
            assert isinstance(start, int) and start >= 0

    def test_article_order_is_ascending(self) -> None:
        """Article start positions must be strictly increasing."""
        articles = split_into_articles(self.SAMPLE_TEXT)
        starts = [start for _, _, start in articles]
        assert starts == sorted(starts)

    def test_empty_text_returns_empty(self) -> None:
        assert split_into_articles("") == []

    def test_no_articles_returns_empty(self) -> None:
        assert split_into_articles("Just some plain text\nWith no article headers.") == []


# ===========================================================================
# 2. Chunk-size cap
# ===========================================================================


class TestChunkSizeCap:
    """Verify that oversized articles are split correctly."""

    REG_TYPE = "technical"
    SOURCE_PDF = "test_regs.pdf"

    def _long_article_text(self, chars: int) -> str:
        """Build a plain-text article body (no sub-articles) of ~*chars* length."""
        # No sub-article patterns so we hit the hard-split path
        base = "X" * 100 + "\n"
        return (base * (chars // len(base) + 1))[:chars]

    def test_short_article_single_chunk(self) -> None:
        text = self._long_article_text(100)
        chunks = chunk_article_text(
            text, "C4.1", "Short Article", self.REG_TYPE,
            self.SOURCE_PDF, 1, ["C4.1 — Short Article"],
        )
        assert len(chunks) == 1

    def test_long_article_splits_into_parts(self) -> None:
        text = self._long_article_text(MAX_CHUNK_CHARS * 3)
        chunks = chunk_article_text(
            text, "C4.2", "Long Article", self.REG_TYPE,
            self.SOURCE_PDF, 1, ["C4.2 — Long Article"],
        )
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["text"]) <= MAX_CHUNK_CHARS, (
                f"Chunk too long: {len(chunk['text'])} chars"
            )

    def test_chunk_ids_are_unique_for_parts(self) -> None:
        text = self._long_article_text(MAX_CHUNK_CHARS * 2 + 500)
        chunks = chunk_article_text(
            text, "C5.1", "Long Engine Article", self.REG_TYPE,
            self.SOURCE_PDF, 2, ["C5.1 — Long Engine Article"],
        )
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk_ids: {ids}"

    def test_sub_article_split_preferred_over_hard_split(self) -> None:
        """An article with sub-articles should split on those, not character count."""
        filler = " regulation text " * 50  # ~900 chars per section
        # Use dotted format (no letter) for sub-articles so the sub-pattern matches
        sub_text = f"4.1 Sub One\n{filler}\n4.2 Sub Two\n{filler}"
        chunks = chunk_article_text(
            sub_text * 10,  # repeat to exceed MAX_CHUNK_CHARS
            "4.0", "Parent Article", self.REG_TYPE,
            self.SOURCE_PDF, 3, ["4.0 — Parent Article"],
        )
        # Should have produced at least 2 chunks
        assert len(chunks) >= 2


# ===========================================================================
# 3. Metadata shape
# ===========================================================================


class TestChunkMetadataShape:
    """Verify that every emitted chunk has the required fields with correct types."""

    REQUIRED_FIELDS: ClassVar[dict[str, type]] = {
        "chunk_id": str,
        "regulation_type": str,
        "season": int,
        "article_number": str,
        "article_path": list,
        "source_pdf": str,
        "page_number": int,
        "text": str,
    }

    def _make_chunk(self, **overrides: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "text": "Some regulation text that is long enough.",
            "article_number": "B3.1",
            "article_title": "Safety",
            "reg_type": "sporting",
            "source_pdf": "test.pdf",
            "page_number": 5,
            "article_path": ["ARTICLE B3: SAFETY", "B3.1 — Safety"],
        }
        defaults.update(overrides)
        return chunk_article_text(
            defaults["text"],
            defaults["article_number"],
            defaults["article_title"],
            defaults["reg_type"],
            defaults["source_pdf"],
            defaults["page_number"],
            defaults["article_path"],
        )[0]

    def test_all_required_fields_present(self) -> None:
        chunk = self._make_chunk()
        for field in self.REQUIRED_FIELDS:
            assert field in chunk, f"Missing field: {field}"

    def test_field_types(self) -> None:
        chunk = self._make_chunk()
        for field, expected_type in self.REQUIRED_FIELDS.items():
            assert isinstance(chunk[field], expected_type), (
                f"Field '{field}' has wrong type: "
                f"expected {expected_type.__name__}, got {type(chunk[field]).__name__}"
            )

    def test_season_is_2026(self) -> None:
        chunk = self._make_chunk()
        assert chunk["season"] == SEASON == 2026

    def test_regulation_type_propagated(self) -> None:
        for reg_type in ("sporting", "technical", "financial"):
            chunk = self._make_chunk(reg_type=reg_type)
            assert chunk["regulation_type"] == reg_type

    def test_article_path_is_list_of_strings(self) -> None:
        chunk = self._make_chunk(article_path=["ARTICLE B3: SAFETY", "B3.1 — Safety"])
        assert all(isinstance(p, str) for p in chunk["article_path"])

    def test_chunk_id_contains_article_number(self) -> None:
        chunk = self._make_chunk(article_number="C4.2.1")
        assert "C4_2_1" in chunk["chunk_id"]

    def test_chunk_id_contains_reg_type(self) -> None:
        chunk = self._make_chunk(reg_type="sporting")
        assert "sporting" in chunk["chunk_id"]

    def test_text_non_empty(self) -> None:
        chunk = self._make_chunk(text="Some regulation text.")
        assert chunk["text"].strip()

    def test_chunks_from_fixture_pdf(self) -> None:
        """Extract chunks from the tiny fixture PDF and validate metadata."""
        assert SAMPLE_PDF.exists(), f"Fixture not found: {SAMPLE_PDF}"
        chunks = extract_chunks(SAMPLE_PDF, "technical")
        assert len(chunks) > 0, "No chunks extracted from fixture PDF"
        for chunk in chunks:
            for field, expected_type in self.REQUIRED_FIELDS.items():
                assert field in chunk, f"Missing field '{field}' in chunk"
                assert isinstance(chunk[field], expected_type), (
                    f"Field '{field}' wrong type in chunk"
                )
            assert chunk["regulation_type"] == "technical"
            assert chunk["season"] == 2026
            assert chunk["source_pdf"] == SAMPLE_PDF.name


# ===========================================================================
# 4. Text extraction from fixture PDF
# ===========================================================================


class TestTextExtraction:
    """Verify pypdf-based text extraction from the fixture PDF."""

    def test_extract_text_returns_pages(self) -> None:
        assert SAMPLE_PDF.exists()
        pages = extract_text_with_pages(SAMPLE_PDF)
        assert len(pages) == 3

    def test_page_numbers_are_1_indexed(self) -> None:
        pages = extract_text_with_pages(SAMPLE_PDF)
        numbers = [pn for pn, _ in pages]
        assert numbers == [1, 2, 3]

    def test_known_content_present(self) -> None:
        pages = extract_text_with_pages(SAMPLE_PDF)
        all_text = "\n".join(text for _, text in pages)
        # Known content from fixture (FIA 2026 lettered format)
        assert "T1.1" in all_text or "Purpose" in all_text
        assert "800 kg" in all_text
        assert "ENGINE" in all_text or "T5" in all_text

    def test_extract_from_bytes(self) -> None:
        pdf_bytes = SAMPLE_PDF.read_bytes()
        pages = extract_text_with_pages(pdf_bytes)
        assert len(pages) == 3

    def test_char_to_page_mapping(self) -> None:
        """_char_to_page returns the correct 1-based page number."""
        page_map = [(0, 1), (100, 2), (200, 3)]
        assert _char_to_page(0, page_map) == 1
        assert _char_to_page(99, page_map) == 1
        assert _char_to_page(100, page_map) == 2
        assert _char_to_page(150, page_map) == 2
        assert _char_to_page(200, page_map) == 3
        assert _char_to_page(999, page_map) == 3


# ===========================================================================
# 5. Idempotency
# ===========================================================================


class TestIdempotency:
    """Verify that re-runs skip already-processed PDFs."""

    def test_load_existing_source_pdfs_empty_file(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "regs.jsonl"
        jsonl.write_text("")
        assert _load_existing_source_pdfs(jsonl) == set()

    def test_load_existing_source_pdfs_missing_file(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "regs.jsonl"
        assert _load_existing_source_pdfs(jsonl) == set()

    def test_load_existing_source_pdfs_populated(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "regs.jsonl"
        records = [
            {"source_pdf": "sporting.pdf", "chunk_id": "a"},
            {"source_pdf": "technical.pdf", "chunk_id": "b"},
            {"source_pdf": "sporting.pdf", "chunk_id": "c"},
        ]
        with jsonl.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        seen = _load_existing_source_pdfs(jsonl)
        assert seen == {"sporting.pdf", "technical.pdf"}

    def test_load_ignores_malformed_lines(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "regs.jsonl"
        jsonl.write_text('{"source_pdf": "ok.pdf"}\nNOT_JSON\n{"source_pdf": "ok2.pdf"}\n')
        seen = _load_existing_source_pdfs(jsonl)
        assert seen == {"ok.pdf", "ok2.pdf"}


# ===========================================================================
# 6. Utility helpers
# ===========================================================================


class TestUtilities:
    """Verify small utility functions."""

    def test_url_to_filename_basic(self) -> None:
        url = (
            "https://www.fia.com/system/files/documents/"
            "fia_2026_f1_regulations_-_section_b_sporting_-_iss_06_-_2026-04-28.pdf"
        )
        assert url_to_filename(url) == (
            "fia_2026_f1_regulations_-_section_b_sporting_-_iss_06_-_2026-04-28.pdf"
        )

    def test_url_to_filename_no_path(self) -> None:
        assert url_to_filename("https://example.com/file.pdf") == "file.pdf"

    def test_make_chunk_id_no_part(self) -> None:
        cid = _make_chunk_id("sporting", "C4.2.1")
        assert cid == "sporting_2026__article_C4_2_1"

    def test_make_chunk_id_with_part(self) -> None:
        cid = _make_chunk_id("technical", "C5.1", part=2)
        assert cid == "technical_2026__article_C5_1__part2"

    def test_make_chunk_id_part_zero(self) -> None:
        cid = _make_chunk_id("financial", "D1", part=0)
        assert cid == "financial_2026__article_D1__part0"

    def test_make_chunk_id_classic_format(self) -> None:
        cid = _make_chunk_id("sporting", "4.2.1")
        assert cid == "sporting_2026__article_4_2_1"
