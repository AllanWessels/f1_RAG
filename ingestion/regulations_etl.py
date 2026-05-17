"""FIA Formula 1 Regulations ETL — M4

Downloads the 2026 FIA F1 regulation PDFs (Sporting / Technical / Financial),
extracts text with pypdf, and chunks by article into a JSONL file at
``data/fia_regulations.jsonl``.

Entry points
------------
    python -m ingestion.regulations_etl                 # all three types
    python -m ingestion.regulations_etl --force         # re-download & re-extract
    python -m ingestion.regulations_etl --type sporting # one type only
    python -m ingestion.regulations_etl --urls URL1 URL2 URL3  # explicit override

The 2026 FIA regulations are published in lettered sections rather than the
traditional three-document layout.  We map them as:

  sporting  -> Section B (Sporting)
  technical -> Section C (Technical)
  financial -> Section D (Financial - F1 Teams)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    from PyPDF2 import PdfReader  # type: ignore[no-reuse-import]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "fia_pdfs"
JSONL_PATH = DATA_DIR / "fia_regulations.jsonl"

PDF_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regulations_etl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEASON = 2026
INDEX_URL = "https://www.fia.com/regulation/category/110"

# Canonical keywords used to identify each regulation type in PDF filenames/URLs.
# For financial we require BOTH "section_d" AND "f1_teams" to distinguish
# the F1 Teams financial regs from the PU Manufacturers financial regs (section_e).
# Each entry is a list of keywords that must ALL appear in the URL (AND logic).
TYPE_KEYWORDS: dict[str, list[str]] = {
    "sporting": ["section_b"],
    "technical": ["section_c"],
    "financial": ["section_d", "f1_teams"],
}

# Fallback hard-coded URLs (discovered 2026-05-17). The index scraper is tried
# first; these are used if discovery fails or the URL is not found.
FALLBACK_URLS: dict[str, str] = {
    "sporting": (
        "https://www.fia.com/system/files/documents/"
        "fia_2026_f1_regulations_-_section_b_sporting_-_iss_06_-_2026-04-28.pdf"
    ),
    "technical": (
        "https://www.fia.com/system/files/documents/"
        "fia_2026_f1_regulations_-_section_c_technical_-_iss_18_-_2026-05-07.pdf"
    ),
    "financial": (
        "https://www.fia.com/system/files/documents/"
        "fia_2026_f1_regulations_-_section_d_financial_-_f1_teams_-_iss_06_-_2026-04-28.pdf"
    ),
}

# Chunk size limit in approximate tokens (1 token ≈ 4 chars)
MAX_CHUNK_TOKENS = 1500
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4  # 6 000 chars

# Article header pattern handles two numbering schemes:
#
#   FIA 2026 lettered: "C4.2.1 Title Word"  (uppercase letter + digits + dots)
#   Classic dotted:    "4.2.1 Title Word"   (digits + at least one dot + digits)
#
# The alternation forces one of:
#   a) [A-Z]\d+  — a letter prefix guarantees it is a regulation article code
#   b) \d+\.\d+  — a dot separating two numeric parts (never a bare date number)
#
# This prevents false positives like "07 May 2026", "2026 Formula", "0 C", etc.
# Title must start with an uppercase letter followed by at least one more alpha
# character (avoids single-letter page markers).
ARTICLE_PATTERN = re.compile(
    r"(?m)^([A-Z]\d+(?:\.\d+)*|\d+\.\d+(?:\.\d+)*)\s{1,8}([A-Z][A-Za-z][^\n]*)",
)

# Chapter / Part heading pattern (e.g. "ARTICLE C4: MASS", "SECTION B: SPORTING")
CHAPTER_PATTERN = re.compile(
    r"(?m)^(?:CHAPTER|PART|SECTION|ANNEX|ARTICLE\s+[A-Z]?\d+)[^\n]*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (compatible; f1-rag-bot/1.0; "
            "+https://github.com/f1-rag)"
        )
    }
)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def _get(url: str, **kwargs: Any) -> requests.Response:
    resp = SESSION.get(url, timeout=60, **kwargs)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# URL discovery
# ---------------------------------------------------------------------------

def _url_matches_type(url: str, reg_type: str) -> bool:
    """Return True if the URL plausibly belongs to *reg_type*.

    All keywords in TYPE_KEYWORDS[reg_type] must appear in the URL (AND logic)
    so that e.g. the financial F1 Teams regs (section_d) are not confused with
    the PU Manufacturers regs (section_e) which also contain 'financial'.
    """
    url_lower = url.lower()
    keywords = TYPE_KEYWORDS[reg_type]
    return all(kw in url_lower for kw in keywords)


_ISS_PATTERN = re.compile(r"iss[_\s-]*(\d+)", re.IGNORECASE)
_DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _url_recency_key(url: str) -> tuple[int, int, int, int]:
    """Return a sortable tuple (year, month, day, issue) extracted from a PDF URL.

    Higher is newer.  Filenames that lack a date or issue number get (0, 0, 0, 0).
    """
    iss_match = _ISS_PATTERN.search(url)
    issue = int(iss_match.group(1)) if iss_match else 0
    date_match = _DATE_PATTERN.search(url)
    if date_match:
        year, month, day = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
    else:
        year, month, day = 0, 0, 0
    return (year, month, day, issue)


def discover_urls(reg_types: list[str]) -> dict[str, str]:
    """Fetch the FIA index page and extract the most-recent PDF URL per type.

    Falls back to FALLBACK_URLS for any type not found on the index page.
    """
    found: dict[str, str] = {}
    try:
        log.info("Fetching FIA index page: %s", INDEX_URL)
        resp = _get(INDEX_URL)
        soup = BeautifulSoup(resp.text, "lxml")
        anchors = soup.find_all("a", href=True)
        pdf_urls_set: set[str] = set()
        for a in anchors:
            href: str = a["href"]
            if not href.lower().endswith(".pdf"):
                continue
            if str(SEASON) not in href:
                continue
            # Make absolute
            if href.startswith("http"):
                pdf_urls_set.add(href)
            else:
                pdf_urls_set.add("https://www.fia.com" + href)

        pdf_urls = list(pdf_urls_set)
        log.info("Found %d unique 2026 PDF links on index page", len(pdf_urls))

        for reg_type in reg_types:
            candidates = [u for u in pdf_urls if _url_matches_type(u, reg_type)]
            if candidates:
                # Pick the most-recent URL by (date, issue) extracted from filename
                best = max(candidates, key=_url_recency_key)
                found[reg_type] = best
                log.info("Discovered %s URL: %s", reg_type, best)
            else:
                log.warning(
                    "No URL found for '%s' on index page; using fallback.", reg_type
                )
    except Exception as exc:
        log.warning("Index page fetch failed (%s); using fallback URLs.", exc)

    # Fill gaps with fallbacks
    for reg_type in reg_types:
        if reg_type not in found:
            found[reg_type] = FALLBACK_URLS[reg_type]
            log.info("Fallback URL for %s: %s", reg_type, found[reg_type])

    return found


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_pdf(url: str, pdf_path: Path, force: bool = False) -> Path:
    """Download *url* to *pdf_path* unless it already exists (and not forced)."""
    if pdf_path.exists() and not force:
        log.info("PDF already exists, skipping download: %s", pdf_path.name)
        return pdf_path

    log.info("Downloading %s → %s", url, pdf_path.name)
    resp = _get(url, stream=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with pdf_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            fh.write(chunk)
    log.info("Saved %s (%.1f KB)", pdf_path.name, pdf_path.stat().st_size / 1024)
    return pdf_path


def url_to_filename(url: str) -> str:
    """Derive a local filename from the PDF URL."""
    return url.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Normalise unicode, collapse runs of whitespace, strip ligatures."""
    text = unicodedata.normalize("NFKC", text)
    # Replace form feeds and carriage returns with newlines
    text = text.replace("\f", "\n").replace("\r", "\n")
    # Collapse multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_text_with_pages(pdf_source: Path | bytes) -> list[tuple[int, str]]:
    """Return a list of (1-based page_number, page_text) tuples."""
    if isinstance(pdf_source, bytes):
        reader = PdfReader(io.BytesIO(pdf_source))
    else:
        reader = PdfReader(str(pdf_source))

    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _normalise(text)
        pages.append((i, text))
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _build_full_text_with_page_map(
    pages: list[tuple[int, str]],
) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate pages and return (full_text, page_map).

    page_map[i] = (start_char, page_number) so we can look up which page a
    character offset belongs to.
    """
    parts: list[str] = []
    page_map: list[tuple[int, int]] = []  # (start_char_of_page, page_number)
    offset = 0
    for page_num, text in pages:
        page_map.append((offset, page_num))
        parts.append(text)
        offset += len(text) + 1  # +1 for the \n separator
    return "\n".join(p for _, p in pages), page_map


def _char_to_page(char_offset: int, page_map: list[tuple[int, int]]) -> int:
    """Return the page number containing *char_offset*."""
    page_num = 1
    for start, pn in page_map:
        if start <= char_offset:
            page_num = pn
        else:
            break
    return page_num


def _extract_chapter_breadcrumbs(text: str, article_start: int) -> list[str]:
    """Find the last chapter/part heading before *article_start* in *text*."""
    text_before = text[:article_start]
    matches = list(CHAPTER_PATTERN.finditer(text_before))
    if matches:
        return [matches[-1].group(0).strip()]
    return []


def _slug(article_number: str) -> str:
    return article_number.replace(".", "_")


def _make_chunk_id(reg_type: str, article_number: str, part: int | None = None) -> str:
    base = f"{reg_type}_{SEASON}__article_{_slug(article_number)}"
    if part is not None:
        base += f"__part{part}"
    return base


def split_into_articles(full_text: str) -> list[tuple[str, str, int]]:
    """Split *full_text* into (article_number, title, start_char) tuples.

    Each entry represents the start of an article header match.
    """
    matches = list(ARTICLE_PATTERN.finditer(full_text))
    articles: list[tuple[str, str, int]] = []
    for m in matches:
        article_number = m.group(1)
        title = m.group(2).strip()
        start = m.start()
        articles.append((article_number, title, start))
    return articles


# Sub-article pattern used when splitting oversized articles.
# Only matches classic dotted format (e.g. 4.2.1) — NOT the lettered FIA format
# (B4.2.1) to avoid matching the article header itself during recursion.
# Requires at least one dot so bare numbers like "10" don't match.
_SUB_ARTICLE_PAT = re.compile(
    r"(?m)^(\d+\.\d+(?:\.\d+)*)\s{1,8}([A-Z][A-Za-z][^\n]*)",
)


def _hard_split(
    text: str,
    article_number: str,
    reg_type: str,
    source_pdf: str,
    page_number: int,
    article_path: list[str],
) -> list[dict[str, Any]]:
    """Split *text* into MAX_CHUNK_CHARS segments, each as a separate chunk."""
    parts: list[dict[str, Any]] = []
    start = 0
    part_idx = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        segment = text[start:end].strip()
        if segment:
            parts.append(
                {
                    "chunk_id": _make_chunk_id(reg_type, article_number, part_idx),
                    "regulation_type": reg_type,
                    "season": SEASON,
                    "article_number": article_number,
                    "article_path": article_path,
                    "source_pdf": source_pdf,
                    "page_number": page_number,
                    "text": segment,
                }
            )
        start = end
        part_idx += 1
    return parts


def chunk_article_text(
    text: str,
    article_number: str,
    article_title: str,
    reg_type: str,
    source_pdf: str,
    page_number: int,
    article_path: list[str],
    _depth: int = 0,
) -> list[dict[str, Any]]:
    """Produce one or more chunk dicts for a single article's text.

    If the article text exceeds MAX_CHUNK_CHARS, split it into sub-parts.

    Parameters
    ----------
    _depth:
        Internal recursion guard.  Sub-article splitting is only attempted
        once (depth 0 → 1).  At depth >= 1 we fall back to hard character
        splitting to prevent infinite recursion when a sub-article's own
        text contains further embedded article-like headers.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= MAX_CHUNK_CHARS:
        return [
            {
                "chunk_id": _make_chunk_id(reg_type, article_number),
                "regulation_type": reg_type,
                "season": SEASON,
                "article_number": article_number,
                "article_path": article_path,
                "source_pdf": source_pdf,
                "page_number": page_number,
                "text": text,
            }
        ]

    # At recursion depth >= 1 always hard-split (avoid infinite recursion when
    # sub-article body itself contains article-like headers in quoted text etc.)
    if _depth >= 1:
        return _hard_split(text, article_number, reg_type, source_pdf, page_number, article_path)

    # Attempt sub-article split at depth 0.
    # Look for sub-articles that START AFTER position 0 (skip the header itself).
    sub_matches = [m for m in _SUB_ARTICLE_PAT.finditer(text) if m.start() > 0]

    if sub_matches:
        chunks: list[dict[str, Any]] = []
        boundaries = [m.start() for m in sub_matches] + [len(text)]
        # Include the preamble (text before first sub-article) if non-empty
        preamble = text[: boundaries[0]].strip()
        if preamble:
            chunks.append(
                {
                    "chunk_id": _make_chunk_id(reg_type, article_number, 0),
                    "regulation_type": reg_type,
                    "season": SEASON,
                    "article_number": article_number,
                    "article_path": article_path,
                    "source_pdf": source_pdf,
                    "page_number": page_number,
                    "text": preamble,
                }
            )
        for i, m in enumerate(sub_matches):
            sub_text = text[boundaries[i] : boundaries[i + 1]].strip()
            sub_num = m.group(1)
            sub_title = m.group(2).strip()
            sub_path = [*article_path, f"{sub_num} — {sub_title}"]
            sub_chunks = chunk_article_text(
                sub_text,
                sub_num,
                sub_title,
                reg_type,
                source_pdf,
                page_number,
                sub_path,
                _depth=_depth + 1,
            )
            chunks.extend(sub_chunks)
        return chunks

    # No sub-articles found — hard-split by character count
    return _hard_split(text, article_number, reg_type, source_pdf, page_number, article_path)


def extract_chunks(
    pdf_path: Path,
    reg_type: str,
) -> list[dict[str, Any]]:
    """Extract all article chunks from *pdf_path* for *reg_type*."""
    log.info("Extracting text from %s", pdf_path.name)
    pages = extract_text_with_pages(pdf_path)
    full_text, page_map = _build_full_text_with_page_map(pages)

    log.info("Parsed %d pages, total chars: %d", len(pages), len(full_text))

    articles = split_into_articles(full_text)
    log.info("Found %d article headers in %s", len(articles), pdf_path.name)

    all_chunks: list[dict[str, Any]] = []
    source_pdf = pdf_path.name

    for idx, (art_num, art_title, art_start) in enumerate(articles):
        # Article text runs from the match start to the start of the next article
        art_end = articles[idx + 1][2] if idx + 1 < len(articles) else len(full_text)
        art_text = full_text[art_start:art_end]

        page_num = _char_to_page(art_start, page_map)
        chapter_breadcrumbs = _extract_chapter_breadcrumbs(full_text, art_start)
        article_path = [*chapter_breadcrumbs, f"{art_num} — {art_title}"]

        chunks = chunk_article_text(
            art_text,
            art_num,
            art_title,
            reg_type,
            source_pdf,
            page_num,
            article_path,
        )
        all_chunks.extend(chunks)

    log.info(
        "Produced %d chunks for %s (%s)", len(all_chunks), pdf_path.name, reg_type
    )
    return all_chunks


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------

def _load_existing_source_pdfs(jsonl_path: Path) -> set[str]:
    """Return the set of source_pdf filenames already in the JSONL."""
    if not jsonl_path.exists():
        return set()
    seen: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                seen.add(obj.get("source_pdf", ""))
            except json.JSONDecodeError:
                pass
    return seen


def _append_chunks(jsonl_path: Path, chunks: list[dict[str, Any]]) -> None:
    """Append *chunks* to the JSONL file (one JSON object per line)."""
    with jsonl_path.open("a", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Appended %d chunks to %s", len(chunks), jsonl_path)


def _rewrite_jsonl(jsonl_path: Path, chunks: list[dict[str, Any]]) -> None:
    """Overwrite the JSONL file with *chunks*."""
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    log.info("Wrote %d chunks to %s", len(chunks), jsonl_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    reg_types: list[str],
    urls: dict[str, str] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Run the full ETL pipeline for the given regulation types.

    Parameters
    ----------
    reg_types:
        Subset of ["sporting", "technical", "financial"].
    urls:
        Optional explicit URL override per type.  If provided, discovery is
        skipped entirely for those types.
    force:
        Re-download and re-extract even if the PDF and JSONL entries exist.

    Returns
    -------
    dict mapping reg_type → chunk count produced in this run.
    """
    # Resolve URLs
    if urls:
        resolved: dict[str, str] = {}
        type_keys = list(TYPE_KEYWORDS.keys())
        for i, reg_type in enumerate(type_keys):
            url_list = list(urls.values()) if isinstance(urls, dict) else list(urls)
            if reg_type in urls:
                resolved[reg_type] = urls[reg_type]
            elif i < len(url_list):
                resolved[reg_type] = url_list[i]
        # Fall back to discovery for any type not covered
        missing = [t for t in reg_types if t not in resolved]
        if missing:
            discovered = discover_urls(missing)
            resolved.update(discovered)
    else:
        resolved = discover_urls(reg_types)

    # Load existing JSONL to support idempotency
    existing_pdfs = _load_existing_source_pdfs(JSONL_PATH) if not force else set()

    counts: dict[str, int] = {}
    new_chunks: list[dict[str, Any]] = []

    for reg_type in reg_types:
        url = resolved[reg_type]
        filename = url_to_filename(url)
        pdf_path = PDF_DIR / filename

        # Download (idempotent)
        download_pdf(url, pdf_path, force=force)

        # Skip extraction if already in JSONL
        if filename in existing_pdfs and not force:
            log.info(
                "Chunks for '%s' already in JSONL; skipping (use --force to redo).",
                filename,
            )
            counts[reg_type] = 0
            continue

        chunks = extract_chunks(pdf_path, reg_type)
        new_chunks.extend(chunks)
        counts[reg_type] = len(chunks)

    if force and new_chunks:
        # When forcing, we need to rebuild: drop old chunks for the re-processed
        # types and append the fresh ones.
        forced_types = set(reg_types)
        if JSONL_PATH.exists():
            kept: list[dict[str, Any]] = []
            with JSONL_PATH.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if obj.get("regulation_type") not in forced_types:
                            kept.append(obj)
                    except json.JSONDecodeError:
                        pass
            _rewrite_jsonl(JSONL_PATH, kept + new_chunks)
        else:
            _rewrite_jsonl(JSONL_PATH, new_chunks)
    elif new_chunks:
        _append_chunks(JSONL_PATH, new_chunks)

    total = sum(counts.values())
    log.info("Done. Total new chunks this run: %d", total)
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and chunk 2026 FIA F1 regulation PDFs."
    )
    parser.add_argument(
        "--type",
        dest="reg_type",
        choices=["sporting", "technical", "financial"],
        default=None,
        help="Process only one regulation type (default: all three).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download and re-extract even if files already exist.",
    )
    parser.add_argument(
        "--urls",
        nargs=3,
        metavar=("SPORTING_URL", "TECHNICAL_URL", "FINANCIAL_URL"),
        default=None,
        help="Explicit PDF URLs: sporting technical financial (in that order).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    reg_types = [args.reg_type] if args.reg_type else ["sporting", "technical", "financial"]

    urls: dict[str, str] | None = None
    if args.urls:
        type_keys = ["sporting", "technical", "financial"]
        urls = dict(zip(type_keys, args.urls, strict=True))

    counts = run(reg_types=reg_types, urls=urls, force=args.force)
    for reg_type, count in counts.items():
        log.info("  %s: %d new chunks", reg_type, count)


if __name__ == "__main__":
    main(sys.argv[1:])
