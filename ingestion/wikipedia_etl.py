"""
Wikipedia race-report scraper for F1 Grand Prix pages (2014-2025 modern era).

Entry points:
    python -m ingestion.wikipedia_etl              # full 2014-2025
    python -m ingestion.wikipedia_etl --year 2021  # one season
    python -m ingestion.wikipedia_etl --force      # re-fetch all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wikipedia_etl")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_API = "https://en.wikipedia.org/api/rest_v1"
USER_AGENT = "f1-rag-etl/0.1 (https://github.com/USER/f1_rag; contact@example.com)"
CHUNK_TARGET_CHARS = 3200  # ~800 tokens @ 4 chars/token
REQ_DELAY = 1.0  # seconds between requests (polite)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_JSONL = DATA_DIR / "wikipedia_races.jsonl"
STATE_FILE = DATA_DIR / "wikipedia_races.state.json"

# ---------------------------------------------------------------------------
# F1 GP calendar 2014-2025 (canonical Wikipedia page-title suffixes)
# ---------------------------------------------------------------------------
# Format: (year, gp_name, wiki_title_slug)
# wiki_title_slug is the Wikipedia article title (underscores), without year prefix.
# The full title is f"{year}_{slug}".
GP_CALENDAR: dict[int, list[tuple[str, str]]] = {
    2014: [
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Malaysian Grand Prix", "Malaysian_Grand_Prix"),
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Chinese Grand Prix", "Chinese_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("German Grand Prix", "German_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("Russian Grand Prix", "Russian_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Brazilian Grand Prix", "Brazilian_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2015: [
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Malaysian Grand Prix", "Malaysian_Grand_Prix"),
        ("Chinese Grand Prix", "Chinese_Grand_Prix"),
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("Russian Grand Prix", "Russian_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Mexican Grand Prix", "Mexican_Grand_Prix"),
        ("Brazilian Grand Prix", "Brazilian_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2016: [
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Chinese Grand Prix", "Chinese_Grand_Prix"),
        ("Russian Grand Prix", "Russian_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("European Grand Prix", "European_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("German Grand Prix", "German_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("Malaysian Grand Prix", "Malaysian_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Mexican Grand Prix", "Mexican_Grand_Prix"),
        ("Brazilian Grand Prix", "Brazilian_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2017: [
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Chinese Grand Prix", "Chinese_Grand_Prix"),
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Russian Grand Prix", "Russian_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("Azerbaijan Grand Prix", "Azerbaijan_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("Malaysian Grand Prix", "Malaysian_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Mexican Grand Prix", "Mexican_Grand_Prix"),
        ("Brazilian Grand Prix", "Brazilian_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2018: [
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Chinese Grand Prix", "Chinese_Grand_Prix"),
        ("Azerbaijan Grand Prix", "Azerbaijan_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("French Grand Prix", "French_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("German Grand Prix", "German_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("Russian Grand Prix", "Russian_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Mexican Grand Prix", "Mexican_Grand_Prix"),
        ("Brazilian Grand Prix", "Brazilian_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2019: [
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Chinese Grand Prix", "Chinese_Grand_Prix"),
        ("Azerbaijan Grand Prix", "Azerbaijan_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("French Grand Prix", "French_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("German Grand Prix", "German_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("Russian Grand Prix", "Russian_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("Mexican Grand Prix", "Mexican_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Brazilian Grand Prix", "Brazilian_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2020: [
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("Styrian Grand Prix", "Styrian_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("70th Anniversary Grand Prix", "70th_Anniversary_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Tuscan Grand Prix", "Tuscan_Grand_Prix"),
        ("Russian Grand Prix", "Russian_Grand_Prix"),
        ("Eifel Grand Prix", "Eifel_Grand_Prix"),
        ("Portuguese Grand Prix", "Portuguese_Grand_Prix"),
        ("Emilia Romagna Grand Prix", "Emilia_Romagna_Grand_Prix"),
        ("Turkish Grand Prix", "Turkish_Grand_Prix"),
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Sakhir Grand Prix", "Sakhir_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2021: [
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Emilia Romagna Grand Prix", "Emilia_Romagna_Grand_Prix"),
        ("Portuguese Grand Prix", "Portuguese_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Azerbaijan Grand Prix", "Azerbaijan_Grand_Prix"),
        ("French Grand Prix", "French_Grand_Prix"),
        ("Styrian Grand Prix", "Styrian_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Dutch Grand Prix", "Dutch_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Russian Grand Prix", "Russian_Grand_Prix"),
        ("Turkish Grand Prix", "Turkish_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Mexico City Grand Prix", "Mexico_City_Grand_Prix"),
        ("São Paulo Grand Prix", "S%C3%A3o_Paulo_Grand_Prix"),
        ("Qatar Grand Prix", "Qatar_Grand_Prix"),
        ("Saudi Arabian Grand Prix", "Saudi_Arabian_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2022: [
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Saudi Arabian Grand Prix", "Saudi_Arabian_Grand_Prix"),
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Emilia Romagna Grand Prix", "Emilia_Romagna_Grand_Prix"),
        ("Miami Grand Prix", "Miami_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Azerbaijan Grand Prix", "Azerbaijan_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("French Grand Prix", "French_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Dutch Grand Prix", "Dutch_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Mexico City Grand Prix", "Mexico_City_Grand_Prix"),
        ("São Paulo Grand Prix", "S%C3%A3o_Paulo_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2023: [
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Saudi Arabian Grand Prix", "Saudi_Arabian_Grand_Prix"),
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Azerbaijan Grand Prix", "Azerbaijan_Grand_Prix"),
        ("Miami Grand Prix", "Miami_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Dutch Grand Prix", "Dutch_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("Qatar Grand Prix", "Qatar_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Mexico City Grand Prix", "Mexico_City_Grand_Prix"),
        ("São Paulo Grand Prix", "S%C3%A3o_Paulo_Grand_Prix"),
        ("Las Vegas Grand Prix", "Las_Vegas_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2024: [
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Saudi Arabian Grand Prix", "Saudi_Arabian_Grand_Prix"),
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("Chinese Grand Prix", "Chinese_Grand_Prix"),
        ("Miami Grand Prix", "Miami_Grand_Prix"),
        ("Emilia Romagna Grand Prix", "Emilia_Romagna_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Dutch Grand Prix", "Dutch_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Azerbaijan Grand Prix", "Azerbaijan_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Mexico City Grand Prix", "Mexico_City_Grand_Prix"),
        ("São Paulo Grand Prix", "S%C3%A3o_Paulo_Grand_Prix"),
        ("Las Vegas Grand Prix", "Las_Vegas_Grand_Prix"),
        ("Qatar Grand Prix", "Qatar_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
    2025: [
        ("Australian Grand Prix", "Australian_Grand_Prix"),
        ("Chinese Grand Prix", "Chinese_Grand_Prix"),
        ("Japanese Grand Prix", "Japanese_Grand_Prix"),
        ("Bahrain Grand Prix", "Bahrain_Grand_Prix"),
        ("Saudi Arabian Grand Prix", "Saudi_Arabian_Grand_Prix"),
        ("Miami Grand Prix", "Miami_Grand_Prix"),
        ("Emilia Romagna Grand Prix", "Emilia_Romagna_Grand_Prix"),
        ("Monaco Grand Prix", "Monaco_Grand_Prix"),
        ("Spanish Grand Prix", "Spanish_Grand_Prix"),
        ("Canadian Grand Prix", "Canadian_Grand_Prix"),
        ("Austrian Grand Prix", "Austrian_Grand_Prix"),
        ("British Grand Prix", "British_Grand_Prix"),
        ("Belgian Grand Prix", "Belgian_Grand_Prix"),
        ("Hungarian Grand Prix", "Hungarian_Grand_Prix"),
        ("Dutch Grand Prix", "Dutch_Grand_Prix"),
        ("Italian Grand Prix", "Italian_Grand_Prix"),
        ("Azerbaijan Grand Prix", "Azerbaijan_Grand_Prix"),
        ("Singapore Grand Prix", "Singapore_Grand_Prix"),
        ("United States Grand Prix", "United_States_Grand_Prix"),
        ("Mexico City Grand Prix", "Mexico_City_Grand_Prix"),
        ("São Paulo Grand Prix", "S%C3%A3o_Paulo_Grand_Prix"),
        ("Las Vegas Grand Prix", "Las_Vegas_Grand_Prix"),
        ("Qatar Grand Prix", "Qatar_Grand_Prix"),
        ("Abu Dhabi Grand Prix", "Abu_Dhabi_Grand_Prix"),
    ],
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Retry on HTTP 429 or 5xx status codes."""
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, requests.ConnectionError | requests.Timeout)


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _get(session: requests.Session, url: str) -> requests.Response:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html"})
    return s


# ---------------------------------------------------------------------------
# Wikipedia page fetcher
# ---------------------------------------------------------------------------

def build_page_title(year: int, slug: str) -> str:
    """Return the Wikipedia page title like '2021_Abu_Dhabi_Grand_Prix'."""
    return f"{year}_{slug}"


def fetch_html(session: requests.Session, page_title: str) -> str | None:
    """Fetch parsed HTML from the Wikipedia REST API."""
    url = f"{BASE_API}/page/html/{page_title}"
    try:
        resp = _get(session, url)
        return resp.text
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.warning("Page not found: %s", page_title)
            return None
        raise


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

# CSS classes / ids that mark noise sections to skip
_SKIP_CLASSES = {
    "infobox",
    "infobox_v2",
    "navbox",
    "navbox-inner",
    "navbox-group",
    "navbox-list",
    "reflist",
    "references",
    "mw-references-wrap",
    "mw-editsection",
    "reference",
    "thumb",
    "toc",
    "wikitable",  # race-result tables — we get narratives only
    "hatnote",
    "sidebar",
    "succession-box",
}

_SKIP_IDS = {"toc", "References", "See_also", "External_links", "Notes"}

_SKIP_SECTION_TITLES = {
    "references",
    "see also",
    "external links",
    "notes",
    "further reading",
    "footnotes",
    "sources",
}


def _is_noise(tag: Tag) -> bool:
    """Return True if this tag is in a noise class/id we want to skip."""
    cls = set(tag.get("class", []))
    if cls & _SKIP_CLASSES:
        return True
    tid = tag.get("id", "")
    return bool(tid in _SKIP_IDS)


def _get_text(tag: Tag) -> str:
    """Return clean text from a tag, stripping edit-section links."""
    for el in tag.find_all(class_="mw-editsection"):
        el.decompose()
    return tag.get_text(separator=" ", strip=True)


def _normalize_section_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


class Section:
    """A logical section in the article with its heading path and paragraphs."""

    def __init__(self, path: list[str]) -> None:
        self.path = path[:]
        self.paragraphs: list[str] = []

    def add_paragraph(self, text: str) -> None:
        text = text.strip()
        if text:
            self.paragraphs.append(text)

    def is_empty(self) -> bool:
        return not self.paragraphs


def parse_sections(html: str) -> list[Section]:
    """
    Parse Wikipedia REST API HTML into a flat list of Section objects.

    The REST API returns HTML where the article body is inside <section> tags,
    with headings (h1-h6) indicating section names. We walk them and extract
    paragraph text, skipping noise elements.
    """
    soup = BeautifulSoup(html, "html.parser")

    # The REST API wraps content in <body> → article body
    body = soup.find("body") or soup

    sections: list[Section] = []
    current_path: list[str] = []  # stack of heading titles

    # Tags we always recurse into (generic containers)
    _CONTAINER_TAGS = {"body", "html", "div", "article", "main", "section", "header", "footer"}

    def process_node(node: Tag | NavigableString) -> None:
        if isinstance(node, NavigableString):
            return

        tag_name = node.name or ""

        # Skip noise elements entirely
        if _is_noise(node):
            return

        # Heading tags update the section path.
        # Wikipedia REST API uses h2 for top-level sections, h3 for subsections,
        # so we treat h2 as depth-0 (index 0) in the breadcrumb path.
        if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            raw_level = int(tag_name[1])
            # h1 is the article title — skip it in the breadcrumb
            if raw_level == 1:
                return
            # h2 → depth index 0, h3 → depth index 1, etc.
            depth_idx = raw_level - 2
            title = _normalize_section_title(_get_text(node))
            if title.lower() in _SKIP_SECTION_TITLES:
                return
            # Replace path from depth_idx onwards with this heading
            current_path[depth_idx:] = [title]
            sections.append(Section(current_path))
            return

        # Paragraph tags: extract text
        if tag_name == "p":
            text = _get_text(node)
            if text and len(text) > 20:  # skip stubs / empty paragraphs
                if not sections:
                    sections.append(Section([]))
                sections[-1].add_paragraph(text)
            return

        # Recurse into container / structural elements
        if tag_name in _CONTAINER_TAGS:
            for child in node.children:
                process_node(child)  # type: ignore[arg-type]

    process_node(body)  # type: ignore[arg-type]

    return [s for s in sections if not s.is_empty()]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_section(section: Section, target_chars: int = CHUNK_TARGET_CHARS) -> list[dict[str, Any]]:
    """
    Split section paragraphs into chunks capped at ~target_chars.
    Returns a list of dicts with keys: section_path, text.
    """
    chunks: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_len = 0

    def flush() -> None:
        if current_parts:
            chunks.append(
                {
                    "section_path": section.path[:],
                    "text": " ".join(current_parts),
                }
            )
            current_parts.clear()
            nonlocal current_len
            current_len = 0

    for para in section.paragraphs:
        para_len = len(para)
        # If adding this paragraph would exceed the target, flush first
        if current_len > 0 and current_len + para_len + 1 > target_chars:
            flush()
        # If a single paragraph exceeds the target, split it by sentences
        if para_len > target_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if current_len > 0 and current_len + len(sent) + 1 > target_chars:
                    flush()
                current_parts.append(sent)
                current_len += len(sent) + 1
        else:
            current_parts.append(para)
            current_len += para_len + 1

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Chunk ID and metadata
# ---------------------------------------------------------------------------

def make_chunk_id(year: int, gp_name: str, section_path: list[str], index: int) -> str:
    """Generate a stable chunk_id."""
    gp_slug = re.sub(r"[^a-z0-9]+", "_", gp_name.lower()).strip("_")
    section_slug = re.sub(r"[^a-z0-9]+", "_", "_".join(section_path).lower()).strip("_") or "intro"
    return f"{year}_{gp_slug}__{section_slug}__{index:04d}"


def build_chunks(
    year: int,
    gp_name: str,
    page_title: str,
    html: str,
) -> list[dict[str, Any]]:
    """Parse HTML and return all chunks with full metadata."""
    sections = parse_sections(html)
    all_chunks: list[dict[str, Any]] = []
    global_idx = 0

    for section in sections:
        raw_chunks = chunk_section(section)
        for raw in raw_chunks:
            chunk: dict[str, Any] = {
                "chunk_id": make_chunk_id(year, gp_name, raw["section_path"], global_idx),
                "year": year,
                "gp_name": gp_name,
                "page_title": page_title,
                "page_url": f"https://en.wikipedia.org/wiki/{page_title}",
                "section_path": raw["section_path"],
                "text": raw["text"],
            }
            all_chunks.append(chunk)
            global_idx += 1

    return all_chunks


# ---------------------------------------------------------------------------
# State management (idempotency)
# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict[str, bool]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(path: Path, state: dict[str, bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def state_key(year: int, gp_name: str) -> str:
    return f"{year}::{gp_name}"


# ---------------------------------------------------------------------------
# Main ETL logic
# ---------------------------------------------------------------------------

def run_etl(
    years: list[int] | None = None,
    force: bool = False,
    output_path: Path = OUTPUT_JSONL,
    state_path: Path = STATE_FILE,
    session: requests.Session | None = None,
    delay: float = REQ_DELAY,
) -> dict[str, int]:
    """
    Run the ETL for the given years (default: all 2014-2025).

    Returns a dict mapping '{year} {gp_name}' -> chunk_count.
    """
    if years is None:
        years = sorted(GP_CALENDAR.keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {} if force else load_state(state_path)
    session = session or make_session()

    chunk_counts: dict[str, int] = {}

    # Open output file in append mode (idempotency: completed pages skipped)
    with output_path.open("a", encoding="utf-8") as fh:
        for year in years:
            calendar = GP_CALENDAR.get(year, [])
            if not calendar:
                logger.warning("No calendar data for year %d", year)
                continue

            for gp_name, slug in calendar:
                key = state_key(year, gp_name)

                if state.get(key) and not force:
                    logger.info("Skipping (already done): %d %s", year, gp_name)
                    chunk_counts[f"{year} {gp_name}"] = 0  # unknown, but done
                    continue

                page_title = build_page_title(year, slug)
                logger.info("Fetching: %s", page_title)

                html = fetch_html(session, page_title)
                if html is None:
                    logger.warning("Skipping missing page: %s", page_title)
                    state[key] = True  # mark as done so we don't retry 404s
                    save_state(state_path, state)
                    chunk_counts[f"{year} {gp_name}"] = 0
                    continue

                chunks = build_chunks(year, gp_name, page_title, html)
                for chunk in chunks:
                    fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                fh.flush()

                state[key] = True
                save_state(state_path, state)
                chunk_counts[f"{year} {gp_name}"] = len(chunks)
                logger.info("  → %d chunks", len(chunks))

                time.sleep(delay)

    return chunk_counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Wikipedia F1 GP race reports.")
    parser.add_argument("--year", type=int, help="Scrape a single season (e.g. 2021)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore state file and re-fetch everything",
    )
    args = parser.parse_args()

    years = [args.year] if args.year else None

    counts = run_etl(years=years, force=args.force)
    print("\n=== Chunk counts per race ===")
    for label, count in counts.items():
        print(f"  {label}: {count}")
    total = sum(counts.values())
    print(f"\nTotal chunks written: {total}")


if __name__ == "__main__":
    main()
