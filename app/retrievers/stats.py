"""
Stats retriever: read-only SQL queries over the Jolpica-derived SQLite database.

Exposes:
  - query_stats(sql, db_path, max_rows) -> StatsResult   (functional interface)
  - StatsRetriever(db_path, max_rows).run(sql) -> StatsResult  (class interface)
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.tracing import traced

# Default database location (relative to project root)
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "f1_stats.sqlite"
DEFAULT_MAX_ROWS = 200

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class StatsResult(BaseModel):
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# SQL validation helpers
# ---------------------------------------------------------------------------

# Patterns for blocked statement types (case-insensitive, word-boundary anchored)
_BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|REPLACE|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)

# Strip single-line (--) and multi-line (/* */) SQL comments
_SQL_LINE_COMMENT = re.compile(r"--[^\r\n]*")
_SQL_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(sql: str) -> str:
    """Remove SQL comments from a query string."""
    sql = _SQL_BLOCK_COMMENT.sub(" ", sql)
    sql = _SQL_LINE_COMMENT.sub(" ", sql)
    return sql


def _validate_sql(sql: str) -> str | None:
    """
    Return an error message if the SQL is not a single SELECT/WITH statement,
    or None if it's acceptable.
    """
    stripped = _strip_comments(sql).strip().rstrip(";").strip()

    # Must start with SELECT or WITH
    if not re.match(r"^(SELECT|WITH)\b", stripped, re.IGNORECASE):
        return (
            "Only SELECT or WITH statements are allowed. "
            "Rejected keywords: INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, "
            "ATTACH, DETACH, REPLACE, PRAGMA, VACUUM."
        )

    # Must not contain any blocked keyword anywhere in the original SQL
    if _BLOCKED_KEYWORDS.search(sql):
        return (
            "Statement contains a disallowed keyword "
            f"({_BLOCKED_KEYWORDS.search(sql).group(0).upper()}). "
            "Only read-only SELECT/WITH queries are permitted."
        )

    # Detect multiple statements: look for a semicolon that is not trailing
    # Work on the comment-stripped version to avoid false positives
    # Split on semicolons and check if any non-empty statement remains after the first
    parts = [p.strip() for p in stripped.split(";") if p.strip()]
    if len(parts) > 1:
        return (
            "Multiple SQL statements detected (found ';' followed by more SQL). "
            "Only a single SELECT/WITH statement is allowed."
        )

    return None


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------


def _execute_sql(
    sql: str,
    db_path: Path,
    max_rows: int,
) -> StatsResult:
    """
    Validate and execute a single read-only SQL query against the SQLite database.
    Returns a StatsResult; never raises.
    """
    # Syntactic validation
    err = _validate_sql(sql)
    if err:
        return StatsResult(sql=sql, columns=[], rows=[], row_count=0, truncated=False, error=err)

    # Open the database in read-only mode via URI
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError as exc:
        return StatsResult(
            sql=sql, columns=[], rows=[], row_count=0, truncated=False,
            error=f"Could not open database: {exc}",
        )

    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [description[0] for description in cursor.description or []]

        # Fetch up to max_rows + 1 so we can detect truncation without loading everything
        raw_rows = cursor.fetchmany(max_rows + 1)
        truncated = len(raw_rows) > max_rows
        if truncated:
            raw_rows = raw_rows[:max_rows]

        rows = [dict(row) for row in raw_rows]
        row_count = len(rows)

        return StatsResult(
            sql=sql,
            columns=columns,
            rows=rows,
            row_count=row_count,
            truncated=truncated,
            error=None,
        )

    except sqlite3.Error as exc:
        return StatsResult(
            sql=sql, columns=[], rows=[], row_count=0, truncated=False,
            error=f"SQL error: {exc}",
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public interfaces
# ---------------------------------------------------------------------------


def query_stats(
    sql: str,
    db_path: Path = DEFAULT_DB_PATH,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> StatsResult:
    """
    Execute a read-only SQL SELECT/WITH query against the F1 stats SQLite database.

    Args:
        sql:      The SQL query to execute. Must be a single SELECT or WITH statement.
        db_path:  Path to the SQLite database file.
        max_rows: Maximum number of rows to return (default 200).

    Returns:
        StatsResult with columns, rows, row_count, truncated flag, and optional error.
    """
    return _execute_sql(sql=sql, db_path=db_path, max_rows=max_rows)


class StatsRetriever:
    """
    Class-based interface for query_stats, suitable for ergonomic tracing wraps.

    Usage:
        retriever = StatsRetriever(db_path=Path("data/f1_stats.sqlite"))
        result = retriever.run("SELECT name FROM races WHERE season = 2021")
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self.db_path = db_path
        self.max_rows = max_rows

    @traced(name="stats_retriever_run")
    def run(self, sql: str) -> StatsResult:
        """Execute a read-only SQL query and return a StatsResult."""
        return _execute_sql(sql=sql, db_path=self.db_path, max_rows=self.max_rows)
