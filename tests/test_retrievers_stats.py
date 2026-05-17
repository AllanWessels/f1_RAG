"""
Unit tests for app.retrievers.stats (StatsRetriever / query_stats).

All tests use an in-memory (or temp-file) SQLite populated with the
production schema — no live database access.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.retrievers.stats import StatsResult, StatsRetriever, query_stats

# ---------------------------------------------------------------------------
# Shared schema DDL (mirrors ingestion/stats_etl.py exactly)
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS circuits (
    circuit_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    locality     TEXT,
    country      TEXT,
    lat          REAL,
    lng          REAL
);

CREATE TABLE IF NOT EXISTS drivers (
    driver_id    TEXT PRIMARY KEY,
    code         TEXT,
    number       TEXT,
    forename     TEXT NOT NULL,
    surname      TEXT NOT NULL,
    dob          TEXT,
    nationality  TEXT
);

CREATE TABLE IF NOT EXISTS constructors (
    constructor_id TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    nationality    TEXT
);

CREATE TABLE IF NOT EXISTS races (
    race_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    season         INTEGER NOT NULL,
    round          INTEGER NOT NULL,
    circuit_id     TEXT NOT NULL REFERENCES circuits(circuit_id),
    name           TEXT NOT NULL,
    date           TEXT,
    UNIQUE (season, round)
);

CREATE TABLE IF NOT EXISTS results (
    result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id        INTEGER NOT NULL REFERENCES races(race_id),
    driver_id      TEXT NOT NULL REFERENCES drivers(driver_id),
    constructor_id TEXT NOT NULL REFERENCES constructors(constructor_id),
    grid           INTEGER,
    position       INTEGER,
    position_text  TEXT,
    position_order INTEGER,
    points         REAL,
    laps           INTEGER,
    status         TEXT,
    UNIQUE (race_id, driver_id)
);

CREATE TABLE IF NOT EXISTS qualifying (
    qualifying_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id        INTEGER NOT NULL REFERENCES races(race_id),
    driver_id      TEXT NOT NULL REFERENCES drivers(driver_id),
    constructor_id TEXT NOT NULL REFERENCES constructors(constructor_id),
    number         TEXT,
    position       INTEGER,
    q1             TEXT,
    q2             TEXT,
    q3             TEXT,
    UNIQUE (race_id, driver_id)
);

CREATE TABLE IF NOT EXISTS pitstops (
    pitstop_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id        INTEGER NOT NULL REFERENCES races(race_id),
    driver_id      TEXT NOT NULL REFERENCES drivers(driver_id),
    stop           INTEGER,
    lap            INTEGER,
    time_of_day    TEXT,
    duration       TEXT,
    UNIQUE (race_id, driver_id, stop)
);

CREATE TABLE IF NOT EXISTS laptimes (
    laptime_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id        INTEGER NOT NULL REFERENCES races(race_id),
    driver_id      TEXT NOT NULL REFERENCES drivers(driver_id),
    lap            INTEGER,
    position       INTEGER,
    time           TEXT,
    UNIQUE (race_id, driver_id, lap)
);

CREATE TABLE IF NOT EXISTS _ingest_state (
    season  INTEGER NOT NULL,
    dataset TEXT NOT NULL,
    PRIMARY KEY (season, dataset)
);
"""


# ---------------------------------------------------------------------------
# Fixture: temp-file database with production schema + minimal seed data
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """
    Return a path to a temp SQLite file that has the full production schema
    and a small seed dataset (3 races, podium results, qualifying data).
    """
    path = tmp_path / "test_f1.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(_DDL)

    # -- Circuits
    conn.executemany(
        "INSERT INTO circuits (circuit_id, name, locality, country, lat, lng) VALUES (?,?,?,?,?,?)",
        [
            ("bahrain", "Bahrain International Circuit", "Sakhir", "Bahrain", 26.0, 50.5),
            ("imola", "Autodromo Enzo e Dino Ferrari", "Imola", "Italy", 44.3, 11.7),
            ("portimao", "Autódromo Internacional do Algarve", "Portimão", "Portugal", 37.2, -8.6),
        ],
    )

    # -- Drivers
    conn.executemany(
        "INSERT INTO drivers (driver_id, code, number, forename, surname, dob, nationality) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            ("hamilton", "HAM", "44", "Lewis", "Hamilton", "1985-01-07", "British"),
            ("max_verstappen", "VER", "33", "Max", "Verstappen", "1997-09-30", "Dutch"),
            ("bottas", "BOT", "77", "Valtteri", "Bottas", "1989-08-28", "Finnish"),
        ],
    )

    # -- Constructors
    conn.executemany(
        "INSERT INTO constructors (constructor_id, name, nationality) VALUES (?,?,?)",
        [
            ("mercedes", "Mercedes", "German"),
            ("red_bull", "Red Bull", "Austrian"),
        ],
    )

    # -- Races (season 2021, rounds 1-3)
    conn.executemany(
        "INSERT INTO races (season, round, circuit_id, name, date) VALUES (?,?,?,?,?)",
        [
            (2021, 1, "bahrain", "Bahrain Grand Prix", "2021-03-28"),
            (2021, 2, "imola", "Emilia Romagna Grand Prix", "2021-04-18"),
            (2021, 3, "portimao", "Portuguese Grand Prix", "2021-05-02"),
        ],
    )

    # -- Results (race 1: HAM P1, VER P2, BOT P3)
    race1_id = conn.execute(
        "SELECT race_id FROM races WHERE season=2021 AND round=1"
    ).fetchone()[0]
    conn.executemany(
        "INSERT INTO results "
        "(race_id, driver_id, constructor_id, grid, position, position_text, "
        " position_order, points, laps, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (race1_id, "hamilton", "mercedes", 1, 1, "1", 1, 25.0, 56, "Finished"),
            (race1_id, "max_verstappen", "red_bull", 3, 2, "2", 2, 18.0, 56, "Finished"),
            (race1_id, "bottas", "mercedes", 2, 3, "3", 3, 15.0, 56, "Finished"),
        ],
    )

    # -- Qualifying (race 1)
    conn.executemany(
        "INSERT INTO qualifying "
        "(race_id, driver_id, constructor_id, number, position, q1, q2, q3) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            (race1_id, "hamilton", "mercedes", "44", 1, "1:29.5", "1:28.9", "1:27.9"),
            (race1_id, "max_verstappen", "red_bull", "33", 2, "1:29.8", "1:29.2", "1:28.3"),
            (race1_id, "bottas", "mercedes", "77", 3, "1:30.1", "1:29.5", "1:28.7"),
        ],
    )

    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Helper: run query_stats against our fixture database
# ---------------------------------------------------------------------------


def _run(db_path: Path, sql: str, max_rows: int = 200) -> StatsResult:
    return query_stats(sql=sql, db_path=db_path, max_rows=max_rows)


# ---------------------------------------------------------------------------
# Test 1: Successful SELECT returns correct rows
# ---------------------------------------------------------------------------


def test_select_races_by_season(db_path: Path) -> None:
    """SELECT name FROM races WHERE season = 2021 returns all 3 race names."""
    result = _run(db_path, "SELECT name FROM races WHERE season = 2021")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.columns == ["name"]
    assert result.row_count == 3
    assert result.truncated is False

    names = [r["name"] for r in result.rows]
    assert "Bahrain Grand Prix" in names
    assert "Emilia Romagna Grand Prix" in names
    assert "Portuguese Grand Prix" in names


# ---------------------------------------------------------------------------
# Test 2: Read-only enforcement — INSERT is rejected
# ---------------------------------------------------------------------------


def test_insert_is_rejected(db_path: Path) -> None:
    """INSERT statement must be rejected with an error, no rows returned."""
    result = _run(
        db_path,
        "INSERT INTO races (season, round, circuit_id, name) VALUES (2099, 1, 'bahrain', 'Test')",
    )

    assert result.error is not None
    assert len(result.rows) == 0
    assert result.row_count == 0

    # Verify the DB was not mutated (still only 3 races)
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    conn.close()
    assert count == 3


def test_drop_is_rejected(db_path: Path) -> None:
    """DROP TABLE must be rejected."""
    result = _run(db_path, "DROP TABLE races")
    assert result.error is not None
    assert len(result.rows) == 0


def test_update_is_rejected(db_path: Path) -> None:
    """UPDATE must be rejected."""
    result = _run(db_path, "UPDATE races SET name = 'X' WHERE season = 2021")
    assert result.error is not None
    assert len(result.rows) == 0


def test_pragma_is_rejected(db_path: Path) -> None:
    """PRAGMA statements must be rejected."""
    result = _run(db_path, "PRAGMA table_info(races)")
    assert result.error is not None
    assert len(result.rows) == 0


# ---------------------------------------------------------------------------
# Test 3: Multi-statement rejection
# ---------------------------------------------------------------------------


def test_multi_statement_rejected(db_path: Path) -> None:
    """SELECT * FROM races; DROP TABLE races; must be rejected."""
    result = _run(db_path, "SELECT * FROM races; DROP TABLE races;")
    assert result.error is not None
    assert len(result.rows) == 0


def test_select_semicolon_select_rejected(db_path: Path) -> None:
    """Two SELECT statements separated by ; must also be rejected."""
    result = _run(db_path, "SELECT 1; SELECT 2")
    assert result.error is not None
    assert len(result.rows) == 0


# ---------------------------------------------------------------------------
# Test 4: Row truncation
# ---------------------------------------------------------------------------


def test_row_truncation(db_path: Path) -> None:
    """
    When the query returns more rows than max_rows, the result is truncated.
    We seed 3 races and cap at max_rows=2, so truncated should be True.
    """
    result = query_stats(
        sql="SELECT * FROM races",
        db_path=db_path,
        max_rows=2,
    )
    assert result.error is None
    assert len(result.rows) == 2
    assert result.row_count == 2
    assert result.truncated is True


def test_no_truncation_when_within_limit(db_path: Path) -> None:
    """When row count <= max_rows, truncated is False."""
    result = query_stats(
        sql="SELECT * FROM races WHERE season = 2021",
        db_path=db_path,
        max_rows=200,
    )
    assert result.error is None
    assert result.truncated is False
    assert result.row_count == 3


# ---------------------------------------------------------------------------
# Test 5: Joins work correctly
# ---------------------------------------------------------------------------


def test_join_races_results_drivers_constructors(db_path: Path) -> None:
    """
    A query joining races, results, drivers, constructors returns sensible output.
    """
    sql = """
        SELECT
            r.name        AS race_name,
            d.forename || ' ' || d.surname AS driver_name,
            c.name        AS constructor_name,
            res.position  AS finish_position,
            res.points    AS points
        FROM races r
        JOIN results res ON res.race_id = r.race_id
        JOIN drivers d   ON d.driver_id = res.driver_id
        JOIN constructors c ON c.constructor_id = res.constructor_id
        WHERE r.season = 2021 AND r.round = 1
        ORDER BY res.position_order
    """
    result = _run(db_path, sql)

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.row_count == 3
    assert "race_name" in result.columns
    assert "driver_name" in result.columns
    assert "constructor_name" in result.columns

    # Hamilton should be P1 with 25 points
    top = result.rows[0]
    assert top["driver_name"] == "Lewis Hamilton"
    assert top["finish_position"] == 1
    assert top["points"] == 25.0
    assert top["constructor_name"] == "Mercedes"

    # Verstappen should be P2
    second = result.rows[1]
    assert second["driver_name"] == "Max Verstappen"
    assert second["finish_position"] == 2


def test_join_qualifying_drivers(db_path: Path) -> None:
    """Qualifying join returns Q3 times for the pole race."""
    sql = """
        SELECT
            d.code       AS driver_code,
            q.position   AS quali_pos,
            q.q3         AS q3_time
        FROM qualifying q
        JOIN drivers d ON d.driver_id = q.driver_id
        JOIN races r   ON r.race_id = q.race_id
        WHERE r.season = 2021 AND r.round = 1
        ORDER BY q.position
    """
    result = _run(db_path, sql)

    assert result.error is None
    assert result.row_count == 3
    assert result.rows[0]["driver_code"] == "HAM"
    assert result.rows[0]["q3_time"] == "1:27.9"


# ---------------------------------------------------------------------------
# Test 6: Bad SQL syntax is caught and surfaced as error (no exception bubbles)
# ---------------------------------------------------------------------------


def test_bad_syntax_returns_error(db_path: Path) -> None:
    """Syntactically invalid SQL must not raise — error is in result.error."""
    result = _run(db_path, "SELECT * FROOOM races WHER season = 2021")
    assert result.error is not None
    assert len(result.rows) == 0
    # No exception should have propagated


def test_nonexistent_table_returns_error(db_path: Path) -> None:
    """Query against a non-existent table returns error, not exception."""
    result = _run(db_path, "SELECT * FROM nonexistent_table_xyz")
    assert result.error is not None
    assert len(result.rows) == 0


# ---------------------------------------------------------------------------
# Test 7: StatsRetriever class returns the same result as query_stats
# ---------------------------------------------------------------------------


def test_stats_retriever_matches_functional_interface(db_path: Path) -> None:
    """StatsRetriever.run() must return the same result as query_stats()."""
    sql = "SELECT name FROM races WHERE season = 2021 ORDER BY round"

    functional_result = query_stats(sql=sql, db_path=db_path)
    retriever = StatsRetriever(db_path=db_path)
    class_result = retriever.run(sql)

    assert functional_result.error == class_result.error
    assert functional_result.columns == class_result.columns
    assert functional_result.rows == class_result.rows
    assert functional_result.row_count == class_result.row_count
    assert functional_result.truncated == class_result.truncated


def test_stats_retriever_respects_max_rows(db_path: Path) -> None:
    """StatsRetriever.max_rows is honoured in .run()."""
    retriever = StatsRetriever(db_path=db_path, max_rows=1)
    result = retriever.run("SELECT * FROM races WHERE season = 2021")

    assert result.error is None
    assert len(result.rows) == 1
    assert result.truncated is True


# ---------------------------------------------------------------------------
# Test 8: WITH (CTE) queries are accepted
# ---------------------------------------------------------------------------


def test_with_cte_accepted(db_path: Path) -> None:
    """WITH ... SELECT (CTE) must be accepted as a valid read-only query."""
    sql = """
        WITH race_winners AS (
            SELECT res.race_id, res.driver_id
            FROM results res
            WHERE res.position = 1
        )
        SELECT d.surname, r.name AS race_name
        FROM race_winners rw
        JOIN drivers d ON d.driver_id = rw.driver_id
        JOIN races r   ON r.race_id = rw.race_id
        WHERE r.season = 2021
    """
    result = _run(db_path, sql)
    assert result.error is None
    assert result.row_count == 1
    assert result.rows[0]["surname"] == "Hamilton"


# ---------------------------------------------------------------------------
# Test 9: Trailing semicolon on a single SELECT is OK
# ---------------------------------------------------------------------------


def test_trailing_semicolon_allowed(db_path: Path) -> None:
    """A single SELECT with a trailing semicolon should be accepted."""
    result = _run(db_path, "SELECT COUNT(*) AS n FROM races;")
    assert result.error is None
    assert result.rows[0]["n"] == 3


# ---------------------------------------------------------------------------
# Test 10: StatsResult is a valid Pydantic model
# ---------------------------------------------------------------------------


def test_stats_result_model_dump(db_path: Path) -> None:
    """StatsResult.model_dump() produces the expected keys."""
    result = _run(db_path, "SELECT name FROM races WHERE season = 2021 LIMIT 1")
    dumped = result.model_dump()
    assert set(dumped.keys()) == {"sql", "columns", "rows", "row_count", "truncated", "error"}
    assert dumped["error"] is None
    assert isinstance(dumped["rows"], list)
    assert isinstance(dumped["columns"], list)
