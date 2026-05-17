"""
Unit tests for ingestion.stats_etl.
All HTTP calls are mocked — no live API access.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestion.stats_etl import (
    _paginate,
    create_schema,
    ingest_qualifying,
    ingest_races,
    ingest_results,
    is_done,
    mark_done,
    run_etl,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memdb() -> sqlite3.Connection:
    """In-memory SQLite with the full schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    create_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

_DRIVER = {
    "driverId": "hamilton",
    "code": "HAM",
    "permanentNumber": "44",
    "givenName": "Lewis",
    "familyName": "Hamilton",
    "dateOfBirth": "1985-01-07",
    "nationality": "British",
}

_CONSTRUCTOR = {
    "constructorId": "mercedes",
    "name": "Mercedes",
    "nationality": "German",
}

_RACE_ROW = {
    "season": "2021",
    "round": "1",
    "raceName": "Bahrain Grand Prix",
    "Circuit": {
        "circuitId": "bahrain",
        "circuitName": "Bahrain International Circuit",
        "Location": {
            "lat": "26.0",
            "long": "50.5",
            "locality": "Sakhir",
            "country": "Bahrain",
        },
    },
    "date": "2021-03-28",
}


# ---------------------------------------------------------------------------
# Mock response builders
# ---------------------------------------------------------------------------


def _drivers_response(drivers: list | None = None) -> dict:
    return {
        "MRData": {
            "total": str(len(drivers or [_DRIVER])),
            "offset": "0",
            "limit": "1000",
            "DriverTable": {"Drivers": drivers or [_DRIVER]},
        }
    }


def _constructors_response(constructors: list | None = None) -> dict:
    return {
        "MRData": {
            "total": str(len(constructors or [_CONSTRUCTOR])),
            "offset": "0",
            "limit": "1000",
            "ConstructorTable": {"Constructors": constructors or [_CONSTRUCTOR]},
        }
    }


def _races_response(races: list | None = None) -> dict:
    return {
        "MRData": {
            "total": str(len(races or [_RACE_ROW])),
            "offset": "0",
            "limit": "1000",
            "RaceTable": {"Races": races or [_RACE_ROW]},
        }
    }


def _results_response(results: list | None = None) -> dict:
    result_row = {
        "number": "44",
        "position": "1",
        "positionText": "1",
        "positionOrder": "1",
        "points": "25",
        "grid": "1",
        "laps": "56",
        "status": "Finished",
        "Driver": _DRIVER,
        "Constructor": _CONSTRUCTOR,
    }
    race_with_results = {**_RACE_ROW, "Results": results or [result_row]}
    return {
        "MRData": {
            "total": "1",
            "offset": "0",
            "limit": "1000",
            "RaceTable": {"Races": [race_with_results]},
        }
    }


def _qualifying_response() -> dict:
    qual_row = {
        "number": "33",
        "position": "1",
        "Driver": {
            "driverId": "max_verstappen",
            "code": "VER",
            "permanentNumber": "33",
            "givenName": "Max",
            "familyName": "Verstappen",
            "dateOfBirth": "1997-09-30",
            "nationality": "Dutch",
        },
        "Constructor": _CONSTRUCTOR,
        "Q1": "1:30.499",
        "Q2": "1:30.318",
        "Q3": "1:28.997",
    }
    return {
        "MRData": {
            "total": "1",
            "offset": "0",
            "limit": "1000",
            "RaceTable": {"Races": [{**_RACE_ROW, "QualifyingResults": [qual_row]}]},
        }
    }


def _pitstops_response() -> dict:
    return {
        "MRData": {
            "total": "0",
            "offset": "0",
            "limit": "1000",
            "RaceTable": {"Races": [{"PitStops": []}]},
        }
    }


def _url_dispatch(url: str, params: dict | None = None) -> dict:
    """Route mock responses based on URL path fragment."""
    if "/drivers.json" in url:
        return _drivers_response()
    if "/constructors.json" in url:
        return _constructors_response()
    if "/qualifying.json" in url:
        return _qualifying_response()
    if "/results.json" in url:
        return _results_response()
    if "/pitstops.json" in url:
        return _pitstops_response()
    # Default: races
    return _races_response()


# ---------------------------------------------------------------------------
# 1. Schema creation
# ---------------------------------------------------------------------------


def test_schema_creates_all_tables(memdb: sqlite3.Connection) -> None:
    required = {
        "circuits",
        "drivers",
        "constructors",
        "races",
        "results",
        "qualifying",
        "pitstops",
        "laptimes",
        "_ingest_state",
    }
    rows = memdb.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    found = {r[0] for r in rows}
    missing = required - found
    assert not missing, f"Missing tables: {missing}"


def test_schema_foreign_keys_defined(memdb: sqlite3.Connection) -> None:
    # results must reference races, drivers, and constructors
    fks = memdb.execute("PRAGMA foreign_key_list(results)").fetchall()
    referenced_tables = {fk[2] for fk in fks}  # column index 2 = table
    assert "races" in referenced_tables
    assert "drivers" in referenced_tables
    assert "constructors" in referenced_tables


# ---------------------------------------------------------------------------
# 2. Pagination loop
# ---------------------------------------------------------------------------


def _make_race_page(races: list, total: int) -> dict:
    return {
        "MRData": {
            "total": str(total),
            "offset": "0",
            "limit": "1000",
            "RaceTable": {"Races": races},
        }
    }


def test_paginate_single_page() -> None:
    """When total <= limit, _paginate makes exactly one request."""
    response = _make_race_page([_RACE_ROW], total=1)
    with patch("ingestion.stats_etl._get", return_value=response) as mock_get:
        results = _paginate("https://example.com/races.json")
    assert len(results) == 1
    mock_get.assert_called_once()


def test_paginate_two_pages() -> None:
    """When total > 1000, _paginate issues a second request with offset=1000."""
    page1 = _make_race_page([_RACE_ROW] * 1000, total=1050)
    page2 = _make_race_page([_RACE_ROW] * 50, total=1050)
    with patch("ingestion.stats_etl._get", side_effect=[page1, page2]) as mock_get:
        results = _paginate("https://example.com/races.json")
    assert len(results) == 1050
    assert mock_get.call_count == 2
    # Second call uses offset=1000 as a keyword argument
    second_call = mock_get.call_args_list[1]
    assert second_call.kwargs["params"]["offset"] == 1000


def test_paginate_terminates_exactly_at_boundary() -> None:
    """When total equals limit exactly, only one request is made."""
    page1 = _make_race_page([_RACE_ROW] * 1000, total=1000)
    with patch("ingestion.stats_etl._get", return_value=page1) as mock_get:
        results = _paginate("https://example.com/races.json")
    assert len(results) == 1000
    mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Idempotency
# ---------------------------------------------------------------------------


def test_idempotency_same_row_count(tmp_path: Path) -> None:
    """Running the ETL twice for the same season yields identical row counts."""
    db_file = tmp_path / "test.sqlite"

    with patch("ingestion.stats_etl._get", side_effect=_url_dispatch):
        run_etl(seasons=[2021], include_laptimes=False, force=False, db_path=db_file)

    conn = sqlite3.connect(db_file)
    count_after_first = conn.execute("SELECT COUNT(*) FROM drivers").fetchone()[0]
    conn.close()

    # Second run — should skip all datasets (idempotent)
    with patch("ingestion.stats_etl._get", side_effect=_url_dispatch) as mock_get:
        run_etl(seasons=[2021], include_laptimes=False, force=False, db_path=db_file)

    # No HTTP calls should have been made (all skipped)
    assert mock_get.call_count == 0

    conn = sqlite3.connect(db_file)
    count_after_second = conn.execute("SELECT COUNT(*) FROM drivers").fetchone()[0]
    conn.close()

    assert count_after_first > 0
    assert count_after_first == count_after_second


def test_force_flag_re_fetches(tmp_path: Path) -> None:
    """--force causes a re-fetch even when _ingest_state is already marked."""
    db_file = tmp_path / "test.sqlite"

    with patch("ingestion.stats_etl._get", side_effect=_url_dispatch):
        run_etl(seasons=[2021], include_laptimes=False, force=False, db_path=db_file)

    # Now force — _get should be called again for every dataset
    with patch("ingestion.stats_etl._get", side_effect=_url_dispatch) as mock_get:
        run_etl(seasons=[2021], include_laptimes=False, force=True, db_path=db_file)

    assert mock_get.call_count > 0


# ---------------------------------------------------------------------------
# 4. Ingest state helpers
# ---------------------------------------------------------------------------


def test_is_done_false_initially(memdb: sqlite3.Connection) -> None:
    assert is_done(memdb, 2021, "drivers") is False


def test_mark_done_then_is_done(memdb: sqlite3.Connection) -> None:
    mark_done(memdb, 2021, "drivers")
    assert is_done(memdb, 2021, "drivers") is True


def test_mark_done_idempotent(memdb: sqlite3.Connection) -> None:
    mark_done(memdb, 2021, "drivers")
    mark_done(memdb, 2021, "drivers")  # should not raise
    count = memdb.execute(
        "SELECT COUNT(*) FROM _ingest_state WHERE season=2021 AND dataset='drivers'"
    ).fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# 5. ingest_races inserts circuit + race rows
# ---------------------------------------------------------------------------


def test_ingest_races_creates_circuit_and_race(memdb: sqlite3.Connection) -> None:
    with patch("ingestion.stats_etl._get", return_value=_races_response()):
        n = ingest_races(memdb, 2021)
    assert n == 1
    circuit = memdb.execute("SELECT * FROM circuits WHERE circuit_id='bahrain'").fetchone()
    assert circuit is not None
    race = memdb.execute("SELECT * FROM races WHERE season=2021 AND round=1").fetchone()
    assert race is not None


# ---------------------------------------------------------------------------
# 6. ingest_results inserts result rows + drivers + constructors
# ---------------------------------------------------------------------------


def test_ingest_results_inserts_rows(memdb: sqlite3.Connection) -> None:
    # Seed the circuit and race so ingest_results can look them up
    with patch("ingestion.stats_etl._get", return_value=_races_response()):
        ingest_races(memdb, 2021)

    with patch("ingestion.stats_etl._get", return_value=_results_response()):
        n = ingest_results(memdb, 2021)
    assert n == 1
    result = memdb.execute("SELECT points, status FROM results").fetchone()
    assert result == (25.0, "Finished")


# ---------------------------------------------------------------------------
# 7. ingest_qualifying inserts q1/q2/q3 times
# ---------------------------------------------------------------------------


def test_ingest_qualifying_inserts_times(memdb: sqlite3.Connection) -> None:
    # Seed the circuit and race so ingest_qualifying can look them up
    with patch("ingestion.stats_etl._get", return_value=_races_response()):
        ingest_races(memdb, 2021)

    with patch("ingestion.stats_etl._get", return_value=_qualifying_response()):
        n = ingest_qualifying(memdb, 2021)
    assert n == 1
    row = memdb.execute("SELECT q1, q2, q3 FROM qualifying").fetchone()
    assert row == ("1:30.499", "1:30.318", "1:28.997")
