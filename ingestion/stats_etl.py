"""
Jolpica-F1 → SQLite ETL for the modern era (2014-2025).

Usage:
    python -m ingestion.stats_etl              # all seasons 2014-2025, no laptimes
    python -m ingestion.stats_etl --force      # re-fetch everything
    python -m ingestion.stats_etl --laptimes   # include per-lap times
    python -m ingestion.stats_etl --season 2021  # single season
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
DB_PATH = Path(__file__).parent.parent / "data" / "f1_stats.sqlite"
SEASONS = list(range(2014, 2026))  # 2014 inclusive, 2025 inclusive
POLITE_DELAY = 0.25  # seconds between requests


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    pass


@retry(
    retry=retry_if_exception_type((RateLimitError, requests.exceptions.ConnectionError)),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(7),
)
def _get(url: str, params: dict | None = None) -> dict:
    """GET with retry on 429/5xx and polite delay."""
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise RateLimitError(f"HTTP {resp.status_code} from {url}")
    resp.raise_for_status()
    time.sleep(POLITE_DELAY)
    return resp.json()


def _paginate(url: str) -> list[dict]:
    """Fetch all pages of a Jolpica endpoint returning MRData, limit=1000."""
    params = {"limit": 1000, "offset": 0}
    results: list[dict] = []
    while True:
        data = _get(url, params=params)
        mr = data["MRData"]
        total = int(mr["total"])
        # Collect whichever table the endpoint returns
        table = next(
            (v for k, v in mr.items() if isinstance(v, dict) and k.endswith("Table")),
            {},
        )
        # The inner list key varies; grab the first list value
        for v in table.values():
            if isinstance(v, list):
                results.extend(v)
                break
        offset = int(params["offset"]) + 1000
        if offset >= total:
            break
        params = {"limit": 1000, "offset": offset}
    return results


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DDL = """
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


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()


# ---------------------------------------------------------------------------
# Insert helpers (upsert-style via INSERT OR IGNORE / INSERT OR REPLACE)
# ---------------------------------------------------------------------------


def _upsert_circuit(conn: sqlite3.Connection, circuit: dict) -> None:
    loc = circuit.get("Location", {})
    conn.execute(
        """
        INSERT OR IGNORE INTO circuits (circuit_id, name, locality, country, lat, lng)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            circuit["circuitId"],
            circuit["circuitName"],
            loc.get("locality"),
            loc.get("country"),
            float(loc["lat"]) if loc.get("lat") else None,
            float(loc["long"]) if loc.get("long") else None,
        ),
    )


def _upsert_driver(conn: sqlite3.Connection, driver: dict) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO drivers (driver_id, code, number, forename, surname, dob, nationality)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            driver["driverId"],
            driver.get("code"),
            driver.get("permanentNumber"),
            driver["givenName"],
            driver["familyName"],
            driver.get("dateOfBirth"),
            driver.get("nationality"),
        ),
    )


def _upsert_constructor(conn: sqlite3.Connection, constructor: dict) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO constructors (constructor_id, name, nationality)
        VALUES (?, ?, ?)
        """,
        (
            constructor["constructorId"],
            constructor["name"],
            constructor.get("nationality"),
        ),
    )


def _get_or_create_race(conn: sqlite3.Connection, season: int, race: dict) -> int:
    """Return the race_id, inserting the race row if absent."""
    circuit = race["Circuit"]
    _upsert_circuit(conn, circuit)
    conn.execute(
        """
        INSERT OR IGNORE INTO races (season, round, circuit_id, name, date)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            season,
            int(race["round"]),
            circuit["circuitId"],
            race["raceName"],
            race.get("date"),
        ),
    )
    row = conn.execute(
        "SELECT race_id FROM races WHERE season=? AND round=?",
        (season, int(race["round"])),
    ).fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# Per-dataset ingest functions
# ---------------------------------------------------------------------------


def ingest_drivers(conn: sqlite3.Connection, season: int) -> int:
    drivers = _paginate(f"{BASE_URL}/{season}/drivers.json")
    for d in drivers:
        _upsert_driver(conn, d)
    conn.commit()
    return len(drivers)


def ingest_constructors(conn: sqlite3.Connection, season: int) -> int:
    constructors = _paginate(f"{BASE_URL}/{season}/constructors.json")
    for c in constructors:
        _upsert_constructor(conn, c)
    conn.commit()
    return len(constructors)


def ingest_races(conn: sqlite3.Connection, season: int) -> int:
    races = _paginate(f"{BASE_URL}/{season}/races.json")
    for race in races:
        _get_or_create_race(conn, season, race)
    conn.commit()
    return len(races)


def ingest_results(conn: sqlite3.Connection, season: int) -> int:
    """Fetch results per-race to avoid season-level pagination quirks."""
    races = conn.execute(
        "SELECT race_id, round FROM races WHERE season=?", (season,)
    ).fetchall()
    count = 0
    for race_id, rnd in races:
        race_objs = _paginate(f"{BASE_URL}/{season}/{rnd}/results.json")
        for race_obj in race_objs:
            for r in race_obj.get("Results", []):
                _upsert_driver(conn, r["Driver"])
                _upsert_constructor(conn, r["Constructor"])
                pos = r.get("position")
                conn.execute(
                    """
                    INSERT OR IGNORE INTO results
                        (race_id, driver_id, constructor_id, grid, position,
                         position_text, position_order, points, laps, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        race_id,
                        r["Driver"]["driverId"],
                        r["Constructor"]["constructorId"],
                        int(r["grid"]) if r.get("grid") else None,
                        int(pos) if pos and pos.isdigit() else None,
                        r.get("positionText"),
                        int(r["positionOrder"]) if r.get("positionOrder") else None,
                        float(r["points"]) if r.get("points") else None,
                        int(r["laps"]) if r.get("laps") else None,
                        r.get("status"),
                    ),
                )
                count += 1
    conn.commit()
    return count


def ingest_qualifying(conn: sqlite3.Connection, season: int) -> int:
    """Fetch qualifying per-race to avoid season-level pagination quirks."""
    races = conn.execute(
        "SELECT race_id, round FROM races WHERE season=?", (season,)
    ).fetchall()
    count = 0
    for race_id, rnd in races:
        race_objs = _paginate(f"{BASE_URL}/{season}/{rnd}/qualifying.json")
        for race_obj in race_objs:
            for q in race_obj.get("QualifyingResults", []):
                _upsert_driver(conn, q["Driver"])
                _upsert_constructor(conn, q["Constructor"])
                conn.execute(
                    """
                    INSERT OR IGNORE INTO qualifying
                        (race_id, driver_id, constructor_id, number, position, q1, q2, q3)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        race_id,
                        q["Driver"]["driverId"],
                        q["Constructor"]["constructorId"],
                        q.get("number"),
                        int(q["position"]) if q.get("position") else None,
                        q.get("Q1"),
                        q.get("Q2"),
                        q.get("Q3"),
                    ),
                )
                count += 1
    conn.commit()
    return count


def ingest_pitstops(conn: sqlite3.Connection, season: int) -> int:
    races = conn.execute(
        "SELECT race_id, round FROM races WHERE season=?", (season,)
    ).fetchall()
    count = 0
    for race_id, rnd in races:
        # _paginate returns list of race objects; pitstops live inside each race object
        race_objs = _paginate(f"{BASE_URL}/{season}/{rnd}/pitstops.json")
        for race_obj in race_objs:
            for s in race_obj.get("PitStops", []):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO pitstops
                        (race_id, driver_id, stop, lap, time_of_day, duration)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        race_id,
                        s["driverId"],
                        int(s["stop"]) if s.get("stop") else None,
                        int(s["lap"]) if s.get("lap") else None,
                        s.get("time"),
                        s.get("duration"),
                    ),
                )
                count += 1
    conn.commit()
    return count


def ingest_laptimes(conn: sqlite3.Connection, season: int) -> int:
    races = conn.execute(
        "SELECT race_id, round FROM races WHERE season=?", (season,)
    ).fetchall()
    count = 0
    for race_id, rnd in races:
        # _paginate returns list of race objects; laps live inside each race object
        race_objs = _paginate(f"{BASE_URL}/{season}/{rnd}/laps.json")
        for race_obj in race_objs:
            for lap in race_obj.get("Laps", []):
                lap_num = int(lap["number"])
                for timing in lap.get("Timings", []):
                    pos_raw = timing.get("position")
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO laptimes
                            (race_id, driver_id, lap, position, time)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            race_id,
                            timing["driverId"],
                            lap_num,
                            int(pos_raw) if pos_raw else None,
                            timing.get("time"),
                        ),
                    )
                    count += 1
    conn.commit()
    return count


# ---------------------------------------------------------------------------
# Idempotency tracking
# ---------------------------------------------------------------------------


def is_done(conn: sqlite3.Connection, season: int, dataset: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM _ingest_state WHERE season=? AND dataset=?",
        (season, dataset),
    ).fetchone()
    return row is not None


def mark_done(conn: sqlite3.Connection, season: int, dataset: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO _ingest_state (season, dataset) VALUES (?, ?)",
        (season, dataset),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

DATASETS = ["drivers", "constructors", "races", "results", "qualifying", "pitstops"]
DATASET_FNS: dict[str, Any] = {
    "drivers": ingest_drivers,
    "constructors": ingest_constructors,
    "races": ingest_races,
    "results": ingest_results,
    "qualifying": ingest_qualifying,
    "pitstops": ingest_pitstops,
    "laptimes": ingest_laptimes,
}


def run_etl(
    seasons: list[int],
    include_laptimes: bool = False,
    force: bool = False,
    db_path: Path = DB_PATH,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    create_schema(conn)

    active_datasets = DATASETS + (["laptimes"] if include_laptimes else [])

    for season in seasons:
        logger.info("Processing season %s", season)
        for dataset in active_datasets:
            if not force and is_done(conn, season, dataset):
                logger.info("  [skip] %s/%s already ingested", season, dataset)
                continue
            logger.info("  Fetching %s/%s …", season, dataset)
            fn = DATASET_FNS[dataset]
            n = fn(conn, season)
            mark_done(conn, season, dataset)
            logger.info("  Done  %s/%s → %d rows", season, dataset, n)

    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Jolpica → SQLite F1 stats ETL")
    parser.add_argument(
        "--season",
        type=int,
        metavar="YEAR",
        help="Ingest a single season (e.g. 2021). Defaults to all 2014-2025.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even seasons already marked done.",
    )
    parser.add_argument(
        "--laptimes",
        action="store_true",
        help="Also ingest per-lap times (high volume, slow).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Output SQLite path (default: {DB_PATH})",
    )
    args = parser.parse_args()

    seasons = [args.season] if args.season else SEASONS
    run_etl(
        seasons=seasons,
        include_laptimes=args.laptimes,
        force=args.force,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
