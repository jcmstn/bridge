"""
SQLite run index — shared by the web and TUI front ends
=========================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07 (moved from web/ 2026-09-23 so the TUIs can record too)

Records every run (start, finish, status, parameters, output paths) from
either front end, to drive the run-history tables on the web landing page
and the bridge_tui.py menu. Pure sqlite — no NiceGUI/Textual import, so it
sits in instruments/ below both front ends.

Lives at a fixed location (_DATA_DIR / "runs.db", the sibling-of-bridge
data/ directory) — deliberately independent of any given run's user-chosen
save directory, so the index always lives somewhere predictable regardless
of where individual runs' data actually landed.

Each helper opens a short-lived connection, does its one statement, commits,
and closes — avoids sharing one sqlite3 connection across the worker thread
(which calls finish_run() from its own `finally:`) and the event-loop thread
(which calls update_point_count() from a ui.timer tick). WAL mode, the
schema and the column migrations are applied once per process per database
file, on first connect, so a write from one thread doesn't block a read
from another.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DATA_DIR / "runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    suite              TEXT NOT NULL,
    measurement        TEXT NOT NULL,
    status             TEXT NOT NULL,
    parameters_json    TEXT NOT NULL,
    data_dir           TEXT NOT NULL,
    output_paths_json  TEXT NOT NULL,
    point_count        INTEGER NOT NULL DEFAULT 0,
    duration_s         REAL,
    error_message      TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_suite      ON runs(suite);
"""


# Added when bridge moved to the per-sample data convention (see
# instruments/data_naming.py) -- nullable, so old rows just show blank.
_MIGRATION_COLUMNS = [
    ("sample", "TEXT"),
    ("device", "TEXT"),
    ("run_number", "TEXT"),
]


_initialized: set[Path] = set()     # database files already set up this process


@contextlib.contextmanager
def _connect():
    """One short-lived connection: commits on success, rolls back on error,
    and is always closed (sqlite3's own `with conn:` only commits)."""
    first = _DB_PATH not in _initialized
    if first:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10.0)
    try:
        if first:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            for name, sql_type in _MIGRATION_COLUMNS:
                try:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {sql_type}")
                except sqlite3.OperationalError:
                    pass  # already migrated -- idempotent, same style as CREATE TABLE IF NOT EXISTS
            _initialized.add(_DB_PATH)
        with conn:
            yield conn
    finally:
        conn.close()


def start_run(suite: str, measurement: str, parameters: dict, data_dir: str,
              output_paths: list[str], *,
              sample: Optional[str] = None, device: Optional[str] = None,
              run_number: Optional[int] = None) -> int:
    """
    Insert a 'running' row and return its id. Called before the first
    connect_*() call, so a connection failure is still captured as an
    'error' row rather than lost entirely.

    `sample`/`device`/`run_number` are the data-convention identity for
    this run (see instruments/data_naming.py's allocate_run()) -- optional
    only because this module predates that convention; every migrated page
    passes them.
    """
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (started_at, suite, measurement, status, "
            "parameters_json, data_dir, output_paths_json, point_count, "
            "sample, device, run_number) "
            "VALUES (datetime('now'), ?, ?, 'running', ?, ?, ?, 0, ?, ?, ?)",
            (suite, measurement, json.dumps(parameters), data_dir,
             json.dumps(output_paths), sample, device,
             None if run_number is None else str(run_number)),
        )
        return int(cur.lastrowid)


def update_point_count(run_id: int, point_count: int) -> None:
    """Live point count of a running run — called once per UI tick that
    brought new points (web drain tick / TUI point callback), not per point."""
    if run_id < 0:
        return
    with _connect() as conn:
        conn.execute("UPDATE runs SET point_count = ? WHERE id = ?", (point_count, run_id))


def finish_run(run_id: int, *, status: str, point_count: int, duration_s: float,
               error_message: Optional[str] = None,
               output_paths: Optional[list[str]] = None) -> None:
    """
    Called from inside the worker thread's own `finally:` block, right after
    the final PNG save and right before run_manager.release(). `status` is
    one of 'completed' | 'aborted' | 'error', derived the same way do_run()
    already derives its final status string in every *_tui.py.
    """
    if run_id < 0:
        return
    with _connect() as conn:
        if output_paths is not None:
            conn.execute(
                "UPDATE runs SET finished_at = datetime('now'), status = ?, "
                "point_count = ?, duration_s = ?, error_message = ?, output_paths_json = ? "
                "WHERE id = ?",
                (status, point_count, duration_s, error_message, json.dumps(output_paths), run_id),
            )
        else:
            conn.execute(
                "UPDATE runs SET finished_at = datetime('now'), status = ?, "
                "point_count = ?, duration_s = ?, error_message = ? WHERE id = ?",
                (status, point_count, duration_s, error_message, run_id),
            )


def recent_runs(limit: int = 50) -> list[dict]:
    """Most recent runs first — used by the landing page's run-history table."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
