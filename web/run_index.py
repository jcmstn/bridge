"""
SQLite run index for bridge/web
===================================
Nothing like this exists anywhere in bridge today — every prior "settings"
file is just the last-used form values for one TUI, not a history of runs.
This is meant to become the entry point for a future run-picker in
bridge/analysis/ (out of scope here — only the index itself, and it being
populated by every run, is built now).

Lives at a FIXED location (_DATA_DIR / "runs.db", the same sibling-of-bridge
data/ directory every script's _DATA_DIR already points at by default) —
deliberately independent of any given run's user-chosen save directory
(bridge/web's new "choose a directory freely" feature), so the index always
lives somewhere predictable regardless of where individual runs' data
actually landed.

Each helper opens a short-lived connection, does its one statement, commits,
and closes — avoids sharing one sqlite3 connection across the worker thread
(which calls finish_run() from its own `finally:`) and the event-loop thread
(which calls update_point_count() from a ui.timer tick). WAL mode is set on
first connect so a write from one thread doesn't block a read from another.
"""

from __future__ import annotations

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


def _connect() -> sqlite3.Connection:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def start_run(suite: str, measurement: str, parameters: dict, data_dir: str,
              output_paths: list[str]) -> int:
    """
    Insert a 'running' row and return its id. Called before the first
    connect_*() call, so a connection failure is still captured as an
    'error' row rather than lost entirely.
    """
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (started_at, suite, measurement, status, "
            "parameters_json, data_dir, output_paths_json, point_count) "
            "VALUES (datetime('now'), ?, ?, 'running', ?, ?, ?, 0)",
            (suite, measurement, json.dumps(parameters), data_dir,
             json.dumps(output_paths)),
        )
        return int(cur.lastrowid)


def update_point_count(run_id: int, point_count: int) -> None:
    """Called from the ui.timer drain tick, once per tick (not per point)."""
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
