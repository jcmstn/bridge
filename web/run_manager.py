"""
Global run-lock singleton for bridge/web
============================================
Only one measurement may run at a time across the ENTIRE app — not just
per-page, and not just per-suite. This is not merely "simpler than a job
queue": the Kepco BOP-GL magnet, Lake Shore 475 gaussmeter, and MercuryiTC
temperature controller are the SAME physical instruments shared between the
DC and MFLI suites' field-sweep measurements, so letting a DC run and an
MFLI run start concurrently would mean two threads issuing GPIB commands to
the same instrument at once.

Every page acquires this lock before connecting to any instrument and
releases it from the worker thread's own `finally:` block — the same
`finally:` that already runs instrument shutdown in every *_tui.py — so the
lock is always released even if a run errors out or every browser tab
watching it is closed.

`RunHandle` also buffers the run's own live records/log lines, so any page
(including a fresh page load, or a different tab entirely) that finds a run
already in progress can repaint the live view from this buffer rather than
missing everything that happened before it looked — and can abort a run
started from a different tab. That's a real hardware-safety property the
single-process TUI never had to solve (only one terminal can be open).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class RunHandle:
    suite: str            # "DC" | "MFLI"
    measurement: str      # human-readable, e.g. "DC Hall Measurement"
    started_at: datetime
    run_id: int = -1       # set once run_index.start_run() returns a row id
    stop_event: threading.Event = field(default_factory=threading.Event)
    records: list = field(default_factory=list)
    log_lines: list = field(default_factory=list)
    status_text: str = "Starting …"

    @property
    def elapsed_s(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()


_lock = threading.Lock()
_current: Optional[RunHandle] = None


def try_acquire(suite: str, measurement: str) -> Optional[RunHandle]:
    """
    Atomically claim the global run lock, or return None if another
    measurement (DC or MFLI) is already running.
    """
    global _current
    with _lock:
        if _current is not None:
            return None
        _current = RunHandle(suite=suite, measurement=measurement, started_at=datetime.now())
        return _current


def release(handle: RunHandle) -> None:
    """Release the lock — only the handle that currently holds it may do so."""
    global _current
    with _lock:
        if _current is handle:
            _current = None


def snapshot() -> Optional[RunHandle]:
    """
    Return the currently-running handle (same object reference), or None.
    Callers read its fields for display purposes; `records`/`log_lines` are
    append-only lists mutated only by the owning worker thread, which is
    safe enough under the GIL for read-only display use from other threads.
    """
    with _lock:
        return _current
