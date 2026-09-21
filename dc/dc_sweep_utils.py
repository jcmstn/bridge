#!/usr/bin/env python3
"""
Shared sweep/output-path helpers for the DC measurement programs
==========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-05

Small, pure-function utilities used by every DC measurement script so the
step-size/bidirectional sweep logic, the "single value or comma-separated
list" parsing, and the guarded-shutdown pattern are each implemented
exactly once.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


def build_segmented_sweep(
    rows: list[tuple[float, float, int]],
    bidirectional: bool,
    atol: float = 1e-9,
) -> np.ndarray:
    """
    Build a multi-row field/current sweep: each row is (start, stop,
    n_points), swept with np.linspace and executed back-to-back in order.

    Adjacent rows that land on the same value (row N's stop == row N+1's
    start, within `atol`) are merged -- the repeated setpoint is emitted
    once, not twice. Rows are free to jump instead; a jump is left alone.

    If `bidirectional`, the full merged forward path is retraced in
    reverse (reversing np.linspace(a, b, n) is np.linspace(b, a, n), so
    this is exactly "run every row backwards, in reverse order" without
    needing separate row-reversal logic), again merging the turn-around
    point via the same rule.
    """
    if not rows:
        raise ValueError("At least one sweep row is required.")

    def _join(chunks: list[np.ndarray]) -> np.ndarray:
        out = chunks[0]
        for chunk in chunks[1:]:
            if len(chunk) and len(out) and abs(chunk[0] - out[-1]) <= atol:
                chunk = chunk[1:]
            out = np.concatenate([out, chunk])
        return out

    forward = _join([np.linspace(start, stop, n) for start, stop, n in rows])
    if not bidirectional:
        return forward
    return _join([forward, forward[::-1]])


def linear_sweep(start: float, stop: float, step: float, bidirectional: bool = True) -> np.ndarray:
    """
    Build a linear sweep from `start` to `stop` using a step size (rather
    than a point count), optionally returning bidirectionally
    (start -> stop -> start) to reveal hysteresis.

    The turn-around point (`stop`) is not duplicated when bidirectional.
    Thin wrapper over build_segmented_sweep for the single-row, step-size
    call sites (dc_iv_curve, dc_gate_sweep, sot_pulsed_switching*).
    """
    if step <= 0:
        raise ValueError(f"Step size must be > 0, got {step!r}.")

    n = max(2, int(round(abs(stop - start) / step)) + 1)
    return build_segmented_sweep([(start, stop, n)], bidirectional)


def parse_sweep_rows(text: str) -> list[tuple[float, float, int]]:
    """
    Parse multi-row sweep text, one row per line: "start, stop, points".
    Blank lines are ignored. Raises ValueError (naming the offending line)
    on a malformed row, a non-positive/non-integer point count, or no
    rows at all.
    """
    rows: list[tuple[float, float, int]] = []
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Line {lineno} ({line!r}): expected 'start, stop, points'.")
        start_s, stop_s, n_s = parts
        try:
            start, stop = float(start_s), float(stop_s)
        except ValueError:
            raise ValueError(f"Line {lineno} ({line!r}): start/stop must be numbers.") from None
        try:
            n = int(n_s)
        except ValueError:
            raise ValueError(f"Line {lineno} ({line!r}): points must be an integer.") from None
        if n < 1:
            raise ValueError(f"Line {lineno} ({line!r}): points must be >= 1.")
        rows.append((start, stop, n))
    if not rows:
        raise ValueError("Expected at least one sweep row.")
    return rows


def parse_value_list(text: str) -> list[float]:
    """
    Parse a single value or comma-separated list of values, e.g.
    "1.5" -> [1.5], "1, 2.5, -3" -> [1.0, 2.5, -3.0].

    Raises ValueError (naming the offending token) on a bad entry, and on
    an empty/whitespace-only string.
    """
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if not tokens:
        raise ValueError("Expected at least one numeric value.")
    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            raise ValueError(f"'{token}' is not a valid number.") from None
    return values


def safe_shutdown(label: str, fn: Callable[[], None]) -> None:
    """
    Run one shutdown_*() cleanup step, logging (not raising) on failure so
    the remaining steps in the same `finally:` block still run.

    Every shutdown_*() helper in instruments/ (shutdown_magnet,
    shutdown_gate, shutdown_source, ...) is a plain VISA call with no
    internal try/except -- one instrument raising during cleanup must
    never skip the others (the magnet is an inductive load; the 6221 may
    still be sourcing current into the DUT).
    """
    try:
        fn()
    except Exception:
        log.exception("Error while shutting down %s", label)


def field_hops(currents_A, n_series: int) -> list[float]:
    """|ΔI| the magnet moves BEFORE each point of `n_series` back-to-back
    sweeps of `currents_A` (series-major, the loop order): the very first
    point starts from 0 A, and every later series' first point returns from
    the previous sweep's last current. Feeds the run-time estimate."""
    hops: list[float] = []
    prev = 0.0
    for _ in range(n_series):
        for current in currents_A:
            hops.append(abs(float(current) - prev))
            prev = float(current)
    return hops
