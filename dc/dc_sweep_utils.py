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
import math
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

# Upper bound on one sweep DIRECTION's point count (a bidirectional sweep may
# have twice this). Every sweep a form previews on each keystroke goes through
# parse_sweep_rows() / build_segmented_sweep() / linear_sweep(), so a mistyped
# step (1e-12) or span (1e9) used to allocate billions of points and freeze --
# or OOM-kill -- the TUI/web form. 100 000 points is ~28 h at 1 s/point, far
# past any real sweep here; raise it if a real sweep ever needs more.
MAX_SWEEP_POINTS = 100_000


def check_sweep_size(n_points: int) -> None:
    """ValueError (the form shows it as a blocking error) if one sweep
    direction would have more than MAX_SWEEP_POINTS points -- checked BEFORE
    anything is allocated."""
    if n_points > MAX_SWEEP_POINTS:
        raise ValueError(
            f"Sweep would have {n_points:,} points per direction, more than the "
            f"{MAX_SWEEP_POINTS:,} limit — use a larger step or fewer points.")


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
    check_sweep_size(sum(n for _, _, n in rows))

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
    n = _one_way_points(start, stop, step)
    return build_segmented_sweep([(start, stop, n)], bidirectional)


def _one_way_points(start: float, stop: float, step: float) -> int:
    if not all(math.isfinite(v) for v in (start, stop, step)):
        raise ValueError("Sweep start, stop and step must be finite numbers.")
    if step <= 0:
        raise ValueError(f"Step size must be > 0, got {step!r}.")
    n = max(2, int(round(abs(stop - start) / step)) + 1)
    check_sweep_size(n)
    return n


def sweep_point_count(start: float, stop: float, step: float, bidirectional: bool = True) -> int:
    """How many points linear_sweep(start, stop, step, bidirectional) yields,
    without building it. Same ValueErrors as linear_sweep(), including the
    MAX_SWEEP_POINTS cap -- for summaries that only need the count."""
    n = _one_way_points(start, stop, step)
    return 2 * n - 1 if bidirectional else n


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
    check_sweep_size(sum(n for _, _, n in rows))
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
            value = float(token)
        except ValueError:
            raise ValueError(f"'{token}' is not a valid number.") from None
        if not math.isfinite(value):
            raise ValueError(f"'{token}' is not a finite number.")
        values.append(value)
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
