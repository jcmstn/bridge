#!/usr/bin/env python3
"""
Shared sweep/output-path helpers for the bridge/DC measurement programs
==========================================================================
Small, pure-function utilities used by every DC measurement script
(dc_hall_measurement.py, dc_iv_curve.py, dc_gate_sweep.py,
dc_spin_valve.py) so the step-size/bidirectional sweep logic and the
"single value or comma-separated list" parsing are each implemented
exactly once.

build_output_path() itself now lives in bridge/instruments/output_paths.py
(promoted so bridge/MFLI and bridge/web can share it too, since it's pure
pathlib logic with no DC-specific imports) and is re-exported below so
every existing `from dc_sweep_utils import build_output_path, ...` keeps
working unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_INSTRUMENTS_DIR = Path(__file__).resolve().parent.parent / "instruments"
if str(_INSTRUMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTRUMENTS_DIR))
from output_paths import build_output_path  # noqa: E402,F401  (re-exported — canonical copy lives in instruments/)


def linear_sweep(start: float, stop: float, step: float, bidirectional: bool = True) -> np.ndarray:
    """
    Build a linear sweep from `start` to `stop` using a step size (rather
    than a point count), optionally returning bidirectionally
    (start -> stop -> start) to reveal hysteresis.

    The turn-around point (`stop`) is not duplicated when bidirectional.
    """
    if step <= 0:
        raise ValueError(f"Step size must be > 0, got {step!r}.")

    n = max(2, int(round(abs(stop - start) / step)) + 1)
    up = np.linspace(start, stop, n)
    if not bidirectional:
        return up
    down = np.linspace(stop, start, n)[1:]
    return np.concatenate([up, down])


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
