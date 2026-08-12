#!/usr/bin/env python3
"""
Shared sweep/output-path helpers for the DC measurement programs
==========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-05

Small, pure-function utilities used by every DC measurement script so the
step-size/bidirectional sweep logic and the "single value or
comma-separated list" parsing are each implemented exactly once.
"""

from __future__ import annotations

import numpy as np


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
