"""Tests for dc/dc_sweep_utils.py: build_segmented_sweep, parse_sweep_rows,
and the linear_sweep/bidirectional_current_sweep wrappers built on top."""

from __future__ import annotations

import numpy as np
import pytest

from dc.dc_sweep_utils import build_segmented_sweep, linear_sweep, parse_sweep_rows
from mfli.mfli_dual_harmonic import bidirectional_current_sweep


def test_two_row_bidirectional_worked_example() -> None:
    # User's own worked example: rows (-1, 1, 10) and (1, 10, 10), sharing
    # a boundary at 1 -> merged forward path is 19 points, not 20; the
    # bidirectional reverse leg retraces it and drops the duplicated
    # turn-around point -> 37 points total.
    rows = [(-1.0, 1.0, 10), (1.0, 10.0, 10)]
    result = build_segmented_sweep(rows, bidirectional=True)
    assert len(result) == 37

    forward = build_segmented_sweep(rows, bidirectional=False)
    assert len(forward) == 19
    np.testing.assert_allclose(result[:19], forward)
    np.testing.assert_allclose(result[19:], forward[::-1][1:])


def test_single_row_matches_linear_sweep() -> None:
    old = linear_sweep(start=-20.0, stop=20.0, step=2.0, bidirectional=True)
    n = max(2, round(abs(20.0 - -20.0) / 2.0) + 1)
    new = build_segmented_sweep([(-20.0, 20.0, n)], bidirectional=True)
    np.testing.assert_allclose(old, new)


def test_single_row_matches_bidirectional_current_sweep() -> None:
    old = bidirectional_current_sweep(-20.0, 20.0, 21)
    new = build_segmented_sweep([(-20.0, 20.0, 21)], bidirectional=True)
    np.testing.assert_allclose(old, new)


def test_deliberate_jump_is_not_deduped() -> None:
    rows = [(0.0, 1.0, 3), (5.0, 6.0, 3)]
    result = build_segmented_sweep(rows, bidirectional=False)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0, 5.0, 5.5, 6.0])


def test_repeated_point_within_one_row_is_kept() -> None:
    result = build_segmented_sweep([(1.0, 1.0, 3)], bidirectional=False)
    np.testing.assert_allclose(result, [1.0, 1.0, 1.0])


def test_rejects_empty_rows() -> None:
    with pytest.raises(ValueError):
        build_segmented_sweep([], bidirectional=False)


def test_parse_sweep_rows_basic() -> None:
    text = "-0.001, 0.001, 21\n0.001, 0.01, 10\n\n0.01, 0.05, 5"
    assert parse_sweep_rows(text) == [
        (-0.001, 0.001, 21),
        (0.001, 0.01, 10),
        (0.01, 0.05, 5),
    ]


def test_parse_sweep_rows_rejects_bad_point_count() -> None:
    with pytest.raises(ValueError, match="Line 1"):
        parse_sweep_rows("0, 1, 0")


def test_parse_sweep_rows_rejects_non_integer_points() -> None:
    with pytest.raises(ValueError, match="Line 1"):
        parse_sweep_rows("0, 1, 2.5")


def test_parse_sweep_rows_rejects_wrong_field_count() -> None:
    with pytest.raises(ValueError, match="Line 1"):
        parse_sweep_rows("0, 1")


def test_parse_sweep_rows_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        parse_sweep_rows("   \n  \n")
