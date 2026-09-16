"""
Tests for instruments/field_geometry.py
========================================
Pure math + string rendering — no hardware, no UI toolkit involved.
"""

from __future__ import annotations

from instruments.field_geometry import field_unit_vector, render_ascii_field_diagram


def _close(v: tuple[float, float, float], expected: tuple[float, float, float]) -> bool:
    return all(abs(a - b) < 1e-9 for a, b in zip(v, expected))


def test_field_unit_vector_out_of_plane() -> None:
    assert _close(field_unit_vector(0, 0), (0.0, 0.0, 1.0))
    assert _close(field_unit_vector(0, 123), (0.0, 0.0, 1.0))  # phi irrelevant at theta=0


def test_field_unit_vector_in_plane_along_x() -> None:
    assert _close(field_unit_vector(90, 0), (1.0, 0.0, 0.0))


def test_field_unit_vector_in_plane_along_y() -> None:
    assert _close(field_unit_vector(90, 90), (0.0, 1.0, 0.0))


def test_field_unit_vector_is_unit_length() -> None:
    import math

    for theta, phi in [(30, 10), (60, 200), (90, 359), (15, 0)]:
        x, y, z = field_unit_vector(theta, phi)
        assert abs(math.sqrt(x * x + y * y + z * z) - 1.0) < 1e-9


def test_render_ascii_field_diagram_stable_size() -> None:
    cases = [(0, 0), (90, 0), (90, 90), (45, 30), (None, None), (10, None)]
    diagrams = [render_ascii_field_diagram(theta, phi) for theta, phi in cases]
    sizes = {(len(d.split("\n")), max(len(line) for line in d.split("\n"))) for d in diagrams}
    assert len(sizes) == 1  # every case renders to the same fixed panel size


def test_render_ascii_field_diagram_varies_with_input() -> None:
    diagrams = {
        render_ascii_field_diagram(theta, phi)
        for theta, phi in [(0, 0), (90, 0), (90, 90), (None, None)]
    }
    assert len(diagrams) == 4  # all visually distinct
