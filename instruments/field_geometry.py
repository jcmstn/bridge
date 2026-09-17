"""
Field-direction geometry: phi/theta <-> unit vector, plus a small ASCII
diagram of the field relative to the film
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-16

Coordinate convention (see docs/data_convention.md "Field direction
convention" for the full writeup): z = film normal (out-of-plane), x =
current/channel direction (in-plane), y = transverse in-plane, completing a
right-handed frame.

    theta_deg  polar angle from +z.   0 deg = out-of-plane, 90 deg = in-plane.
    phi_deg    azimuth from +x, in the xy-plane, 0-360 deg.
                Meaningless when theta==0 (any phi is the same OOP point).

Pure module -- no Textual/nicegui/plotly import -- reused verbatim by every
`*_tui.py`'s Static diagram and by `web/field_diagram.py`'s plotly figure
(which imports `field_unit_vector` for its own 3D trace).

    >>> field_unit_vector(0, 0)
    (0.0, 0.0, 1.0)
    >>> x, y, z = field_unit_vector(90, 0)
    >>> round(x, 3), round(y, 3), round(z, 3)
    (1.0, 0.0, 0.0)
"""

from __future__ import annotations

import math
from typing import Optional

__all__ = ["field_unit_vector", "render_ascii_field_diagram", "field_direction_summary_line"]


def field_unit_vector(theta_deg: float, phi_deg: float) -> tuple[float, float, float]:
    """Spherical (theta from +z, phi from +x in the xy-plane) -> Cartesian
    unit vector (x, y, z)."""
    t = math.radians(theta_deg)
    p = math.radians(phi_deg)
    return (math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t))


# ── ASCII isometric diagram ─────────────────────────────────────────────────
# One fixed iso transform (classic "floor tile" projection): the film's xy
# extent draws as a diamond, +z draws straight up. Terminal cells are ~2x
# taller than wide, so the column scale is larger than the row scale to keep
# the diamond looking roughly regular instead of squashed.

_COS30 = math.cos(math.radians(30))
_SIN30 = math.sin(math.radians(30))
_WIDTH, _HEIGHT = 23, 11
_ORIGIN = (_WIDTH // 2, _HEIGHT // 2)
_COL_SCALE, _ROW_SCALE = 5.2, 3.6


def _iso(x: float, y: float, z: float) -> tuple[int, int]:
    sx = (x - y) * _COS30
    sy = (x + y) * _SIN30 - z
    col = _ORIGIN[0] + round(sx * _COL_SCALE)
    row = _ORIGIN[1] + round(sy * _ROW_SCALE)
    return col, row


def _line(p0: tuple[int, int], p1: tuple[int, int]):
    """Bresenham, grid points from p0 to p1 inclusive."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def render_ascii_field_diagram(theta_deg: Optional[float], phi_deg: Optional[float]) -> str:
    """A small isometric diagram: the film as a diamond, its normal as a
    vertical line, its x (current) and y axes as labelled ticks from the
    origin, and (if theta is set) the field direction as an arrow -- one
    panel, not separate top-down/side views, so reading it needs no mental
    composition. phi defaults to 0 deg for the drawing when left blank
    (theta alone still pins a real direction whenever theta is 0 or 90 --
    only a genuinely tilted, phi-unset field is ambiguous, and this picks
    the current-axis convention rather than drawing nothing)."""
    grid = [[" "] * _WIDTH for _ in range(_HEIGHT)]

    def put(col: int, row: int, ch: str) -> None:
        if 0 <= row < _HEIGHT and 0 <= col < _WIDTH:
            grid[row][col] = ch

    def draw_line(p0, p1, ch: str) -> None:
        for col, row in _line(p0, p1):
            put(col, row, ch)

    # Film plane, drawn as a diamond (corners at (+-1, +-1, 0)).
    corners = [_iso(1, 1, 0), _iso(1, -1, 0), _iso(-1, -1, 0), _iso(-1, 1, 0)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        draw_line(a, b, "·")  # ·

    origin = _iso(0, 0, 0)
    put(*origin, "+")

    # Film normal (+z), always drawn.
    normal_tip = _iso(0, 0, 1.3)
    draw_line(origin, normal_tip, "│")  # │
    put(*normal_tip, "z")

    # In-plane x (current) / y axes, so phi's reference direction is visible.
    x_tip = _iso(1.3, 0, 0)
    draw_line(origin, x_tip, "-")
    put(*x_tip, "x")

    y_tip = _iso(0, 1.3, 0)
    draw_line(origin, y_tip, "-")
    put(*y_tip, "y")

    if theta_deg is not None:
        fx, fy, fz = field_unit_vector(theta_deg, phi_deg if phi_deg is not None else 0.0)
        tip = _iso(fx * 1.3, fy * 1.3, fz * 1.3)
        draw_line(origin, tip, "*")
        put(*tip, "◆")  # ◆ arrowhead

    return "\n".join("".join(row) for row in grid)


def field_direction_summary_line(theta_deg: Optional[float], phi_deg: Optional[float]) -> str:
    """One-line, human-readable field direction, for both front ends' live
    summary panel. Shared here so the wording doesn't drift across the six
    measurement modules that carry these fields."""
    if theta_deg is None:
        return "Field direction unset — field_theta_deg/field_phi_deg columns left blank."
    phi_txt = f"{phi_deg:g}°" if phi_deg is not None else "unset"
    if theta_deg == 0:
        return "Field fully out-of-plane (θ=0°) — φ irrelevant."
    if theta_deg == 90:
        return f"Field in-plane (θ=90°), φ={phi_txt} from current axis."
    return f"Field tilted {theta_deg:g}° from out-of-plane, φ={phi_txt}."


def demo() -> None:
    assert field_unit_vector(0, 0) == (0.0, 0.0, 1.0)
    x, y, z = field_unit_vector(90, 0)
    assert abs(x - 1.0) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9
    x, y, z = field_unit_vector(90, 90)
    assert abs(x) < 1e-9 and abs(y - 1.0) < 1e-9 and abs(z) < 1e-9

    diagram = render_ascii_field_diagram(45, 30)
    lines = diagram.split("\n")
    assert len(lines) == _HEIGHT and all(len(line) == _WIDTH for line in lines)
    assert render_ascii_field_diagram(None, None) != diagram  # blank-theta variant differs
    print(diagram)
    print("ok")


if __name__ == "__main__":
    demo()
