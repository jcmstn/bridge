"""
Live field-direction 3D diagram (web only)
==============================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-16

Plotly figure builder for the same field-direction panel every *_tui.py
draws as ASCII (instruments/field_geometry.py's render_ascii_field_diagram).
Web-only because it needs plotly — the vector math itself
(field_unit_vector) is shared, imported from there.

nicegui[plotly] is already a dependency and its bundled plotly.js includes
3D traces, so this needs nothing new. Callers must set
`fig.update_layout(uirevision=...)` to persist (already done here) — without
it every `ui.plotly(...).update()` call resets the camera and a user who
rotated the view loses it on the next keystroke.
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go

from instruments.field_geometry import field_unit_vector

_AXIS_LEN = 1.3
_VECTOR_LEN = 1.3


def build_field_diagram_figure(theta_deg: Optional[float], phi_deg: Optional[float]) -> go.Figure:
    """Translucent film-plane square + x(current)/y/z(normal) axes, plus the
    field direction as a line+cone arrow when theta is set."""
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=[-1, 1, 1, -1], y=[-1, -1, 1, 1], z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=0.15, color="steelblue", showscale=False,
        hoverinfo="skip", name="film",
    ))

    for (ex, ey, ez), label in (
        ((_AXIS_LEN, 0, 0), "x (I)"), ((0, _AXIS_LEN, 0), "y"), ((0, 0, _AXIS_LEN), "z (normal)"),
    ):
        fig.add_trace(go.Scatter3d(
            x=[0, ex], y=[0, ey], z=[0, ez], mode="lines+text",
            line=dict(color="grey", width=3), text=["", label], textposition="top center",
            hoverinfo="skip", showlegend=False,
        ))

    if theta_deg is not None:
        fx, fy, fz = field_unit_vector(theta_deg, phi_deg if phi_deg is not None else 0.0)
        fig.add_trace(go.Scatter3d(
            x=[0, fx * _VECTOR_LEN], y=[0, fy * _VECTOR_LEN], z=[0, fz * _VECTOR_LEN],
            mode="lines", line=dict(color="crimson", width=7),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Cone(
            x=[fx * _VECTOR_LEN], y=[fy * _VECTOR_LEN], z=[fz * _VECTOR_LEN],
            u=[fx * 0.001], v=[fy * 0.001], w=[fz * 0.001],  # near-zero: just orients the arrowhead
            anchor="tip", showscale=False, sizemode="absolute", sizeref=0.35,
            colorscale=[[0, "crimson"], [1, "crimson"]], hoverinfo="skip",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False, range=[-_AXIS_LEN, _AXIS_LEN]),
            yaxis=dict(visible=False, range=[-_AXIS_LEN, _AXIS_LEN]),
            zaxis=dict(visible=False, range=[-_AXIS_LEN, _AXIS_LEN]),
            aspectmode="cube",
        ),
        uirevision="field-diagram",
        showlegend=False,
    )
    return fig


def demo() -> None:
    fig = build_field_diagram_figure(45, 30)
    assert fig.layout.uirevision == "field-diagram"
    assert len(fig.data) == 6  # plane + 3 axes + vector line + cone
    fig_blank = build_field_diagram_figure(None, None)
    assert len(fig_blank.data) == 4  # plane + 3 axes only, no vector
    print("ok")


if __name__ == "__main__":
    demo()
