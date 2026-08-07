#!/usr/bin/env python3
"""
bridge/web — NiceGUI entrypoint
====================================
Alternative front end to the Textual TUI (bridge/DC/dc_tui.py,
bridge/MFLI/mfli_tui.py), covering the same 7 measurements plus one new
capability the TUI doesn't have: freely choosing the save directory for a
run, anywhere on disk (see directory_picker.py), rather than only a
sub-folder name under a hardcoded data/ root.

Runs alongside the TUI, not instead of it — both remain ongoing entry
points into the same bridge/DC, bridge/MFLI and bridge/instruments code.
Localhost-only (127.0.0.1), no authentication — the same trust boundary
the terminal-based TUI already has.

Run with:
    uv run python app.py     (from bridge/web/)

`reload=False` is deliberate, not an oversight: NiceGUI's default file-watch
auto-reload would restart the server process — and with it, the run lock
and any live instrument connections — if a .py file changed mid-measurement.
"""

from __future__ import annotations

import sys
from pathlib import Path

from nicegui import ui

_WEB_DIR = Path(__file__).resolve().parent
_DC_DIR = _WEB_DIR / "dc"
_MFLI_DIR = _WEB_DIR / "mfli"
for _p in (_WEB_DIR, _DC_DIR, _MFLI_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import hall               # noqa: E402
import iv_curve            # noqa: E402
import gate_sweep          # noqa: E402
import spin_valve          # noqa: E402
import dual_harmonic       # noqa: E402
import diff_resistance     # noqa: E402
import phase_calibration   # noqa: E402

import run_index  # noqa: E402
from run_controller import busy_banner  # noqa: E402

APP_TITLE = "Bridge Measurement Suite"


@ui.page("/dc/hall")
def _dc_hall_page() -> None:
    hall.page()


@ui.page("/dc/iv-curve")
def _dc_iv_curve_page() -> None:
    iv_curve.page()


@ui.page("/dc/gate-sweep")
def _dc_gate_sweep_page() -> None:
    gate_sweep.page()


@ui.page("/dc/spin-valve")
def _dc_spin_valve_page() -> None:
    spin_valve.page()


@ui.page("/mfli/dual-harmonic")
def _mfli_dual_harmonic_page() -> None:
    dual_harmonic.page()


@ui.page("/mfli/diff-resistance")
def _mfli_diff_resistance_page() -> None:
    diff_resistance.page()


@ui.page("/mfli/phase-calibration")
def _mfli_phase_calibration_page() -> None:
    phase_calibration.page()


def _card(title: str, description: str, route: str) -> None:
    with ui.card().classes("w-full"):
        ui.label(title).classes("text-lg font-bold")
        ui.label(description).classes("text-sm text-grey-7")
        ui.button("Open", on_click=lambda: ui.navigate.to(route)).props("color=primary").classes("mt-1")


def _recent_runs_table() -> None:
    columns = [
        {"name": "started_at", "label": "Started", "field": "started_at", "sortable": True},
        {"name": "suite", "label": "Suite", "field": "suite"},
        {"name": "measurement", "label": "Measurement", "field": "measurement"},
        {"name": "status", "label": "Status", "field": "status"},
        {"name": "point_count", "label": "Points", "field": "point_count"},
        {"name": "duration_s", "label": "Duration (s)", "field": "duration_s"},
        {"name": "data_dir", "label": "Data directory", "field": "data_dir"},
    ]

    @ui.refreshable
    def _table() -> None:
        runs = run_index.recent_runs(limit=20)
        if not runs:
            ui.label("No runs recorded yet — this fills in as you run measurements.") \
                .classes("text-grey-6")
            return
        rows = [{
            "started_at": r["started_at"], "suite": r["suite"], "measurement": r["measurement"],
            "status": r["status"], "point_count": r["point_count"],
            "duration_s": f"{r['duration_s']:.1f}" if r["duration_s"] is not None else "—",
            "data_dir": r["data_dir"],
        } for r in runs]
        ui.table(columns=columns, rows=rows, row_key="started_at").classes("w-full").props("dense")

    _table()
    ui.timer(5.0, _table.refresh)


@ui.page("/")
def landing() -> None:
    ui.page_title(APP_TITLE)
    busy_banner()
    ui.label(APP_TITLE).classes("text-3xl font-bold")
    ui.label(
        "Web front end for the DC and MFLI measurement programs — runs alongside "
        "the Textual TUI (dc_tui.py / mfli_tui.py), not instead of it."
    ).classes("text-grey-7 mb-4")

    with ui.row().classes("w-full gap-4 items-start"):
        with ui.column().classes("flex-1 gap-3"):
            ui.label("DC Suite").classes("text-xl font-bold")
            _card("DC Hall Measurement", hall.DC_HALL_DESCRIPTION, "/dc/hall")
            _card("DC I-V Curve", iv_curve.DC_IV_DESCRIPTION, "/dc/iv-curve")
            _card("DC Gate Sweep", gate_sweep.DC_GATE_SWEEP_DESCRIPTION, "/dc/gate-sweep")
            _card("DC Spin-Valve / Field Sweep", spin_valve.DC_SPIN_VALVE_DESCRIPTION, "/dc/spin-valve")
        with ui.column().classes("flex-1 gap-3"):
            ui.label("MFLI Suite").classes("text-xl font-bold")
            _card("MFLI Dual-Harmonic Measurement",
                  dual_harmonic.MFLI_DUAL_HARMONIC_DESCRIPTION, "/mfli/dual-harmonic")
            _card("MFLI Differential Resistance vs. Bias",
                  diff_resistance.MFLI_DIFF_RESISTANCE_DESCRIPTION, "/mfli/diff-resistance")
            _card("MFLI Phase Calibration",
                  phase_calibration.MFLI_PHASE_CALIBRATION_DESCRIPTION, "/mfli/phase-calibration")

    ui.separator().classes("my-4")
    ui.label("Recent runs").classes("text-xl font-bold mb-2")
    _recent_runs_table()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="127.0.0.1", port=8080, title=APP_TITLE, reload=False)
