#!/usr/bin/env python3
"""
bridge/web — NiceGUI entrypoint
====================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-08-07

Alternative front end to the Textual TUI (dc/dc_tui.py, mfli/mfli_tui.py),
covering the same 7 measurements plus one new capability the TUI doesn't
have: freely choosing the save directory for a run, anywhere on disk (see
directory_picker.py), rather than only a sub-folder name under a hardcoded
data/ root. Runs alongside the TUI, not instead of it.

Localhost-only ("localhost", not 127.0.0.1 — some network/VPN setups block
the loopback IP specifically while allowing the hostname), no
authentication — the same trust boundary the terminal-based TUI already has.

Run with:
    uv run python app.py     (from bridge/web/)

Port defaults to 8080; override with the BRIDGE_WEB_PORT env var if that
port is unavailable, e.g.:
    set BRIDGE_WEB_PORT=8090 && uv run python app.py     (Windows cmd)

On Windows, a bind failure at this port is usually Hyper-V/WSL2 port
reservation, not a conflicting process — see the README Troubleshooting
section.

`reload=False` is deliberate, not an oversight: NiceGUI's default file-watch
auto-reload would restart the server process — and with it, the run lock
and any live instrument connections — if a .py file changed mid-measurement.
"""

from __future__ import annotations

import os

from nicegui import ui

from web.dc import hall, iv_curve, gate_sweep, spin_valve
from web.mfli import dual_harmonic, diff_resistance, phase_calibration
from web import run_index
from web.run_controller import busy_banner

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
        {"name": "sample", "label": "Sample", "field": "sample"},
        {"name": "device", "label": "Device", "field": "device"},
        {"name": "run_number", "label": "Run #", "field": "run_number"},
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
            "sample": r.get("sample") or "—", "device": r.get("device") or "—",
            "run_number": r.get("run_number") or "—",
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
    _port = int(os.environ.get("BRIDGE_WEB_PORT", 8080))
    ui.run(host="localhost", port=_port, title=APP_TITLE, reload=False)
