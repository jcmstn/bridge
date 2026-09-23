#!/usr/bin/env python3
"""
Bridge Measurement Suite  ── one Textual menu for every measurement program
============================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-23 (replaces the per-suite pickers dc/dc_tui.py,
mfli/mfli_tui.py and sot/sot_tui.py)

Terminal twin of the web landing page (web/app.py): the DC / MFLI / SOT
programs as cards in three columns — each with its description, a
collapsible wiring schematic and a Launch button — above the shared
"Recent runs" history (instruments/run_index.py, written by both front
ends). Launching a program exits this menu, runs that program's own App,
and comes back here when it quits.

Run with:
    uv run python bridge_tui.py

Adding a program: append a Program(...) to its suite in PROGRAMS — nothing
else registers it. Its description and schematic live next to its
MEASUREMENT_TYPE in its own *_tui.py (single source of truth).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, DataTable, Footer, Header, Static

from dc.dc_gate_sweep_tui import DC_GATE_SWEEP_DESCRIPTION, DC_GATE_SWEEP_SCHEMATIC, DCGateSweepApp
from dc.dc_hall_measurement_tui import DC_HALL_DESCRIPTION, DC_HALL_SCHEMATIC, DCHallMeasurementApp
from dc.dc_iv_curve_tui import DC_IV_DESCRIPTION, DC_IV_SCHEMATIC, DCIVCurveApp
from dc.dc_spin_valve_tui import DC_SPIN_VALVE_DESCRIPTION, DC_SPIN_VALVE_SCHEMATIC, DCSpinValveApp
from instruments import run_index
from mfli.mfli_diff_resistance_tui import (
    MFLI_DIFF_RESISTANCE_DESCRIPTION, MFLI_DIFF_RESISTANCE_SCHEMATIC, MFLIDiffResistanceApp)
from mfli.mfli_dual_harmonic_tui import (
    MFLI_DUAL_HARMONIC_DESCRIPTION, MFLI_DUAL_HARMONIC_SCHEMATIC, MFLIDualHarmonicApp)
from mfli.mfli_noise_spectrum_tui import (
    MFLI_NOISE_SPECTRUM_DESCRIPTION, MFLI_NOISE_SPECTRUM_SCHEMATIC, MFLINoiseSpectrumApp)
from mfli.mfli_phase_calibration_tui import (
    MFLI_PHASE_CALIBRATION_DESCRIPTION, MFLI_PHASE_CALIBRATION_SCHEMATIC, MFLIPhaseCalibrationApp)
from sot.sot_nonlocal_switching_tui import NLSW_DESCRIPTION, NLSW_SCHEMATIC, NonlocalSwitchingApp
from sot.sot_pulsed_switching_tui import SOT_PULSED_DESCRIPTION, SOT_PULSED_SCHEMATIC, SOTPulsedSwitchingApp

log = logging.getLogger("bridge_tui")


@dataclass(frozen=True)
class Program:
    key: str            # launch-button id suffix + LauncherApp's exit result
    title: str
    description: str
    schematic: str
    app: type[App]      # constructed with no arguments


PROGRAMS: dict[str, list[Program]] = {
    "DC Suite": [
        Program("hall", "Hall Measurement (DC, field sweep)",
                DC_HALL_DESCRIPTION, DC_HALL_SCHEMATIC, DCHallMeasurementApp),
        Program("iv", "I-V Curve (DC current sweep, optional gate)",
                DC_IV_DESCRIPTION, DC_IV_SCHEMATIC, DCIVCurveApp),
        Program("gate_sweep", "Gate Sweep (gate voltage sweep, optional field)",
                DC_GATE_SWEEP_DESCRIPTION, DC_GATE_SWEEP_SCHEMATIC, DCGateSweepApp),
        Program("spin_valve", "Spin-Valve / Field Sweep (fixed gate, field sweep)",
                DC_SPIN_VALVE_DESCRIPTION, DC_SPIN_VALVE_SCHEMATIC, DCSpinValveApp),
    ],
    "MFLI Suite": [
        Program("dual", "Dual-Harmonic Measurement (1f / 2f; MFLI or 6221 source)",
                MFLI_DUAL_HARMONIC_DESCRIPTION, MFLI_DUAL_HARMONIC_SCHEMATIC, MFLIDualHarmonicApp),
        Program("diff", "Differential Resistance vs. Bias (dV/dI)",
                MFLI_DIFF_RESISTANCE_DESCRIPTION, MFLI_DIFF_RESISTANCE_SCHEMATIC, MFLIDiffResistanceApp),
        Program("phase_cal", "Phase Calibration (1f Y-null + 2f channel ID)",
                MFLI_PHASE_CALIBRATION_DESCRIPTION, MFLI_PHASE_CALIBRATION_SCHEMATIC,
                MFLIPhaseCalibrationApp),
        Program("noise", "Noise Floor Estimate (2 MFLI + 6221)",
                MFLI_NOISE_SPECTRUM_DESCRIPTION, MFLI_NOISE_SPECTRUM_SCHEMATIC, MFLINoiseSpectrumApp),
    ],
    "SOT Suite": [
        Program("pulsed", "SOT pulsed switching (4200A or 6221 pulse · DC or lock-in read)",
                SOT_PULSED_DESCRIPTION, SOT_PULSED_SCHEMATIC, SOTPulsedSwitchingApp),
        Program("nonlocal", "Nonlocal spin-current switching (6221 pulse + 2182A nonlocal read)",
                NLSW_DESCRIPTION, NLSW_SCHEMATIC, NonlocalSwitchingApp),
    ],
}

_BY_KEY: dict[str, Program] = {p.key: p for programs in PROGRAMS.values() for p in programs}

# Same columns as the web landing page's run-history table.
RUN_COLUMNS = [
    ("started_at", "Started"), ("sample", "Sample"), ("device", "Device"),
    ("run_number", "Run #"), ("suite", "Suite"), ("measurement", "Measurement"),
    ("status", "Status"), ("point_count", "Points"), ("duration_s", "Duration (s)"),
    ("data_dir", "Data directory"),
]
_STATUS_STYLE = {"completed": "green", "running": "bold yellow", "aborted": "yellow", "error": "bold red"}


def _run_row(run: dict) -> list:
    duration = run.get("duration_s")
    cells = {k: "—" if run.get(k) in (None, "") else str(run[k]) for k, _ in RUN_COLUMNS}
    cells["duration_s"] = f"{duration:.1f}" if duration is not None else "—"
    row: list = [cells[k] for k, _ in RUN_COLUMNS]
    status_i = [k for k, _ in RUN_COLUMNS].index("status")
    row[status_i] = Text(cells["status"], style=_STATUS_STYLE.get(run.get("status"), ""))
    return row


def _card(program: Program) -> Vertical:
    return Vertical(
        Static(program.title, classes="card-title"),
        Static(program.description, classes="card-desc"),
        Collapsible(
            ScrollableContainer(Static(Text(program.schematic, no_wrap=True), classes="schematic"),
                                classes="schematic-scroll"),
            title="Wiring", collapsed=True),
        Button("▶  Launch", id=f"launch_{program.key}", variant="success"),
        classes="card",
    )


class LauncherApp(App):
    TITLE = "Bridge Measurement Suite"
    SUB_TITLE = "Choose a measurement to run"

    CSS = """
    #suites { height: 1fr; }
    .suite { width: 1fr; padding: 0 1; }
    .suite-title { text-style: bold; color: $accent; margin: 1 0; }
    .card { border: round $primary; padding: 0 1; height: auto; margin-bottom: 1; }
    .card-title { text-style: bold; margin-bottom: 1; }
    .card-desc { color: $text-muted; margin-bottom: 1; }
    .card Collapsible { margin: 0 0 1 0; padding: 0; border: none; }
    .card Collapsible > Contents { padding: 0; }
    .schematic-scroll { height: auto; max-height: 20; overflow-x: auto; }
    .schematic { width: auto; background: $surface; padding: 0 1; }
    #runs_title { text-style: bold; padding: 0 1; height: 1; }
    #runs_empty { color: $text-muted; padding: 0 1; height: 1; }
    #recent_runs { height: auto; max-height: 12; margin: 0 1; }
    """

    BINDINGS = [Binding("q", "quit", "Quit", show=True)]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="suites"):
            for suite, programs in PROGRAMS.items():
                with VerticalScroll(classes="suite"):
                    yield Static(suite, classes="suite-title")
                    for program in programs:
                        yield _card(program)
        yield Static("Recent runs", id="runs_title")
        yield Static("No runs recorded yet — this fills in as you run measurements.", id="runs_empty")
        yield DataTable(id="recent_runs", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#recent_runs", DataTable)
        for key, label in RUN_COLUMNS:
            table.add_column(label, key=key)
        self.refresh_runs()
        self.set_interval(5.0, self.refresh_runs)

    def refresh_runs(self) -> None:
        try:
            runs = run_index.recent_runs(limit=20)
        except Exception:
            log.exception("Could not read the run history")
            runs = []
        table = self.query_one("#recent_runs", DataTable)
        table.clear()
        for run in runs:
            table.add_row(*_run_row(run))
        self.query_one("#runs_empty").display = not runs
        table.display = bool(runs)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("launch_"):
            self.exit(result=event.button.id.removeprefix("launch_"))


def main() -> None:
    """Loop: pick a program, run it, come back to the menu."""
    while True:
        program = _BY_KEY.get(LauncherApp().run())
        if program is None:
            break  # user quit the menu
        program.app().run()


if __name__ == "__main__":
    main()
