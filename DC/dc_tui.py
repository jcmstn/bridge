#!/usr/bin/env python3
"""
DC Measurement Suite  ── single entry point for the Keithley 6221/2182/2400 TUIs
==================================================================================
Same idea as bridge/MFLI/mfli_tui.py, one folder over: picks between the
available DC (Keithley 6221 current source + 2182 nanovoltmeter, plus an
optional/required Keithley 2400 gate depending on the program) measurement
programs and shows a quick wiring schematic for each, so you don't have to
remember which lead goes where before launching the full parameter form.

Run with:
    python dc_tui.py

Requirements: same as the individual *_tui.py modules
    pip install textual matplotlib pymeasure pyvisa numpy pandas
"""

from __future__ import annotations

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from dc_hall_measurement_tui import DC_HALL_DESCRIPTION, DCHallMeasurementApp
from dc_iv_curve_tui import DC_IV_DESCRIPTION, DCIVCurveApp
from dc_gate_sweep_tui import DC_GATE_SWEEP_DESCRIPTION, DCGateSweepApp
from dc_spin_valve_tui import DC_SPIN_VALVE_DESCRIPTION, DCSpinValveApp


# ─────────────────────────────────────────────────────────────────────────────
# Wiring schematics  ── descriptions are imported from each program's own TUI
# module (single source of truth — also shown in that program's sidebar)
# ─────────────────────────────────────────────────────────────────────────────

HALL_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source)
    Output ──▶ sample ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ transverse (Hall) voltage leads

  Magnet field sweep  (optional, "Sweep magnetic field" switch)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample
"""

IV_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source)
    Output ──▶ DUT ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ across the DUT itself (2-terminal), or
                                   across the inner voltage-sense leads
                                   (4-terminal / Kelvin)

  Gate voltage  (optional, "Enable gate" switch)
    KEITHLEY 2400 (gate source) ──▶ gate electrode
"""

GATE_SWEEP_SCHEMATIC = """\
  KEITHLEY 6221  (fixed DC sense current)
    Output ──▶ DUT ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ across the DUT

  KEITHLEY 2400  (gate source — the swept axis)
    Output ──▶ gate electrode

  Field  (optional, single value or list — parked, not swept)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample
"""

SPIN_VALVE_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source)
    Output ──▶ sample ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ longitudinal voltage leads

  KEITHLEY 2400  (gate source — fixed per sweep, single value or list)
    Output ──▶ gate electrode

  Magnet field sweep  (the swept axis)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample
"""


# ─────────────────────────────────────────────────────────────────────────────
# Picker screen
# ─────────────────────────────────────────────────────────────────────────────

def _card(title: str, description: str, schematic: str, button_id: str,
          button_label: str) -> Vertical:
    return Vertical(
        Static(title, classes="card-title"),
        Static(description, classes="card-desc"),
        Static("Wiring", classes="schematic-title"),
        Static(Text(schematic, no_wrap=True), classes="schematic"),
        Button(button_label, id=button_id, variant="success"),
        classes="card",
    )


class LauncherApp(App):
    TITLE = "DC Measurement Suite"
    SUB_TITLE = "Choose a measurement to run"

    CSS = """
    #picker { padding: 1 2; }
    .intro { margin-bottom: 1; text-style: bold; }
    .card { border: solid $primary; padding: 1 2; margin-bottom: 2; height: auto; }
    .card-title { text-style: bold underline; margin-bottom: 1; }
    .card-desc { color: $text-muted; margin-bottom: 1; }
    .schematic-title { text-style: bold; margin-bottom: 1; }
    .schematic { background: $surface; border: round $accent; padding: 1 2;
                 margin-bottom: 1; width: auto; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll(id="picker"):
            yield Static("Select a measurement to run:", classes="intro")
            yield _card(
                "1) Hall Measurement (DC, field sweep)",
                DC_HALL_DESCRIPTION,
                HALL_SCHEMATIC,
                "launch_hall", "▶  Launch Hall measurement TUI",
            )
            yield _card(
                "2) I-V Curve (DC current sweep, optional gate)",
                DC_IV_DESCRIPTION,
                IV_SCHEMATIC,
                "launch_iv", "▶  Launch I-V curve TUI",
            )
            yield _card(
                "3) Gate Sweep (gate voltage sweep, optional field)",
                DC_GATE_SWEEP_DESCRIPTION,
                GATE_SWEEP_SCHEMATIC,
                "launch_gate_sweep", "▶  Launch Gate Sweep TUI",
            )
            yield _card(
                "4) Spin-Valve / Field Sweep (fixed gate, field sweep)",
                DC_SPIN_VALVE_DESCRIPTION,
                SPIN_VALVE_SCHEMATIC,
                "launch_spin_valve", "▶  Launch Spin-Valve TUI",
            )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "launch_hall":
            self.exit(result="hall")
        elif event.button.id == "launch_iv":
            self.exit(result="iv")
        elif event.button.id == "launch_gate_sweep":
            self.exit(result="gate_sweep")
        elif event.button.id == "launch_spin_valve":
            self.exit(result="spin_valve")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── loops: pick a program, run it, return to the picker
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    while True:
        mode = LauncherApp().run()
        if mode == "hall":
            DCHallMeasurementApp().run()
        elif mode == "iv":
            DCIVCurveApp().run()
        elif mode == "gate_sweep":
            DCGateSweepApp().run()
        elif mode == "spin_valve":
            DCSpinValveApp().run()
        else:
            break  # user quit the picker


if __name__ == "__main__":
    main()
