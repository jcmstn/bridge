#!/usr/bin/env python3
"""
DC Measurement Suite  ── single entry point for the Keithley 6221/2182 TUIs
==============================================================================
Same idea as bridge/MFLI/mfli_tui.py, one folder over: picks between the
available DC (Keithley 6221 current source + 2182 nanovoltmeter)
measurement programs and shows a quick wiring schematic for each, so you
don't have to remember which lead goes where before launching the full
parameter form.

Run with:
    python dc_tui.py

Requirements: same as dc_hall_measurement_tui.py / dc_iv_curve_tui.py
    pip install textual matplotlib pymeasure pyvisa numpy pandas
"""

from __future__ import annotations

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from dc_hall_measurement_tui import DCHallMeasurementApp
from dc_iv_curve_tui import DCIVCurveApp


# ─────────────────────────────────────────────────────────────────────────────
# Program descriptions & wiring schematics
# ─────────────────────────────────────────────────────────────────────────────

HALL_DESC = (
    "Sources a fixed DC sense current with a Keithley 6221 and reads the "
    "transverse (Hall) voltage with a Keithley 2182, reversing the current "
    "each rep to cancel thermal-EMF offsets. Optionally sweeps a Kepco "
    "electromagnet's field (bidirectionally, for hysteresis) with the "
    "field measured live via a Lake Shore 475 Gaussmeter at every point — "
    "the standard DC (non-lock-in) Hall-effect measurement."
)

HALL_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source)
    Output ──▶ sample ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ transverse (Hall) voltage leads

  Magnet field sweep  (optional, "Sweep magnetic field" switch)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample
"""

IV_DESC = (
    "Sweeps a DC current with a Keithley 6221 and records the DC voltage "
    "response with a Keithley 2182 at each point — a direct I-V curve, "
    "swept bidirectionally so hysteresis is visible. Far more informative "
    "than a single-point resistance for anything nonlinear (contacts, "
    "tunnel junctions, diodes, gated 2D systems). No magnet is involved; "
    "the current sweep is the whole measurement."
)

IV_SCHEMATIC = """\
  KEITHLEY 6221  (DC current source)
    Output ──▶ DUT ── common ground

  KEITHLEY 2182  (nanovoltmeter)
    Channel 1 (differential) ──▶ across the DUT itself (2-terminal), or
                                   across the inner voltage-sense leads
                                   (4-terminal / Kelvin)
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
                HALL_DESC,
                HALL_SCHEMATIC,
                "launch_hall", "▶  Launch Hall measurement TUI",
            )
            yield _card(
                "2) I-V Curve (DC current sweep)",
                IV_DESC,
                IV_SCHEMATIC,
                "launch_iv", "▶  Launch I-V curve TUI",
            )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "launch_hall":
            self.exit(result="hall")
        elif event.button.id == "launch_iv":
            self.exit(result="iv")


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
        else:
            break  # user quit the picker


if __name__ == "__main__":
    main()
