#!/usr/bin/env python3
"""
SOT Measurement Suite  ── entry point for the Keithley 4200A SOT TUIs
====================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-07

Picks between the two 4200A programs in this category and shows a wiring
schematic for each. Mirrors dc/dc_tui.py.

Run with:
    python sot_tui.py
"""

from __future__ import annotations

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from sot.sot_dc_characterization_tui import SOT_DCCHAR_DESCRIPTION, SOTDCCharApp
from sot.sot_switching_tui import SOT_SWITCHING_DESCRIPTION, SOTSwitchingApp


DCCHAR_SCHEMATIC = """\
  KEITHLEY 4200A-SCS  (KXCI — GPIB or LAN)
    SMU1  ──▶ outer current leads of the Hall bar   (forces I)
    SMU2  ──▶ inner longitudinal voltage leads       (forces 0 A, reads V)

  True 4-probe: no lead resistance, no KXCI remote-sense toggle needed.
  Current reversal (+I/-I) cancels the thermal-EMF offset.
"""

SWITCHING_SCHEMATIC = """\
  KEITHLEY 4200A-SCS  (KXCI — GPIB or LAN)
    SMU1  ──▶ channel current leads   (the switching-current staircase)
    SMU2  ──▶ transverse (Hall) leads  (forces 0 A, reads V_xy)

  KEPCO BOP-GL      ──GPIB──▶ electromagnet coil   (assist field, one file
                                                    per magnet-current value)
  LAKE SHORE 475    ──GPIB──▶ Gaussmeter probe at the sample

  Single tilted magnet: the mount tilt (field_angle_from_oop_deg) is
  recorded, not set; flipping the magnet-current sign flips Hx and Hz
  together. PMU-pulsed switching (Stage 4) is a planned follow-up.
"""


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
    TITLE = "SOT Measurement Suite"
    SUB_TITLE = "Choose a Keithley 4200A measurement to run"

    CSS = """
    #picker { padding: 1 2; }
    .intro { margin-bottom: 1; text-style: bold; }
    .picker-grid { layout: grid; grid-size: 2; grid-gutter: 1 2; height: auto; }
    .card { border: solid $primary; padding: 1 2; height: auto; }
    .card-title { text-style: bold underline; margin-bottom: 1; }
    .card-desc { color: $text-muted; margin-bottom: 1; }
    .schematic-title { text-style: bold; margin-bottom: 1; }
    .schematic { background: $surface; border: round $accent; padding: 1 2;
                 margin-bottom: 1; width: auto; }
    """

    BINDINGS = [Binding("q", "quit", "Quit", show=True)]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll(id="picker"):
            yield Static("Select a measurement to run:", classes="intro")
            with Vertical(classes="picker-grid"):
                yield _card(
                    "1) DC characterisation (4-probe R_xx)",
                    SOT_DCCHAR_DESCRIPTION,
                    DCCHAR_SCHEMATIC,
                    "launch_dcchar", "▶  Launch DC characterisation TUI",
                )
                yield _card(
                    "2) SOT switching (DC current-staircase loops)",
                    SOT_SWITCHING_DESCRIPTION,
                    SWITCHING_SCHEMATIC,
                    "launch_switching", "▶  Launch SOT switching TUI",
                )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "launch_dcchar":
            self.exit(result="dcchar")
        elif event.button.id == "launch_switching":
            self.exit(result="switching")


def main() -> None:
    while True:
        mode = LauncherApp().run()
        if mode == "dcchar":
            SOTDCCharApp().run()
        elif mode == "switching":
            SOTSwitchingApp().run()
        else:
            break


if __name__ == "__main__":
    main()
