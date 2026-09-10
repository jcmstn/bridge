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
from sot.sot_pulsed_switching_tui import SOT_PULSED_DESCRIPTION, SOTPulsedSwitchingApp
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
  together. Quasi-static DC only — for pulsed, see program 3.
"""

PULSED_SCHEMATIC = """\
  KEITHLEY 4200A-SCS  (KXCI — GPIB 17)   — pulse only, FORCE triax, 2-wire local sense
    PMU1-1 ──▶ RPM1 ──▶ I+ pad of the Hall-cross main channel
                        centre = force, guard = floating, outer = circuit COMMON.
                        The KULT module bridge_sot_pulse.c routes RPM1 to the
                        pulse pathway for the burst and back on exit — that
                        lifts the PMU's 50 Ω out of the 6221 read path.

  COMMON BUS ──▶ I- pad   (PMU FORCE outer shell + 6221 output LO land here)

  KEITHLEY 6221  HI ──▶ I+ pad ,  LO ──▶ common bus   (delayed R_xy read)
                 In parallel with the PMU — OFF while pulsing.
  KEITHLEY 2182  ──▶ transverse (Hall) arms           (V_xy, floating diff)

  The two 4200A SMUs and the preamps are unused.

  KEPCO BOP-GL      ──GPIB──▶ electromagnet   (ONE static tilted field)
  LAKE SHORE 475    ──GPIB──▶ Gaussmeter probe at the sample

  Cycle: 6221 OFF → [reset pulse] → write pulse → wait (e.g. 5 s) →
  6221 ON, reversal-averaged R_xy (6221 forces ±I, 2182 reads V_xy) → 6221 OFF.
  Re-run at ∓field for the ±H_z control.
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
                yield _card(
                    "3) SOT pulsed switching (PMU pulse + delayed R_xy)",
                    SOT_PULSED_DESCRIPTION,
                    PULSED_SCHEMATIC,
                    "launch_pulsed", "▶  Launch SOT pulsed switching TUI",
                )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "launch_dcchar":
            self.exit(result="dcchar")
        elif event.button.id == "launch_switching":
            self.exit(result="switching")
        elif event.button.id == "launch_pulsed":
            self.exit(result="pulsed")


def main() -> None:
    while True:
        mode = LauncherApp().run()
        if mode == "dcchar":
            SOTDCCharApp().run()
        elif mode == "switching":
            SOTSwitchingApp().run()
        elif mode == "pulsed":
            SOTPulsedSwitchingApp().run()
        else:
            break


if __name__ == "__main__":
    main()
