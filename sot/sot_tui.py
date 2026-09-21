#!/usr/bin/env python3
"""
SOT Measurement Suite  ── single entry point for the pulsed-switching TUIs
=========================================================================
Author: Joacim Stenlund <joacim.stenlund@physics.uu.se>
Created: 2026-09-07 (restored 2026-09-21 for the four current programs)

Picks between the pulsed-switching programs in this category and shows a
wiring schematic for each. Mirrors dc/dc_tui.py and mfli/mfli_tui.py.

Run with:
    uv run python sot/sot_tui.py
"""

from __future__ import annotations

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from sot.sot_nonlocal_switching_tui import NLSW_DESCRIPTION, NonlocalSwitchingApp
from sot.sot_pulsed_switching_2h_tui import SOT_PULSED_2H_DESCRIPTION, SOTPulsedSwitching2HApp
from sot.sot_pulsed_switching_6221_tui import SOT_PULSED_6221_DESCRIPTION, SOTPulsedSwitching6221App
from sot.sot_pulsed_switching_tui import SOT_PULSED_DESCRIPTION, SOTPulsedSwitchingApp


# ─────────────────────────────────────────────────────────────────────────────
# Wiring schematics (the descriptions come from each program's own TUI module)
# ─────────────────────────────────────────────────────────────────────────────

PULSED_SCHEMATIC = """\
  KEITHLEY 4200A-SCS  (KXCI — GPIB 17)   — pulse only, FORCE triax, 2-wire local sense
    PMU1-1 ──▶ RPM1 ──▶ I+ pad of the Hall-cross main channel
                        centre = force, guard = floating, outer = circuit COMMON.
                        The KULT module bridge_sot_pulse.c routes RPM1 to the
                        pulse pathway for the burst and back on exit.

  COMMON BUS ──▶ I- pad   (PMU FORCE outer shell + 6221 output LO land here)

  KEITHLEY 6221  HI ──▶ I+ pad ,  LO ──▶ common bus   (delayed R_xy read)
                 In parallel with the PMU — OFF while pulsing.
  KEITHLEY 2182  ──▶ transverse (Hall) arms           (V_xy, floating diff)

  KEPCO BOP-GL      ──GPIB──▶ electromagnet   (ONE static tilted field)
  LAKE SHORE 475    ──GPIB──▶ Gaussmeter probe at the sample

  Cycle: 6221 OFF → PMU write pulse → wait → 6221 ON, reversal-averaged
  R_xy (6221 forces ±I, 2182 reads V_xy) → 6221 OFF. The 2182 is only the
  reader; re-run the whole sweep for statistics.
"""

PULSED_2H_SCHEMATIC = """\
  4200A PMU / RPM1 / I+ pad / common bus, 6221 HI→I+ , LO→common, and the Kepco +
  Lake Shore 475 static field: wired EXACTLY as program 1. New:

  KEITHLEY 6221  WAVE sine + phase marker (read phase only, OFF while pulsing)
    Trigger Link phase marker (pin 1) ──▶ ZURICH MFLI  AUX IN 1
  ZURICH MFLI    Signal Input (differential) ──▶ the transverse (Hall) arms
                 ExtRef-locked to the marker; reads 1f (resistive) and 2f of V_xy

  The 2182 is not used.
"""

PULSED_6221_SCHEMATIC = """\
  KEITHLEY 6221  (the ONLY source — no 4200A, no 2182)   HI ──▶ I+ pad ,  LO ──▶ I- pad
    WAVE square, ONE cycle = the write pulse (hardware-timed, per amplitude)
    then WAVE sine + phase marker = the read; wave OFF between the two.
    Trigger Link phase marker (pin 1) ──▶ ZURICH MFLI  AUX IN 1
  ZURICH MFLI    Signal Input (differential) ──▶ the transverse (Hall) arms
                 ExtRef-locked to the marker; one harmonic (2f default, 1f = AHE/PHE)

  KEPCO BOP-GL      ──GPIB──▶ electromagnet   (ONE static tilted field)
  LAKE SHORE 475    ──GPIB──▶ Gaussmeter probe at the sample

  Software-timed at the µs–ms scale: Joule heating is I²R·t, far above the
  4200A's ns pulses — start well below the switching current.
"""

NONLOCAL_SCHEMATIC = """\
  KEITHLEY 6221  (the ONLY source — no 4200A, no MFLI)
    HI ──▶ injector electrode
    LO ──▶ return electrode, a bit further from the injector, on the side away
           from the detector (the current returns through it). Triax OUTPUT LOW
           FLOATING — the program sets it.
    WAVE square, ONE cycle: 0 → ±I → 0 (one lobe, never a ± pair) = the write pulse,
    then plain DC I_sense (±I current reversal by default, switchable) = the read.
  KEITHLEY 2182A  ch1 ──▶ detector magnet electrode / reference electrode past the
                          magnet (V_NL). ch2 unused (its LO is bonded to ch1 LO).

  KEPCO BOP-GL + LAKE SHORE 475  (optional) — external-field initialization of the
    magnet state before the sweep; one run per init current (e.g. +B and -B).
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
    TITLE = "SOT Measurement Suite"
    SUB_TITLE = "Choose a pulsed-switching measurement to run"

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
                    "1) SOT pulsed switching (4200A PMU pulse + 6221/2182 R_xy)",
                    SOT_PULSED_DESCRIPTION,
                    PULSED_SCHEMATIC,
                    "launch_pulsed", "▶  Launch SOT pulsed switching TUI",
                )
                yield _card(
                    "2) SOT pulsed switching, 2nd-harmonic read (4200A PMU + 6221 AC / MFLI)",
                    SOT_PULSED_2H_DESCRIPTION,
                    PULSED_2H_SCHEMATIC,
                    "launch_pulsed_2h", "▶  Launch SOT pulsed switching (2f) TUI",
                )
                yield _card(
                    "3) SOT pulsed switching, 6221-only (6221 pulse + 6221 AC / MFLI)",
                    SOT_PULSED_6221_DESCRIPTION,
                    PULSED_6221_SCHEMATIC,
                    "launch_pulsed_6221", "▶  Launch SOT pulsed switching (6221-only) TUI",
                )
                yield _card(
                    "4) Nonlocal spin-current switching (6221 pulse + 2182A nonlocal read)",
                    NLSW_DESCRIPTION,
                    NONLOCAL_SCHEMATIC,
                    "launch_nonlocal", "▶  Launch nonlocal switching TUI",
                )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        results = {"launch_pulsed": "pulsed", "launch_pulsed_2h": "pulsed_2h",
                   "launch_pulsed_6221": "pulsed_6221", "launch_nonlocal": "nonlocal"}
        if event.button.id in results:
            self.exit(result=results[event.button.id])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── loops: pick a program, run it, return to the picker
# ─────────────────────────────────────────────────────────────────────────────

_PROGRAMS = {
    "pulsed": SOTPulsedSwitchingApp,
    "pulsed_2h": SOTPulsedSwitching2HApp,
    "pulsed_6221": SOTPulsedSwitching6221App,
    "nonlocal": NonlocalSwitchingApp,
}


def main() -> None:
    while True:
        app_cls = _PROGRAMS.get(LauncherApp().run())
        if app_cls is None:
            break  # user quit the picker
        app_cls().run()


if __name__ == "__main__":
    main()
