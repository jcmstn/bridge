#!/usr/bin/env python3
"""
MFLI Measurement Suite  ── single entry point for all MFLI TUIs
=================================================================
Picks between the available dual-MFLI measurement programs and shows a
quick wiring schematic for each, so you don't have to remember which
signal goes where before launching the full parameter form.

Run with:
    python mfli_tui.py

Requirements: same as mfli_dual_harmonic_tui.py / mfli_diff_resistance_tui.py
    pip install textual matplotlib zhinst-core zhinst-utils numpy pandas pyvisa
"""

from __future__ import annotations

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from mfli_dual_harmonic_tui import MFLIDualHarmonicApp
from mfli_diff_resistance_tui import MFLIDiffResistanceApp


# ─────────────────────────────────────────────────────────────────────────────
# Program descriptions & wiring schematics
# ─────────────────────────────────────────────────────────────────────────────

DUAL_HARMONIC_DESC = (
    "Drives an AC current through the sample and reads the 1st-harmonic "
    "response on the leader while the follower reads the 2nd-harmonic "
    "response — the standard setup for e.g. a nonlinear/planar Hall "
    "measurement. Optionally sweeps a Kepco electromagnet's field "
    "(bidirectionally, for hysteresis) with the field measured live via a "
    "Lake Shore 475 Gaussmeter at every point."
)

DUAL_HARMONIC_SCHEMATIC = """\
  LEADER MFLI  (current source, 1f)
    Signal Output 1 ──[ R_series ]──▶ sample/DUT ── common ground
    Signal Input 1  (differential)  ──▶ demod 1f   (V_Rseries → I)

  FOLLOWER MFLI  (2f)
    Signal Input 1  (differential)  ──▶ demod 2f   (across the sample)

  MDS cabling  (both units)
    Leader Ref Out      ───BNC───▶ Follower Ref In
    Leader Trigger Out 1 ──▶ fanned out to Trigger In 1 on BOTH units
    (equal cable lengths on the fan-out)

  Magnet field sweep  (optional, "Sweep magnetic field" switch)
    Kepco BOP-GL      ──GPIB──▶ electromagnet coil
    Lake Shore 475    ──GPIB──▶ Gaussmeter probe at the sample
"""

DIFF_RESISTANCE_DESC = (
    "Superimposes a small AC excitation on top of a DC bias applied to the "
    "DUT, sweeps that DC bias, and records the complex ratio dV/dI at each "
    "point — an 'I-V-curve-equivalent' characterization far more informative "
    "than a single-point resistance for anything nonlinear (contacts, "
    "tunnel junctions, diodes, gated 2D systems). No magnet is involved; "
    "the bias sweep is the whole measurement."
)

DIFF_RESISTANCE_SCHEMATIC = """\
  LEADER MFLI  (AC excitation + DC bias, I-sense)
    Signal Output 1 ──[ R_series ]──┬── DUT ── common ground
                                     │
    Signal Input 1 (differential) ──┘   (measures V_Rseries → I)
    (DC bias is summed onto this SAME output — no separate bias wire)

  FOLLOWER MFLI  (V-sense)
    Signal Input 1 (differential) ──── across the DUT itself
                                        (2-terminal), or across the inner
                                        voltage-sense leads (4-terminal /
                                        Kelvin)

  MDS cabling  (both units)
    Leader Ref Out      ───BNC───▶ Follower Ref In
    Leader Trigger Out 1 ──▶ fanned out to Trigger In 1 on BOTH units
    (equal cable lengths on the fan-out)
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
    TITLE = "MFLI Measurement Suite"
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
                "1) Dual-Harmonic Measurement (1f / 2f)",
                DUAL_HARMONIC_DESC,
                DUAL_HARMONIC_SCHEMATIC,
                "launch_dual", "▶  Launch dual-harmonic TUI",
            )
            yield _card(
                "2) Differential Resistance vs. Bias (dV/dI)",
                DIFF_RESISTANCE_DESC,
                DIFF_RESISTANCE_SCHEMATIC,
                "launch_diff", "▶  Launch differential-resistance TUI",
            )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "launch_dual":
            self.exit(result="dual")
        elif event.button.id == "launch_diff":
            self.exit(result="diff")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  ── loops: pick a program, run it, return to the picker
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    while True:
        mode = LauncherApp().run()
        if mode == "dual":
            MFLIDualHarmonicApp().run()
        elif mode == "diff":
            MFLIDiffResistanceApp().run()
        else:
            break  # user quit the picker


if __name__ == "__main__":
    main()
