"""
Every TUI RunScreen keeps its progress row one line tall.

`#progress_row` is a Horizontal, whose default height is `1fr`; without an
explicit `height: auto` it swallows the free vertical space and shoves the
results table and log far down the screen (fixed in the MFLI/SOT TUIs in
a300c6f, then again for the DC TUIs). Pure CSS-string check -- no app loop.
"""

from __future__ import annotations

import importlib

import pytest

RUN_SCREEN_MODULES = [
    "dc.dc_gate_sweep_tui",
    "dc.dc_hall_measurement_tui",
    "dc.dc_iv_curve_tui",
    "dc.dc_spin_valve_tui",
    "mfli.mfli_diff_resistance_tui",
    "mfli.mfli_dual_harmonic_6221_tui",
    "mfli.mfli_dual_harmonic_tui",
    "mfli.mfli_noise_spectrum_tui",
    "mfli.mfli_phase_calibration_tui",
    "sot.nonlocal_switching_tui",
    "sot.sot_pulsed_switching_2h_tui",
    "sot.sot_pulsed_switching_6221_tui",
    "sot.sot_pulsed_switching_tui",
]


@pytest.mark.parametrize("module", RUN_SCREEN_MODULES)
def test_progress_row_height_auto(module):
    css = importlib.import_module(module).RunScreen.CSS
    rule = next(line for line in css.splitlines() if "#progress_row" in line)
    assert "height: auto" in rule
